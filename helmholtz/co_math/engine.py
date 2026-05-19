from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents import AGENTS, AgentContext, ReviewerPanel
from .core import ProjectStore, Workspace, now_iso
from .llm import OptionalLLM
from .retrieval import LocalContextIndex


class CoMathematician:
    def __init__(self, workspace: Workspace, llm: Optional[OptionalLLM] = None):
        self.workspace = workspace
        self.llm = llm or OptionalLLM()
        self.reviewers = ReviewerPanel()

    def run(self, project_ref: str, rounds: int = 1, max_review_rounds: int = 2) -> ProjectStore:
        project = self.workspace.load_project(project_ref)
        for _ in range(max(1, rounds)):
            made_progress = False
            index = LocalContextIndex.from_paths(project.context_paths())
            context = AgentContext(project=project, index=index, llm=self.llm)
            for workstream in project.data["workstreams"]:
                if workstream["status"] not in {"queued", "needs_revision"}:
                    continue
                if workstream["kind"] == "review" and not self._review_has_inputs(project, workstream):
                    continue
                self._advance_workstream(context, workstream, max_review_rounds=max_review_rounds)
                made_progress = True
            self._refresh_project_status(project)
            self._write_working_paper(project)
            project.save()
            if not made_progress:
                break
        return project

    def steer(self, project_ref: str, message: str) -> ProjectStore:
        project = self.workspace.load_project(project_ref)
        steering = {"time": now_iso(), "message": message}
        project.data.setdefault("steering", []).append(steering)
        project.append_message("user", message)
        project.append_event("human_steering", message=message)
        for workstream in project.data["workstreams"]:
            if workstream["status"] == "blocked":
                workstream["status"] = "queued"
                workstream["updated_at"] = now_iso()
                workstream.setdefault("uncertainty", []).append("Reopened after human steering.")
        self._write_working_paper(project)
        project.save()
        return project

    def status(self, project_ref: str, details: bool = False) -> str:
        project = self.workspace.load_project(project_ref)
        lines = [
            f"Project: {project.data['id']}",
            f"Title: {project.data['title']}",
            f"Status: {project.data['status']}",
            f"Question: {project.data['question']}",
            "",
            "Workstreams:",
        ]
        for workstream in project.data["workstreams"]:
            lines.append(
                f"- {workstream['id']} [{workstream['kind']}]: {workstream['status']} "
                f"(review rounds: {workstream['review_rounds']})"
            )
            if details:
                if workstream.get("report_path"):
                    lines.append(f"  report: {workstream['report_path']}")
                for item in workstream.get("uncertainty", [])[-3:]:
                    lines.append(f"  uncertainty: {item}")
        if details:
            lines.extend(["", f"Working paper: {project.working_paper_file}"])
            if project.failed_explorations():
                lines.append("Failed explorations:")
                for item in project.failed_explorations():
                    lines.append(f"- {item['workstream_id']}: {item['reason']}")
        return "\n".join(lines)

    def export_latex(self, project_ref: str, output: Optional[str] = None) -> Path:
        project = self.workspace.load_project(project_ref)
        target = Path(output).expanduser() if output else project.path / "working_paper.tex"
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._latex_from_project(project), encoding="utf-8")
        project.append_event("exported_latex", output=str(target))
        project.save()
        return target

    def _advance_workstream(self, context: AgentContext, workstream: Dict[str, Any], max_review_rounds: int) -> None:
        agent = AGENTS.get(workstream["kind"], AGENTS["theory"])
        product = agent.run(context, workstream)
        workdir = context.project.workstream_dir(workstream["id"])
        report_path = workdir / "report.md"
        reviews = self.reviewers.review(workstream, product.report)
        approved = self.reviewers.approved(reviews)
        report = self._append_review_record(product.report, reviews, approved)
        report_path.write_text(report, encoding="utf-8")

        workstream["report_path"] = str(report_path)
        workstream["artifact_paths"] = [str(path) for path in product.artifact_paths]
        workstream["review_rounds"] = int(workstream.get("review_rounds", 0)) + 1
        workstream["review_state"] = reviews
        workstream["uncertainty"] = product.uncertainty
        workstream["updated_at"] = now_iso()

        if approved:
            workstream["status"] = "complete"
            context.project.append_event("workstream_completed", workstream_id=workstream["id"], report=str(report_path))
        elif workstream["review_rounds"] >= max_review_rounds:
            reason = "; ".join(review["comment"] for review in reviews if not review["approved"])
            workstream["status"] = "blocked"
            context.project.append_failed_exploration(
                workstream["id"],
                reason=reason or "reviewers did not approve the workstream",
                evidence=[str(report_path)],
            )
            context.project.append_event("workstream_blocked", workstream_id=workstream["id"], reason=reason)
        else:
            workstream["status"] = "needs_revision"
            context.project.append_event("workstream_needs_revision", workstream_id=workstream["id"])

    @staticmethod
    def _append_review_record(report: str, reviews: List[Dict[str, Any]], approved: bool) -> str:
        rows = ["\n## Review Record\n"]
        for review in reviews:
            mark = "approved" if review["approved"] else "rejected"
            rows.append(f"- `{review['reviewer']}` {mark}: {review['comment']}")
        rows.append(f"\nFinal review gate: `{'approved' if approved else 'not approved'}`\n")
        return report.rstrip() + "\n" + "\n".join(rows)

    @staticmethod
    def _review_has_inputs(project: ProjectStore, current: Dict[str, Any]) -> bool:
        return any(row["id"] != current["id"] and row["status"] in {"complete", "blocked"} for row in project.data["workstreams"])

    @staticmethod
    def _refresh_project_status(project: ProjectStore) -> None:
        statuses = {workstream["status"] for workstream in project.data["workstreams"]}
        if statuses <= {"complete"}:
            project.data["status"] = "complete"
        elif "blocked" in statuses:
            project.data["status"] = "needs_human_attention"
        else:
            project.data["status"] = "active"

    def _write_working_paper(self, project: ProjectStore) -> None:
        project.working_paper_file.write_text(self._render_working_paper(project), encoding="utf-8")

    @staticmethod
    def _render_working_paper(project: ProjectStore) -> str:
        lines = [
            f"# Working Paper: {project.data['title']}",
            "",
            f"Project id: `{project.data['id']}`  ",
            f"Status: `{project.data['status']}`  ",
            f"Paper design reference: <{project.data.get('paper_source', '')}>",
            "",
            "## Research Question",
            "",
            project.data["question"],
            "",
            "## Goals",
            "",
        ]
        for goal in project.data["goals"]:
            status = ", ".join(
                f"{workstream['id']}={workstream['status']}"
                for workstream in project.data["workstreams"]
                if workstream["goal_id"] == goal["id"]
            )
            lines.append(f"- **{goal['title']}** (`{goal['id']}`): {goal['statement']} [{status}]")

        lines.extend(["", "## Coordinator Notes", ""])
        steering = project.data.get("steering", [])
        if steering:
            for item in steering[-5:]:
                lines.append(f"- Human steering at {item['time']}: {item['message']}")
        else:
            lines.append("- No human steering after initialization yet.")

        lines.extend(["", "## Workstream Summaries", ""])
        for workstream in project.data["workstreams"]:
            lines.append(f"### {workstream['id']}: {workstream['title']}")
            lines.append("")
            lines.append(f"- Status: `{workstream['status']}`")
            lines.append(f"- Kind: `{workstream['kind']}`")
            if workstream.get("report_path"):
                lines.append(f"- Report: `{workstream['report_path']}`")
            if workstream.get("artifact_paths"):
                lines.append("- Artifacts: " + ", ".join(f"`{path}`" for path in workstream["artifact_paths"]))
            if workstream.get("review_state"):
                approved = all(row["approved"] for row in workstream["review_state"])
                lines.append(f"- Latest review gate: `{'approved' if approved else 'not approved'}`")
            if workstream.get("uncertainty"):
                lines.append("- Margin notes:")
                for item in workstream["uncertainty"]:
                    lines.append(f"  - {item}")
            lines.append("")

        failures = project.failed_explorations()
        lines.extend(["## Failed Explorations", ""])
        if not failures:
            lines.append("- None recorded yet.")
        else:
            for item in failures:
                lines.append(f"- `{item['workstream_id']}` at {item['time']}: {item['reason']}")

        lines.extend(
            [
                "",
                "## Audit Trail",
                "",
                f"- Project state: `{project.project_file}`",
                f"- Messages: `{project.messages_file}`",
                f"- Events: `{project.events_file}`",
                f"- Failed explorations: `{project.failed_file}`",
                "",
                "> Margin note: This document is a working artifact. Approval means the prototype's process checks passed, not that a theorem has been formally verified.",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _latex_from_project(project: ProjectStore) -> str:
        def esc(value: str) -> str:
            replacements = {
                "\\": r"\textbackslash{}",
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
            }
            for src, dst in replacements.items():
                value = value.replace(src, dst)
            return value

        sections = [
            r"\documentclass{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{hyperref}",
            r"\usepackage{marginnote}",
            r"\begin{document}",
            f"\\title{{{esc(project.data['title'])}}}",
            r"\author{AI co-mathematician prototype}",
            r"\maketitle",
            r"\section*{Research Question}",
            esc(project.data["question"]),
            r"\section*{Goals}",
            r"\begin{itemize}",
        ]
        for goal in project.data["goals"]:
            sections.append(f"\\item \\textbf{{{esc(goal['title'])}}}: {esc(goal['statement'])}")
        sections.extend([r"\end{itemize}", r"\section*{Workstream Status}"])
        for workstream in project.data["workstreams"]:
            sections.append(f"\\subsection*{{{esc(workstream['id'] + ': ' + workstream['title'])}}}")
            sections.append(f"Status: \\texttt{{{esc(workstream['status'])}}}.")
            if workstream.get("uncertainty"):
                sections.append(r"\marginnote{Unresolved uncertainty recorded in project workspace.}")
                sections.append(r"\begin{itemize}")
                for item in workstream["uncertainty"]:
                    sections.append(f"\\item {esc(item)}")
                sections.append(r"\end{itemize}")
        sections.extend(
            [
                r"\section*{Audit Trail}",
                "The full audit trail is stored beside this export in the project workspace: "
                r"\texttt{project.json}, \texttt{messages.jsonl}, \texttt{events.jsonl}, "
                r"and \texttt{failed\_explorations.jsonl}.",
                r"\end{document}",
                "",
            ]
        )
        return "\n".join(sections)
