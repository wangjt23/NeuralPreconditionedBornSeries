from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import ProjectStore, now_iso
from .llm import OptionalLLM
from .retrieval import DocumentChunk, LocalContextIndex, tokens


@dataclass
class AgentContext:
    project: ProjectStore
    index: LocalContextIndex
    llm: OptionalLLM


@dataclass
class WorkProduct:
    report: str
    uncertainty: List[str]
    artifact_paths: List[Path]


def _goal_header(project: ProjectStore, workstream: Dict[str, Any]) -> str:
    return (
        f"# {workstream['title']}\n\n"
        f"**Workstream:** `{workstream['id']}`  \n"
        f"**Kind:** `{workstream['kind']}`  \n"
        f"**Claim status:** working material, not a final proof  \n"
        f"**Research question:** {project.data['question']}\n\n"
    )


def _short_snippet(chunk: DocumentChunk, limit: int = 420) -> str:
    compact = chunk.compact_text
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _top_terms(text: str, count: int = 8) -> List[str]:
    values = tokens(text)
    scores: Dict[str, int] = {}
    for value in values:
        scores[value] = scores.get(value, 0) + 1
    return [term for term, _ in sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:count]]


def _format_hits(hits: List[tuple[DocumentChunk, float]]) -> str:
    if not hits:
        return (
            "- No matching local note was found. Treat any literature claim as unverified until an "
            "external search or uploaded source supports it.\n"
        )
    rows = []
    for chunk, score in hits:
        rows.append(
            f"- source: `{chunk.citation}` (score {score:.2f})\n"
            f"  > {_short_snippet(chunk)}"
        )
    return "\n".join(rows) + "\n"


def _llm_synthesis(context: AgentContext, system: str, prompt: str) -> str:
    result = context.llm.complete(system=system, prompt=prompt)
    if not result:
        return ""
    return (
        "\n## Optional Model Synthesis\n\n"
        f"Provider: `{result.provider}`\n\n"
        f"{result.text.strip()}\n"
    )


class BaseAgent:
    name = "base"

    def run(self, context: AgentContext, workstream: Dict[str, Any]) -> WorkProduct:
        raise NotImplementedError


class LiteratureAgent(BaseAgent):
    name = "literature"

    def run(self, context: AgentContext, workstream: Dict[str, Any]) -> WorkProduct:
        query = f"{context.project.data['question']} {workstream['statement']}"
        hits = context.index.search(query, k=8)
        uncertainty = []
        if not hits:
            uncertainty.append("No supporting local literature/context hit was found.")
        terms = ", ".join(_top_terms(query))
        report = (
            _goal_header(context.project, workstream)
            + "## Local Context Findings\n\n"
            + _format_hits(hits)
            + "\n## Synthesis\n\n"
            + "- The local corpus search should be treated as a first pass, not a complete literature review.\n"
            + f"- Search terms emphasized by the coordinator: `{terms}`.\n"
            + "- Any theorem imported into a proof workstream should be copied with its exact hypotheses.\n"
            + "- Missing hypotheses and notation drift are the main risks to track in later workstreams.\n"
            + "\n## Known Gaps\n\n"
            + "- External databases such as arXiv, MathSciNet, zbMATH, and journal pages are not queried by this local prototype.\n"
            + "- Snippets are retrieved by lexical overlap; absence of a hit is not evidence of absence.\n"
            + "\n## Provenance\n\n"
            + "- source: user question recorded in `messages.jsonl`.\n"
            + "- source: local context files listed in `project.json`.\n"
        )
        report += _llm_synthesis(
            context,
            "You are a mathematical literature assistant. Be cautious and cite exact sources.",
            f"Question:\n{context.project.data['question']}\n\nLocal hits:\n{_format_hits(hits)}",
        )
        return WorkProduct(report=report, uncertainty=uncertainty, artifact_paths=[])


class TheoryAgent(BaseAgent):
    name = "theory"

    def run(self, context: AgentContext, workstream: Dict[str, Any]) -> WorkProduct:
        query = f"{context.project.data['question']} {workstream['statement']}"
        hits = context.index.search(query, k=5)
        terms = _top_terms(query)
        steering = context.project.data.get("steering", [])[-3:]
        steering_rows = "\n".join(f"- {item['time']}: {item['message']}" for item in steering) or "- No steering message yet."
        candidate_terms = ", ".join(f"`{term}`" for term in terms[:6]) or "`problem`"
        uncertainty = [
            "Proof strategies are provisional until each imported lemma is checked with exact hypotheses.",
            "Counterexample search has not been exhausted.",
        ]
        report = (
            _goal_header(context.project, workstream)
            + "## Refined Mathematical Task\n\n"
            + f"- Preserve the central question: {context.project.data['question']}\n"
            + f"- Terms currently carrying the load of the problem: {candidate_terms}.\n"
            + "- Before proving anything, force every theorem statement to name its domain, quantifiers, and exceptional cases.\n"
            + "\n## Candidate Work Routes\n\n"
            + "1. **Definition-first route.** Rewrite the problem as a sequence of precise definitions and only then state lemmas.\n"
            + "2. **Extremal-example route.** Search for boundary cases where the expected statement almost fails.\n"
            + "3. **Reduction route.** Try to reduce the question to a local lemma already present in the notes or literature.\n"
            + "\n## Obstacle Log\n\n"
            + "- Do not accept a proof step that changes the meaning of a norm, topology, group action, boundary condition, or limiting process.\n"
            + "- Do not treat numerical evidence as a proof unless a separate certified bound is produced.\n"
            + "- If two strategies disagree, preserve both branches and escalate to the coordinator instead of forcing consensus.\n"
            + "\n## Local Evidence To Audit\n\n"
            + _format_hits(hits)
            + "\n## Human Steering Incorporated\n\n"
            + steering_rows
            + "\n\n## Known Gaps\n\n"
            + "- No theorem in this report is certified as proved.\n"
            + "- The next proof-producing workstream should attach exact lemma statements and a line-by-line dependency map.\n"
            + "\n## Provenance\n\n"
            + "- source: user question recorded in `messages.jsonl`.\n"
            + "- source: local context hits above.\n"
        )
        report += _llm_synthesis(
            context,
            "You are a cautious research mathematician. Separate proof ideas from proved claims.",
            f"Question:\n{context.project.data['question']}\n\nGoal:\n{workstream['statement']}\n\nRelevant snippets:\n{_format_hits(hits)}",
        )
        return WorkProduct(report=report, uncertainty=uncertainty, artifact_paths=[])


class ComputationAgent(BaseAgent):
    name = "computation"

    def run(self, context: AgentContext, workstream: Dict[str, Any]) -> WorkProduct:
        workdir = context.project.workstream_dir(workstream["id"])
        artifact = workdir / "probe_plan.py"
        query = f"{context.project.data['question']} {workstream['statement']}"
        terms = _top_terms(query, count=10)
        artifact.write_text(self._probe_source(context.project.data["question"], terms), encoding="utf-8")
        hits = context.index.search(query, k=4)
        uncertainty = [
            "Probe code is a scaffold; mathematical certification requires problem-specific invariants.",
            "The generated probes should be reviewed before any result is cited.",
        ]
        report = (
            _goal_header(context.project, workstream)
            + "## Computational Probe Design\n\n"
            + f"- Artifact: `{artifact}`\n"
            + "- The scaffold records terms, hypotheses, observations, and reviewer-visible invariants.\n"
            + "- It deliberately refuses to label a run as certified unless a human or prover supplies a certificate predicate.\n"
            + "\n## Suggested Experiments\n\n"
            + "1. Construct small examples that isolate each definition in the problem.\n"
            + "2. Add adversarial examples for boundary cases and degenerate parameter choices.\n"
            + "3. Convert any observed pattern into a conjecture only after recording the sample range.\n"
            + "\n## Local Evidence To Audit\n\n"
            + _format_hits(hits)
            + "\n## Known Gaps\n\n"
            + "- No computational result has been executed or certified by this report alone.\n"
            + "- The scaffold needs domain-specific generators before it can falsify or support conjectures.\n"
            + "\n## Provenance\n\n"
            + "- source: generated artifact `probe_plan.py`.\n"
            + "- source: user question recorded in `messages.jsonl`.\n"
        )
        return WorkProduct(report=report, uncertainty=uncertainty, artifact_paths=[artifact])

    @staticmethod
    def _probe_source(question: str, terms: List[str]) -> str:
        encoded_terms = json.dumps(terms, indent=2)
        encoded_question = json.dumps(question)
        return f'''"""Computational probe scaffold for a co-mathematician workstream.

This file is intentionally conservative: it records observations and refuses
to mark them certified unless a certificate predicate is supplied.
"""

QUESTION = {encoded_question}
TERMS = {encoded_terms}


def record_observation(name, parameters, outcome, certified=False):
    return {{
        "name": name,
        "parameters": dict(parameters),
        "outcome": outcome,
        "certified": bool(certified),
    }}


def require_certificate(observation, predicate):
    if not predicate(observation):
        raise ValueError("observation did not satisfy the supplied certificate predicate")
    observation = dict(observation)
    observation["certified"] = True
    return observation


if __name__ == "__main__":
    example = record_observation(
        "placeholder-boundary-case",
        {{"terms": TERMS[:5]}},
        "replace with a domain-specific computation",
    )
    print(example)
'''


class MetaReviewAgent(BaseAgent):
    name = "review"

    def run(self, context: AgentContext, workstream: Dict[str, Any]) -> WorkProduct:
        completed = [row for row in context.project.data["workstreams"] if row["status"] == "complete"]
        blocked = [row for row in context.project.data["workstreams"] if row["status"] == "blocked"]
        rows = []
        for row in completed + blocked:
            status = row["status"]
            path = row.get("report_path") or "(no report)"
            rows.append(f"- `{row['id']}` ({status}): `{path}`")
        if not rows:
            rows.append("- No completed or blocked workstreams are available yet.")
        uncertainty = []
        if blocked:
            uncertainty.append("At least one workstream is blocked and should be reviewed by the user.")
        report = (
            _goal_header(context.project, workstream)
            + "## Audit Summary\n\n"
            + "\n".join(rows)
            + "\n\n## Synthesis For Human Review\n\n"
            + "- Treat complete workstreams as reviewed working notes, not final mathematical truth.\n"
            + "- Inspect every claim that depends on an external theorem, generated code, or unstated regularity condition.\n"
            + "- Blocked workstreams are preserved as useful negative information rather than hidden failures.\n"
            + "\n## Known Gaps\n\n"
            + "- This meta-review cannot replace expert peer review or formal verification.\n"
            + "- It checks process hygiene and provenance more strongly than mathematical depth.\n"
            + "\n## Provenance\n\n"
            + "- source: internal workstream reports listed above.\n"
            + "- source: reviewer records stored in `project.json`.\n"
        )
        return WorkProduct(report=report, uncertainty=uncertainty, artifact_paths=[])


class ReviewerPanel:
    def review(self, workstream: Dict[str, Any], report: str) -> List[Dict[str, Any]]:
        return [
            self._logical_reviewer(workstream, report),
            self._provenance_reviewer(workstream, report),
            self._uncertainty_reviewer(workstream, report),
        ]

    @staticmethod
    def approved(reviews: List[Dict[str, Any]]) -> bool:
        return all(review["approved"] for review in reviews)

    def _logical_reviewer(self, workstream: Dict[str, Any], report: str) -> Dict[str, Any]:
        lower = report.lower()
        risky_finality = bool(re.search(r"\b(qed|therefore proved|we have proved)\b", lower))
        has_status = "claim status:" in lower
        approved = has_status and not risky_finality
        comment = "Report keeps claim status explicit and avoids final-proof language."
        if not has_status:
            comment = "Missing explicit claim status."
        elif risky_finality:
            comment = "Uses final-proof language without a formal certificate."
        return {"reviewer": "logical-skeptic", "approved": approved, "comment": comment, "time": now_iso()}

    def _provenance_reviewer(self, workstream: Dict[str, Any], report: str) -> Dict[str, Any]:
        lower = report.lower()
        has_provenance = "## provenance" in lower
        has_source = "source:" in lower
        approved = has_provenance and has_source
        comment = "Provenance section names auditable sources."
        if not has_provenance:
            comment = "Missing Provenance section."
        elif not has_source:
            comment = "Provenance section does not name any source."
        return {"reviewer": "provenance-checker", "approved": approved, "comment": comment, "time": now_iso()}

    def _uncertainty_reviewer(self, workstream: Dict[str, Any], report: str) -> Dict[str, Any]:
        lower = report.lower()
        has_gaps = "## known gaps" in lower
        stale_markers = any(marker in lower for marker in ("tbd", "todo", "???"))
        approved = has_gaps and not stale_markers
        comment = "Known gaps are surfaced without stale placeholders."
        if not has_gaps:
            comment = "Missing Known Gaps section."
        elif stale_markers:
            comment = "Contains unresolved placeholder markers."
        return {"reviewer": "uncertainty-manager", "approved": approved, "comment": comment, "time": now_iso()}


AGENTS: Dict[str, BaseAgent] = {
    "literature": LiteratureAgent(),
    "theory": TheoryAgent(),
    "computation": ComputationAgent(),
    "review": MetaReviewAgent(),
}

