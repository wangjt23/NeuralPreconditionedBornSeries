from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str, fallback: str = "project") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:52].strip("-") or fallback)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def infer_workstream_kind(title: str, statement: str) -> str:
    text = f"{title} {statement}".lower()
    if any(word in text for word in ("literature", "reference", "paper", "survey", "context")):
        return "literature"
    if any(word in text for word in ("compute", "numerical", "experiment", "search", "simulation", "code")):
        return "computation"
    if any(word in text for word in ("review", "verify", "audit", "check")):
        return "review"
    return "theory"


def default_goals(question: str) -> List[Dict[str, str]]:
    return [
        {
            "title": "Clarify definitions and intent",
            "statement": (
                "Refine the problem statement, record assumptions, and identify ambiguous terms "
                "before attempting a solution."
            ),
        },
        {
            "title": "Map local and literature context",
            "statement": (
                "Search the local mathematical notes and supplied references for related definitions, "
                "lemmas, examples, and prior strategies."
            ),
        },
        {
            "title": "Develop proof strategies and obstacle log",
            "statement": (
                "Generate plausible proof routes, counterexample routes, and a durable list of gaps "
                "that should not be silently forgotten."
            ),
        },
        {
            "title": "Design computational probes",
            "statement": (
                "Specify lightweight symbolic or numerical experiments that could test conjectures "
                "or expose edge cases."
            ),
        },
        {
            "title": "Independent review and synthesis",
            "statement": (
                "Audit the reports, separate supported claims from speculative ones, and summarize "
                "what a human mathematician should inspect next."
            ),
        },
    ]


def normalize_goal(goal: str, index: int) -> Dict[str, str]:
    if ":" in goal:
        title, statement = goal.split(":", 1)
        return {"title": title.strip() or f"Goal {index}", "statement": statement.strip() or goal.strip()}
    return {"title": f"Goal {index}", "statement": goal.strip()}


@dataclass
class ProjectStore:
    path: Path
    data: Dict[str, Any]

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def project_file(self) -> Path:
        return self.path / "project.json"

    @property
    def working_paper_file(self) -> Path:
        return self.path / "working_paper.md"

    @property
    def events_file(self) -> Path:
        return self.path / "events.jsonl"

    @property
    def messages_file(self) -> Path:
        return self.path / "messages.jsonl"

    @property
    def failed_file(self) -> Path:
        return self.path / "failed_explorations.jsonl"

    def save(self) -> None:
        self.data["updated_at"] = now_iso()
        write_json(self.project_file, self.data)

    def append_event(self, event: str, **payload: Any) -> None:
        append_jsonl(self.events_file, {"time": now_iso(), "event": event, **payload})

    def append_message(self, role: str, content: str) -> None:
        append_jsonl(self.messages_file, {"time": now_iso(), "role": role, "content": content})

    def append_failed_exploration(self, workstream_id: str, reason: str, evidence: Optional[List[str]] = None) -> None:
        append_jsonl(
            self.failed_file,
            {
                "time": now_iso(),
                "workstream_id": workstream_id,
                "reason": reason,
                "evidence": evidence or [],
            },
        )

    def events(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.events_file)

    def messages(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.messages_file)

    def failed_explorations(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.failed_file)

    def workstream_dir(self, workstream_id: str) -> Path:
        directory = self.path / "workstreams" / workstream_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def context_paths(self) -> List[Path]:
        paths: List[Path] = []
        for raw in self.data.get("context_paths", []):
            path = Path(raw)
            if path.exists():
                paths.append(path)
        return paths

    def get_workstream(self, workstream_id: str) -> Dict[str, Any]:
        for workstream in self.data["workstreams"]:
            if workstream["id"] == workstream_id:
                return workstream
        raise KeyError(f"Unknown workstream: {workstream_id}")


class Workspace:
    def __init__(self, root: Path):
        self.root = root

    @classmethod
    def from_cwd(cls, cwd: Optional[Path] = None, workspace: Optional[str] = None) -> "Workspace":
        base = Path(cwd or Path.cwd())
        root = Path(workspace).expanduser() if workspace else base / ".co_math"
        return cls(root.resolve())

    @property
    def projects_dir(self) -> Path:
        return self.root / "projects"

    def ensure(self) -> None:
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> List[ProjectStore]:
        self.ensure()
        projects: List[ProjectStore] = []
        for project_file in sorted(self.projects_dir.glob("*/project.json")):
            data = read_json(project_file)
            if data:
                projects.append(ProjectStore(project_file.parent, data))
        return projects

    def load_project(self, ref: str) -> ProjectStore:
        candidate = Path(ref)
        if candidate.exists():
            project_dir = candidate if candidate.is_dir() else candidate.parent
            project_file = project_dir / "project.json"
            data = read_json(project_file)
            if data:
                return ProjectStore(project_dir.resolve(), data)

        matches = [project for project in self.list_projects() if project.id == ref or project.id.startswith(ref)]
        if not matches:
            raise FileNotFoundError(f"No project found for '{ref}' in {self.projects_dir}")
        if len(matches) > 1:
            ids = ", ".join(project.id for project in matches)
            raise ValueError(f"Project reference '{ref}' is ambiguous: {ids}")
        return matches[0]

    def create_project(
        self,
        question: str,
        goals: Optional[Iterable[str]] = None,
        context_paths: Optional[Iterable[str]] = None,
        title: Optional[str] = None,
        paper_source: str = "",
        cwd: Optional[Path] = None,
    ) -> ProjectStore:
        self.ensure()
        base = Path(cwd or Path.cwd()).resolve()
        project_title = title or question.strip().splitlines()[0][:80] or "AI co-mathematician project"
        project_id = f"{slugify(project_title)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=False)

        if goals:
            goal_rows = [normalize_goal(goal, index + 1) for index, goal in enumerate(goals)]
        else:
            goal_rows = default_goals(question)

        context = self._resolve_context_paths(context_paths, base)
        normalized_goals: List[Dict[str, Any]] = []
        workstreams: List[Dict[str, Any]] = []
        created_at = now_iso()
        for index, goal in enumerate(goal_rows, start=1):
            goal_id = f"g{index:02d}"
            kind = infer_workstream_kind(goal["title"], goal["statement"])
            workstream_id = f"w{index:02d}-{kind}"
            normalized_goals.append(
                {
                    "id": goal_id,
                    "title": goal["title"],
                    "statement": goal["statement"],
                    "status": "active",
                    "workstream_ids": [workstream_id],
                }
            )
            workstreams.append(
                {
                    "id": workstream_id,
                    "goal_id": goal_id,
                    "kind": kind,
                    "title": goal["title"],
                    "statement": goal["statement"],
                    "status": "queued",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "review_rounds": 0,
                    "review_state": [],
                    "report_path": None,
                    "artifact_paths": [],
                    "uncertainty": [],
                }
            )

        data = {
            "id": project_id,
            "title": project_title,
            "question": question.strip(),
            "status": "active",
            "created_at": created_at,
            "updated_at": created_at,
            "context_paths": [str(path) for path in context],
            "goals": normalized_goals,
            "workstreams": workstreams,
            "steering": [],
            "paper_source": paper_source,
            "design_notes": [
                "Stateful workspace with durable project files.",
                "Coordinator/workstream hierarchy with progressive disclosure.",
                "Reports require review approval before completion.",
                "Failed explorations are preserved as first-class records.",
            ],
        }
        project = ProjectStore(project_dir, data)
        project.save()
        project.append_event("project_created", question=question, goals=len(normalized_goals), context_paths=len(context))
        project.append_message("user", question)
        for file_name in ("events.jsonl", "messages.jsonl", "failed_explorations.jsonl"):
            (project_dir / file_name).touch(exist_ok=True)
        return project

    def _resolve_context_paths(self, context_paths: Optional[Iterable[str]], base: Path) -> List[Path]:
        if context_paths:
            paths = []
            for raw in context_paths:
                path = Path(raw).expanduser()
                if not path.is_absolute():
                    path = base / path
                if path.exists():
                    paths.append(path.resolve())
            return sorted(dict.fromkeys(paths))

        paths = []
        for path in base.glob("*.md"):
            if ".co_math" not in path.parts:
                paths.append(path.resolve())
        return sorted(paths)
