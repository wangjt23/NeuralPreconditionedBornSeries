from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .core import Workspace
from .engine import CoMathematician


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m co_math",
        description="Local AI co-mathematician workbench prototype.",
    )
    parser.add_argument("--workspace", help="Workspace directory. Defaults to ./.co_math")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create a stateful math research project.")
    init.add_argument("--question", required=True, help="Research question or problem statement.")
    init.add_argument("--title", help="Short project title.")
    init.add_argument("--paper-source", default="", help="Primary paper, note, or URL that anchors the project.")
    init.add_argument("--goal", action="append", default=[], help="Goal, optionally as 'Title: statement'. Repeatable.")
    init.add_argument("--context", action="append", default=[], help="Context file. Defaults to all Markdown files in cwd.")

    run = subparsers.add_parser("run", help="Advance queued workstreams.")
    run.add_argument("project", help="Project id, prefix, or project directory.")
    run.add_argument("--rounds", type=int, default=1, help="Coordinator rounds to run.")
    run.add_argument("--max-review-rounds", type=int, default=2, help="Review attempts before blocking.")

    status = subparsers.add_parser("status", help="Show project status.")
    status.add_argument("project", help="Project id, prefix, or project directory.")
    status.add_argument("--details", action="store_true", help="Show reports, uncertainties, and failure records.")

    steer = subparsers.add_parser("steer", help="Add human steering and reopen blocked workstreams.")
    steer.add_argument("project", help="Project id, prefix, or project directory.")
    steer.add_argument("message", help="Human steering message.")

    export = subparsers.add_parser("export", help="Export the working paper.")
    export.add_argument("project", help="Project id, prefix, or project directory.")
    export.add_argument("--format", choices=["latex"], default="latex")
    export.add_argument("--output", help="Output path. Defaults to working_paper.tex in the project.")

    demo = subparsers.add_parser("demo", help="Create and run a small demo project.")
    demo.add_argument(
        "--question",
        default="Can the local notes suggest a rigorous route for a group-invariant approximation theorem?",
    )
    demo.add_argument("--rounds", type=int, default=1)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = Workspace.from_cwd(Path.cwd(), args.workspace)
    engine = CoMathematician(workspace)

    if args.command == "init":
        project = workspace.create_project(
            question=args.question,
            goals=args.goal,
            context_paths=args.context,
            title=args.title,
            paper_source=args.paper_source,
            cwd=Path.cwd(),
        )
        engine._write_working_paper(project)
        project.save()
        print(project.id)
        print(f"Working paper: {project.working_paper_file}")
        return 0

    if args.command == "run":
        project = engine.run(args.project, rounds=args.rounds, max_review_rounds=args.max_review_rounds)
        print(engine.status(project.id, details=True))
        return 0

    if args.command == "status":
        print(engine.status(args.project, details=args.details))
        return 0

    if args.command == "steer":
        project = engine.steer(args.project, args.message)
        print(f"Recorded steering for {project.id}")
        return 0

    if args.command == "export":
        if args.format == "latex":
            output = engine.export_latex(args.project, args.output)
            print(output)
            return 0

    if args.command == "demo":
        project = workspace.create_project(question=args.question, cwd=Path.cwd())
        engine._write_working_paper(project)
        project.save()
        engine.run(project.id, rounds=args.rounds)
        print(engine.status(project.id, details=True))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
