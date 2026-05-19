from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResult:
    text: str
    provider: str


class OptionalLLM:
    """Small adapter for an external model command.

    Set CO_MATH_LLM_CMD to a command that reads a JSON object from stdin:
    {"system": "...", "prompt": "..."}

    The command should write plain text to stdout. This keeps the prototype
    provider-neutral and dependency-free.
    """

    def __init__(self, command: Optional[str] = None, timeout_seconds: int = 90):
        self.command = command if command is not None else os.environ.get("CO_MATH_LLM_CMD")
        self.timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self.command)

    def complete(self, system: str, prompt: str) -> Optional[LLMResult]:
        if not self.command:
            return None
        args = shlex.split(self.command)
        payload = json.dumps({"system": system, "prompt": prompt})
        try:
            completed = subprocess.run(
                args,
                input=payload,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return LLMResult(text=f"LLM bridge unavailable: {exc}", provider="external-command-error")
        text = completed.stdout.strip()
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            return LLMResult(
                text=f"LLM bridge failed with exit code {completed.returncode}: {stderr}",
                provider="external-command-error",
            )
        return LLMResult(text=text, provider="external-command")

