#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

"""Select numerical regression suites from a complete local Git diff.

The narrow CPU allowlist is intentional: similarly named files contain shared
GPU preparation, output mapping, or pruning. Unknown paths run both suites.
This is a scheduling policy, not an authorization check for the Workstation.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class ScopeDecision:
    cpu: bool
    gpu: bool
    reasons: tuple[str, ...]


# These implementations only execute CPU algorithms. Their companion headers
# are shared unless explicitly listed below: GPU host bridges include several
# amplitude headers even though they use a different contraction algorithm.
CPU_ONLY = frozenset({
    "src/api/Amplitude.cpp",
    "src/api/AmplitudeCache.cpp",
    "src/api/FullTensorCpu.cpp",
    "src/amplitude/AmplitudePackedAutoSlab.cpp",
    "src/amplitude/AmplitudePackedAutoSlab.hpp",
    "src/amplitude/AmplitudePackedContraction.cpp",
    "src/amplitude/AmplitudePackedContractionGeneral.cpp",
    "src/amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbit.cpp",
    "src/amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbitCostSelected.cpp",
    "src/amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbitCostSelected_HalfSjC.cpp",
    "src/amplitude/AmplitudePackedSJTensors_HalfSjC.cpp",
    "src/amplitude/AmplitudePackedXQSlab.cpp",
    "src/amplitude/AmplitudePackedXSlab.cpp",
    "src/amplitude/AmplitudePackedXSlab.hpp",
    "src/full_tensor/FullTensorExecutionPlanner.cpp",
    "src/full_tensor/FullTensorExecutionPlanner.hpp",
    "src/full_tensor/FullTensorPackedSJTensors_HalfSjC.cpp",
    "src/core/CpuBlas.hpp",
    "examples/simple_example.cpp",
    "examples/coherent_boundary_example.cpp",
})

# Shared host/device layouts and generic GPU API headers fall through to both.
# The files below contain the GPU dispatch/adapter or vendor-only operations.
GPU_ONLY = frozenset({
    "src/api/FullTensorGpu.cpp",
    "src/gpu/CoherentProductionGpuContraction.hpp",
    "src/gpu/CoherentGpuCuTensor.hpp",
    "src/gpu/CuTensorCommon.hpp",
    "src/gpu/CuTensorCompatibility.hpp",
    "src/gpu/FullTensorSemiFusedCuTensor.hpp",
    "src/gpu/FullTensorSemiFusedNvtx.hpp",
    "src/gpu/ProductionGpuFallback.hpp",
    "src/gpu/production/FullTensorGpu.hpp",
    "src/gpu/production/GpuProductionCommon.hpp",
    "examples/mathematica_example.nb",
    "tests/test_mathematica_api.wl",
})

METADATA_ONLY = frozenset({
    "CITATION.cff", "LICENSE", "LICENSE.txt", "COPYING", "COPYING.txt",
    ".gitignore", ".gitattributes", ".editorconfig",
    ".github/check_repository.py", ".github/actionlint.yaml",
    ".github/workflows/docs.yml",
})


def _path_scope(path: str) -> tuple[bool, bool, str]:
    # Do not normalize away suspicious components into an allowlisted path.
    if not path or path.startswith("/") or ".." in path.split("/"):
        return True, True, "unrecognized path"
    if (path in METADATA_ONLY or path.endswith(".md")
            or path.startswith((".documentation/", "docs/"))):
        return False, False, "documentation or metadata"
    if path in {".github/ci_scope.py", ".github/test_ci_scope.py"}:
        return True, True, "regression selection policy"
    if path == ".github/workflows/regression.yml":
        return True, False, "CPU workflow"
    if path == ".github/workflows/gpu-regression.yml":
        return False, True, "Workstation workflow"
    if path.startswith(".github/consumer/"):
        return True, False, "installed CPU consumer"
    if path.startswith(".github/workstation/"):
        return False, True, "Workstation integration"
    if path in CPU_ONLY:
        return True, False, "CPU implementation or consumer"
    if (path in GPU_ONLY or path.startswith("mathematica/")
            or (path.startswith("src/gpu/")
                and PurePosixPath(path).suffix in {".cu", ".cuh"})):
        return False, True, "GPU or Mathematica implementation"
    return True, True, "shared or unclassified implementation/build input"


def classify_paths(paths: Iterable[str]) -> ScopeDecision:
    """Combine path requirements; callers must include old and new rename sides."""
    cpu = gpu = False
    reasons = []
    for path in sorted(set(paths)):
        need_cpu, need_gpu, reason = _path_scope(path)
        cpu |= need_cpu
        gpu |= need_gpu
        # JSON quoting keeps unusual Git filenames unambiguous in log output.
        reasons.append(f"{json.dumps(path, ensure_ascii=True)}: {reason}")
    if not reasons:
        reasons.append("No changed files in the complete comparison.")
    return ScopeDecision(cpu, gpu, tuple(reasons))


def _both(reason: str) -> ScopeDecision:
    return ScopeDecision(True, True, (reason,))


def _commit(value: object) -> str:
    if (not isinstance(value, str)
            or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", value)
            or set(value) == {"0"}):
        raise ValueError("missing, zero, or invalid comparison commit")
    return value


def scope_from_event(
    event_name: str,
    event: object,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> ScopeDecision:
    """Use push endpoints or PR merge-base comparison; uncertainty selects both."""
    if event_name == "workflow_dispatch":
        return _both("Manual dispatch requests both numerical suites.")
    if event_name not in {"push", "pull_request"}:
        return _both("Unsupported event; run both numerical suites.")
    try:
        if not isinstance(event, dict):
            raise ValueError("invalid event object")
        if event_name == "push":
            base = _commit(event.get("before"))
            head = _commit(event.get("after"))
            comparison = [base, head]
        else:
            pr = event["pull_request"]
            base = _commit(pr["base"]["sha"])
            head = _commit(pr["head"]["sha"])
            comparison = [f"{base}...{head}"]
        result = run(
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--name-only",
             "-z", "--no-renames", *comparison, "--"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            timeout=60,
        )
        data = result.stdout
        if not isinstance(data, bytes) or (data and not data.endswith(b"\0")):
            raise ValueError("invalid NUL-delimited Git diff")
        return classify_paths(os.fsdecode(path) for path in data.split(b"\0") if path)
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError):
        return _both("Comparison unavailable or invalid; run both numerical suites.")


def write_outputs(decision: ScopeDecision, env: Mapping[str, str]) -> None:
    values = f"cpu={str(decision.cpu).lower()}\ngpu={str(decision.gpu).lower()}\n"
    print(values, end="")
    if env.get("GITHUB_OUTPUT"):
        with open(env["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            stream.write(values)
    if env.get("GITHUB_STEP_SUMMARY"):
        with open(env["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write("### Numerical regression scope\n\n")
            stream.write(f"CPU: **{decision.cpu}**; Workstation: **{decision.gpu}**.\n\n")
            for reason in decision.reasons:
                # HTML escaping also protects Markdown filenames containing
                # markup or backticks without silently dropping their reason.
                stream.write(f"- <code>{html.escape(reason)}</code>\n")


def main() -> int:
    try:
        with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as stream:
            event = json.load(stream)
        decision = scope_from_event(os.environ.get("GITHUB_EVENT_NAME", ""), event)
    except (KeyError, OSError, ValueError):
        decision = _both("Event data unavailable or invalid; run both numerical suites.")
    write_outputs(decision, os.environ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
