# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

"""Policy and event tests; no network or mutation of the source checkout."""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ci_scope


BASE = "a" * 40
HEAD = "b" * 40


class ScopeTests(unittest.TestCase):
    def assert_scope(self, paths, cpu, gpu):
        decision = ci_scope.classify_paths(paths)
        self.assertEqual((decision.cpu, decision.gpu), (cpu, gpu))
        self.assertTrue(decision.reasons)

    def test_documentation_and_metadata_skip_numerical_suites(self):
        self.assert_scope([
            ".documentation/theory/contractions.md",
            ".documentation/render_figures.py",
            ".documentation/theory/assets/network.svg",
            ".documentation/requirements.txt", ".documentation/mkdocs.yml",
            "docs/testing.md", "README.md", "src/README.md", "LICENSE",
            "CITATION.cff", "AGENTS.md", "EXPERIMENTS.md",
            "mathematica/README.md", ".github/workstation/README.md",
            ".github/check_repository.py", ".github/workflows/docs.yml",
        ], False, False)

    def test_verified_cpu_implementations_skip_workstation(self):
        for path in [
            "src/api/Amplitude.cpp", "src/api/FullTensorCpu.cpp",
            "src/amplitude/AmplitudePackedContractionGeneral.cpp",
            "src/amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbit.cpp",
            "src/amplitude/AmplitudePackedXQSlab.cpp",
            "src/full_tensor/FullTensorExecutionPlanner.cpp",
            "src/core/CpuBlas.hpp", "examples/simple_example.cpp",
        ]:
            with self.subTest(path=path):
                self.assert_scope([path], True, False)

    def test_cpu_allowlist_matches_exact_repository_filenames(self):
        repository = Path(__file__).resolve().parents[1]
        for path in ci_scope.CPU_ONLY:
            with self.subTest(path=path):
                parent = repository
                for component in Path(path).parts:
                    # Path.exists() alone cannot catch case errors on macOS.
                    self.assertIn(component, {entry.name for entry in parent.iterdir()})
                    parent /= component
                self.assertTrue(parent.is_file())

    def test_cpu_named_shared_files_require_both(self):
        for path in [
            "src/full_tensor/CpuContraction.cpp",
            "src/full_tensor/CpuContraction.hpp",
            "src/amplitude/AmplitudePackedCoherentI1Batch.cpp",
            "src/amplitude/AmplitudePackedCoherentI1Batch.hpp",
            "src/amplitude/AmplitudePackedContraction.hpp",
        ]:
            with self.subTest(path=path):
                self.assert_scope([path], True, True)

    def test_shared_numerical_and_build_inputs_require_both(self):
        for path in [
            "src/boosters/B4TensorFilling.hpp", "src/wigner/ThreeJRecurrence.hpp",
            "src/amplitude/coherent_intertwiner.cpp",
            "src/amplitude/AmplitudeInputSupport.cpp",
            "src/amplitude/AmplitudePackedBoosters.cpp",
            "src/full_tensor/FullTensorQReflectionOrbitPlan.cpp",
            "src/core/LowLevelTuning.hpp", "src/core/Complex64.hpp",
            "src/api/CoherentAmplitude.cpp", "src/api/Configuration.cpp",
            "include/sl2cfoamxx/tensor.hpp", "CMakeLists.txt",
            "cmake/sl2cfoamxxConfig.cmake.in", "python/CMakeLists.txt",
            "python/bindings.cpp", "julia/SL2CfoamXX/src/computations.jl",
            "tests/test_public_api.cpp", "tests/test_python_api.py",
            "tests/test_julia_api.jl", ".github/ci_scope.py",
            ".github/test_ci_scope.py",
        ]:
            with self.subTest(path=path):
                self.assert_scope([path], True, True)

    def test_cuda_and_mathematica_require_workstation(self):
        for path in [
            "src/gpu/CoherentGpuBackend.cu",
            "src/gpu/FullTensorSemiFusedGpuKernels.cuh",
            "src/api/FullTensorGpu.cpp",
            "src/gpu/CoherentProductionGpuContraction.hpp",
            "mathematica/bridge.cpp", "mathematica/CMakeLists.txt",
            "mathematica/SL2CfoamXX/Kernel/SL2CfoamXX.wl",
            "examples/mathematica_example.nb", "tests/test_mathematica_api.wl",
            ".github/workstation/validate_gpu_request.py",
        ]:
            with self.subTest(path=path):
                self.assert_scope([path], False, True)

    def test_workflow_and_consumer_changes(self):
        self.assert_scope([".github/workflows/regression.yml"], True, False)
        self.assert_scope([".github/workflows/gpu-regression.yml"], False, True)
        self.assert_scope([".github/consumer/CMakeLists.txt"], True, False)

    def test_unknown_paths_fail_conservatively(self):
        for path in ["src/new_algorithm.cpp", "build-options.json",
                     ".github/workflows/new-build.yml", "src/gpu/new_host.cpp",
                     "src/../docs/fake.md", "/docs/fake.md"]:
            with self.subTest(path=path):
                self.assert_scope([path], True, True)

    def test_combined_changes_and_rename_sides_are_unioned(self):
        self.assert_scope(["README.md", "src/api/FullTensorCpu.cpp"], True, False)
        self.assert_scope(["src/api/FullTensorCpu.cpp",
                           "src/gpu/CoherentGpuBackend.cu"], True, True)
        # A code-to-documentation rename must retain the deleted source path.
        self.assert_scope(["src/core/Complex64.hpp", "docs/old-header.md"], True, True)
        self.assert_scope(["docs/old.md", "docs/new.md"], False, False)

    def test_empty_complete_diff_is_a_skip(self):
        self.assert_scope([], False, False)


class EventTests(unittest.TestCase):
    def fake_run(self, data=b"README.md\0"):
        self.calls = []

        def run(args, **kwargs):
            self.calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, stdout=data, stderr=b"")
        return run

    def test_push_compares_before_and_after_without_rename_detection(self):
        decision = ci_scope.scope_from_event(
            "push", {"before": BASE, "after": HEAD}, self.fake_run())
        self.assertFalse(decision.cpu or decision.gpu)
        args, kwargs = self.calls[0]
        self.assertEqual(args[-3:], [BASE, HEAD, "--"])
        for option in ("-z", "--name-only", "--no-renames", "--no-ext-diff", "--no-textconv"):
            self.assertIn(option, args)
        self.assertTrue(kwargs["check"])

    def test_pull_request_uses_merge_base_comparison(self):
        decision = ci_scope.scope_from_event(
            "pull_request", {"pull_request": {
                "base": {"sha": BASE}, "head": {"sha": HEAD}}},
            self.fake_run(b"src/api/FullTensorCpu.cpp\0"))
        self.assertEqual((decision.cpu, decision.gpu), (True, False))
        self.assertEqual(self.calls[0][0][-2:], [f"{BASE}...{HEAD}", "--"])

    def test_deleted_source_and_both_rename_sides_are_retained(self):
        decision = ci_scope.scope_from_event(
            "push", {"before": BASE, "after": HEAD}, self.fake_run(
                b"src/api/FullTensorCpu.cpp\0src/gpu/moved.cu\0docs/new name.md\0"))
        self.assertEqual((decision.cpu, decision.gpu), (True, True))
        self.assertEqual(len(decision.reasons), 3)

    def test_nul_parsing_preserves_spaces_and_newlines(self):
        decision = ci_scope.scope_from_event(
            "push", {"before": BASE, "after": HEAD},
            self.fake_run(b"docs/two words.md\0docs/two\nlines.md\0"))
        self.assertFalse(decision.cpu or decision.gpu)
        self.assertEqual(len(decision.reasons), 2)
        self.assertIn("\\n", "".join(decision.reasons))

    def test_manual_and_invalid_comparisons_request_both_without_git(self):
        cases = [
            ("workflow_dispatch", {}), ("unknown_event", {}),
            ("push", {"after": HEAD}),
            ("push", {"before": "0" * 40, "after": HEAD}),
            ("push", {"before": "--help", "after": HEAD}),
            ("push", []), ("pull_request", {"pull_request": {}}),
        ]
        for name, event in cases:
            with self.subTest(name=name, event=event):
                def forbidden(*args, **kwargs):
                    self.fail("Git must not run without valid commits")
                decision = ci_scope.scope_from_event(name, event, forbidden)
                self.assertTrue(decision.cpu and decision.gpu)

    def test_failed_or_malformed_diff_requests_both(self):
        for error in [subprocess.CalledProcessError(128, "git"),
                      subprocess.TimeoutExpired("git", 60), OSError("unavailable")]:
            with self.subTest(error=error):
                def failing(*args, **kwargs):
                    raise error
                decision = ci_scope.scope_from_event(
                    "push", {"before": BASE, "after": HEAD}, failing)
                self.assertTrue(decision.cpu and decision.gpu)
        decision = ci_scope.scope_from_event(
            "push", {"before": BASE, "after": HEAD}, self.fake_run(b"README.md"))
        self.assertTrue(decision.cpu and decision.gpu)

    def test_empty_successful_diff_skips(self):
        decision = ci_scope.scope_from_event(
            "push", {"before": BASE, "after": HEAD}, self.fake_run(b""))
        self.assertFalse(decision.cpu or decision.gpu)

    def test_cli_outputs_and_summary_are_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = root / "event.json"
            event.write_text(json.dumps({"before": BASE, "after": HEAD}))
            env = {"GITHUB_EVENT_NAME": "push", "GITHUB_EVENT_PATH": str(event),
                   "GITHUB_OUTPUT": str(root / "outputs"),
                   "GITHUB_STEP_SUMMARY": str(root / "summary")}
            decision = ci_scope.classify_paths(["docs/<unsafe>`name.md"])
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(ci_scope, "scope_from_event", return_value=decision), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ci_scope.main(), 0)
            self.assertEqual((root / "outputs").read_text(), "cpu=false\ngpu=false\n")
            summary = (root / "summary").read_text()
            self.assertIn("&lt;unsafe&gt;", summary)
            self.assertNotIn("<unsafe>", summary)

    def test_missing_event_file_falls_back_to_both(self):
        with patch.dict(os.environ, {}, clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(ci_scope.main(), 0)
        self.assertEqual(output.getvalue(), "cpu=true\ngpu=true\n")


class GitDiffIntegrationTests(unittest.TestCase):
    def test_real_diff_keeps_renamed_deleted_and_other_changed_paths(self):
        # All fixture commits live in a newly created temporary repository.
        # No Git command runs in, stages, or changes the source checkout.
        with tempfile.TemporaryDirectory(prefix="ci-scope-git-fixture-") as temporary:
            repository = Path(temporary)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
            env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_TERMINAL_PROMPT="0")

            def git(*arguments):
                return subprocess.run(
                    ["git", "-c", "user.name=CI scope fixture",
                     "-c", "user.email=ci-scope@example.invalid",
                     "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={os.devnull}",
                     *arguments], cwd=repository, env=env, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                ).stdout

            def write(relative, content):
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            git("init", "--quiet")
            cpu_path = "src/api/FullTensorCpu.cpp"
            gpu_path = "src/gpu/CoherentGpuBackend.cu"
            write(cpu_path, "CPU fixture\n")
            write(gpu_path, "GPU fixture\n")
            write("README.md", "Original documentation\n")
            git("add", "--all")
            git("commit", "--quiet", "-m", "Fixture baseline")
            before = git("rev-parse", "HEAD").decode().strip()

            renamed_path = "docs/moved source.txt"
            (repository / "docs").mkdir()
            (repository / cpu_path).rename(repository / renamed_path)
            (repository / gpu_path).unlink()
            write("README.md", "Updated documentation\n")
            write("docs/new page.md", "Another changed file\n")
            git("add", "--all")
            git("commit", "--quiet", "-m", "Rename, delete, and edit fixture files")
            after = git("rev-parse", "HEAD").decode().strip()

            observed_paths = []

            def local_diff(arguments, **kwargs):
                result = subprocess.run(arguments, cwd=repository, env=env, **kwargs)
                observed_paths.extend(os.fsdecode(p) for p in result.stdout.split(b"\0") if p)
                return result

            decision = ci_scope.scope_from_event(
                "push", {"before": before, "after": after}, local_diff)
            self.assertEqual(set(observed_paths), {
                cpu_path, gpu_path, renamed_path, "README.md", "docs/new page.md"})
            self.assertTrue(decision.cpu and decision.gpu)
            self.assertEqual(len(decision.reasons), 5)


if __name__ == "__main__":
    unittest.main()
