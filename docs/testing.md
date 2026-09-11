<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Regression testing

Regression tests check that established scientific results, API behavior, and
installed packages continue to work after a change. The suite keeps **one
test file per language** in [`tests/`](../tests); tests remain disabled
in ordinary builds unless `BUILD_TESTING=ON` is requested.

## What happens when you change the code?

After you **push to `main` or update a pull request targeting `main`**, GitHub
Actions checks the changed paths and selects the numerical suites that can be
affected. Metadata and documentation-link checks run every time. Selected CPU
jobs build the library on fresh Linux and macOS machines and compare known
calculations and API behavior with their expected results, using the documented
floating-point tolerances.

The shared [scope selector](../.github/ci_scope.py) applies these rules:

| Changed files | Hosted CPU suite | Workstation suite on `main` |
|---|---|---|
| `docs/`, `.documentation/`, Markdown, or recognized metadata | Not required | Not required |
| Explicitly recognized CPU algorithms or CPU consumers | Run | Not required |
| CUDA kernels, recognized GPU-only adapters, or Mathematica implementation/tests | Not required | Run |
| Shared Wigner/booster code, public headers, CMake, shared API wrappers, or unclassified paths | Run | Run |

A change touching several groups runs the union of their required suites.
The CPU-only list is deliberately narrow: a host-side header may also control
GPU preparation or output layout. An unavailable or incomplete comparison
selects both suites rather than assuming the change is harmless. The scope
summary lists the decision and reasons. Starting a workflow manually forces
that workflow's numerical suite, regardless of the changed paths.

### Reading the Actions page

The **Regression** workflow contains scope selection, metadata/documentation
checks, and conditional Linux and macOS CPU jobs. Open a run to see which
jobs ran or were skipped, then open a job for its dependency, build, and CTest
steps. Individual numerical and API results appear in the test output.

Open **Actions → Regression → the latest run** to see the results. On a pull
request, open **Checks**, or click **Details** beside a check. Select a failed
job and expand its first failed step to read the error; downloadable diagnostic
logs appear under **Artifacts** on the run summary.

| Status | Interpretation |
|---|---|
| Green / passed | The work that ran succeeded; inspect skipped jobs before inferring which numerical suites ran. |
| Red / failed | Open the failing step and investigate before treating the change as ready. |
| Yellow / pending or running | A result is still outstanding. An offline Workstation is not a numerical test failure. |
| Not required / skipped | Scope selection omitted that suite for this change; this is not a numerical test pass. |
| Cancelled or interrupted | Execution did not complete; rerun the relevant workflow when it can finish. |

A compilation error, changed numerical result, or failed import needs a fix
or an explained, reviewed change to the expected behavior. If the log instead
shows a temporary infrastructure problem, rerun the relevant workflow once it
recovers. Repeating a reproducible failure until it happens to turn green does
not resolve it.

The README's **Regression** badge reports the native Actions workflow result
for `main`. It can be green when metadata checks pass and both CPU jobs are
skipped. The **GPU regression** badge reads the current `main` commit's
combined commit status; `GPU / Workstation` is the only custom commit-status
context maintained by this repository. It therefore follows that commit's GPU
status instead of the last completed request workflow.

A GPU status of **success** means either that all nine required Workstation
tests passed or that GPU execution was **not required** for these changes.
The latter has an explicit status description and a skipped request job; it
does not claim a numerical test pass. Open the commit's **GPU / Workstation**
status and the **GPU regression request** run to distinguish the two. A green
request-submission job alone never establishes GPU correctness.

## Automatic checks on GitHub

The [Regression workflow](../.github/workflows/regression.yml) runs for pushes
and pull requests to `main`, and can also be started from the Actions tab.
The metadata group always runs; the remaining groups run when CPU testing is
selected or the workflow is started manually.

| Check group | What it checks |
|---|---|
| Metadata and documentation links | CI helper tests, matching CMake/Julia/citation/README versions, CFF schema, local Markdown links and heading anchors |
| Ubuntu 24.04 | GCC 13, OpenBLAS, Python 3.10/NumPy, Julia 1.10; C++, Python, Julia CPU workflows |
| macOS 15 | Homebrew GCC, Accelerate, Python 3.13/NumPy, Julia 1.10; the same CPU workflows |
| Installed consumer, on each platform | Fresh production build with tests OFF, installation, independent compilation and execution of the two existing C++ examples |
| Linux Debug configuration | Separate CPU build and C++ workflow with assertions enabled; catches code excluded from Release builds by `NDEBUG` |

The C++ workflow includes independent reference values and comparisons among
fixed, coherent, and full-tensor calculations; it also covers input validation,
storage/cache behavior, configuration, and progress. Wrapper workflows exercise
their language-facing tensor, numerical, lifetime, and progress behavior.
The hosted jobs run CPU-labelled tests only: a CPU pass does not establish
CUDA correctness. Release and Debug use separate build directories. Failed
jobs retain build and CTest logs for 14 days.

### Workstation GPU automation

The [GPU regression request workflow](../.github/workflows/gpu-regression.yml)
checks each push to this repository's `main` branch. If Workstation coverage
is required, its **GPU / Workstation request** job posts a pending status for
that exact commit and finishes. Its success means the request was submitted,
not that the tests passed. If coverage is not required, the request job is
skipped and the status says **Not required for these changes**.

To force a Workstation run, use **Actions → GPU regression request → Run
workflow**, with `main` selected. Each request has its own run identifier and
attempt number; the execution service verifies that identity and revision
before accepting the request.

When online, the Workstation normally discovers a submitted request within
approximately one minute, then waits for its execution queue and builds the
requested revision. It enables CUDA and all four language APIs and runs all
nine CPU, GPU, and Mathematica workflows, even when the changed files affect
only GPU or Mathematica code. Missing or skipped required tests fail the
check. This validates the V100 backend; separate hardware validation is still
needed for Ampere kernels and multi-GPU behavior.

The external **GPU / Workstation** status remains pending while the machine
is offline, while a run is queued, or after an interrupted execution with no
completed result. There is no 40-minute waiting timeout converted into a test
failure. Configuration, build, test-inventory, and test failures are reported
as failures; a completed test pass requires all nine workflows. For an
interrupted execution, rerun the request workflow to create another attempt.
An undispatched request can be picked up when the Workstation returns online.

Pull requests, including those from forks, run metadata checks and any selected
hosted CPU jobs. They do not submit code to the Workstation: GPU requests are
limited to this repository's `main` branch. The public workflow uses hosted
runners for scope selection and request submission; accepted revisions run
through a separately managed Workstation execution service.

With Actions enabled, the workflow files select future runs automatically.
You can additionally make suitable checks **required** in GitHub's branch
rules for `main`. This is a separate repository setting: adding a workflow
does not itself prevent merging failed changes. Choose checks with the scope
policy in mind, since an unaffected numerical job may be skipped. The hosted
regression workflow requests read-only repository access and does not publish
packages or create releases.
[GitHub required status checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging)

## CPU checks locally

With GNU C++, CMake, a CBLAS provider, Python/NumPy, and Julia available:

```sh
cmake -S . -B build-tests \
  -DBUILD_TESTING=ON -DSL2CFOAMXX_BUILD_PYTHON=ON \
  -DSL2CFOAMXX_BUILD_JULIA=ON
cmake --build build-tests --parallel 2
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  ctest --test-dir build-tests -L cpu --output-on-failure --no-tests=error
```

Use a fresh build directory when changing compiler, BLAS, CUDA mode, or ABI.
See [building](building.md) for dependency locations. The workflows are the
executable reference for their exact configuration commands.

The metadata/link checker needs only Python 3.10+:

```sh
python3 .github/check_repository.py
```

It checks local links in maintained Markdown, including headings and image
paths, without querying external sites. The CI metadata job separately validates
`CITATION.cff` with `cffconvert==2.0.0`.

## GPU and Mathematica checks

GPU tests require a supported NVIDIA device and compatible CUDA/cuTENSOR
libraries. Mathematica tests additionally require a licensed Wolfram kernel
and LibraryLink SDK. See [GPU dependencies](building.md#nvidia-gpus) and the
[Mathematica build guide](mathematica.md).

Configure all four APIs and run the CPU, GPU, and Mathematica tests with:

```sh
cmake -S . -B build-gpu-tests \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CUDAHOSTCXX:-${CXX:-g++}}" \
  -DENABLE_CUDA=ON -DCUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DCUTENSOR_ROOT="$CUTENSOR_ROOT" -DBUILD_TESTING=ON \
  -DSL2CFOAMXX_BUILD_PYTHON=ON -DSL2CFOAMXX_BUILD_JULIA=ON \
  -DSL2CFOAMXX_BUILD_MATHEMATICA=ON \
  -DWolfram_ROOT="$Wolfram_ROOT" -DWolfram_EXECUTABLE="$WOLFRAM_KERNEL"
cmake --build build-gpu-tests --parallel 2
ctest --test-dir build-gpu-tests --output-on-failure --no-tests=error
```

Run the relevant GPU and language tests before accepting numerical/backend
changes. Check that CTest discovers the expected tests and that none are
skipped: a successful CPU-only run does not establish GPU correctness.
Supporting Volta and Ampere requires runs on both architectures; one GPU does
not validate the other architecture's specialized kernels. Performance scaling
requires separate measurements on the target hardware.

## Adding a regression

When a bug is reproduced, add its smallest meaningful case to the relevant
one of the four existing test files. Preserve the failing input and a
reference that can detect the original error: an independently established
value, a distinct fixed/coherent/tensor route, or a mathematical identity.
Check tolerances against numerical scale and cancellation; do not loosen a
tolerance or replace an expected value simply to make CI pass. When a scientific
convention intentionally changes, review and document the changed reference.
Keeping these cases meaningful is the main ongoing maintenance; GitHub handles
running them after subsequent changes.

Generated API references can automatically show signatures, types, documented
defaults, and comments, and CI can check example execution and documentation
links. Scientific conventions, normalization, convergence guidance, tutorials,
and the interpretation of limitations still need reviewed explanatory writing.

Shared hosted runners are unsuitable for strict performance thresholds.
Keep timing comparisons on controlled hardware, with repeated measurements,
fixed affinity/cache policy, and recorded workload and environment details.
Numerical pass/fail criteria remain separate from performance observations.
