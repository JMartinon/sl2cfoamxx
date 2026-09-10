<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Regression testing

Regression tests check that established scientific results, API behavior, and
installed packages continue to work after a change. The suite keeps **one
test file per language** in [`tests/`](../tests); tests remain disabled
in ordinary builds unless `BUILD_TESTING=ON` is requested.

## What happens when you change the code?

After you **push to `main` or update a pull request targeting `main`**, GitHub
Actions starts automatically. It copies the code onto fresh Linux and macOS
machines, builds the library, and runs known calculations and API checks.
Each check compares the result with its expected behavior, allowing the
documented floating-point tolerance. You do not need to start these machines
or run the hosted checks yourself.

### Reading the Actions page

The **Regression** workflow contains metadata/documentation checks, Linux CPU
checks, and macOS CPU checks. Each push or pull-request update creates a new
run. Open a job to see its steps, such as installing dependencies, compiling,
and running CTest. Numerical and API test results appear in the test output.

Open **Actions → Regression → the latest run** to see the results. On a pull
request, open **Checks**, or click **Details** beside a check. Select a failed
job and expand its first failed step to read the error; downloadable diagnostic
logs appear under **Artifacts** on the run summary.

| Status | What to do |
|---|---|
| Green / passed | The checks that ran succeeded for that commit. |
| Red / failed | Open the failing step and investigate before treating the change as ready. |
| Yellow / pending or running | Wait for the checks to finish. |
| Skipped or cancelled | That job did not complete its checks; this is not a test pass. |

A compilation error, changed numerical result, or failed import needs a fix
or an explained, reviewed change to the expected behavior. If the log instead
shows a temporary download failure or unavailable runner, use **Re-run failed
jobs** once the infrastructure recovers. Repeating a reproducible failure until
it happens to turn green does not resolve it.

The README's **Regression** badge reports the latest `main` CPU checks. The
separate **GPU regression** badge reports the V100 Workstation checks for
public `main` commits, including licensed Mathematica validation. Open
**Actions → GPU regression** or the commit's **GPU / Workstation** status to
see the outcome and the exact revision tested. A GPU pass requires all nine
CPU, GPU, and Mathematica workflows to complete without skipped tests.

## Automatic checks on GitHub

The [Regression workflow](../.github/workflows/regression.yml) runs for pushes
and pull requests to `main`, and can also be started from the Actions tab.

| Check group | What it checks |
|---|---|
| Metadata and documentation links | Matching CMake/Julia/citation/README versions, CFF schema, local Markdown links and heading anchors |
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

The [GPU regression workflow](../.github/workflows/gpu-regression.yml) starts
after every push to this repository's `main` branch. It can also be started
from **Actions → GPU regression → Run workflow**, with `main` selected. A
GitHub-hosted job requests a run for that exact commit and waits for the
NVIDIA V100 Workstation to report its result. Each request has its own run
identifier and attempt number; a result from an earlier commit or rerun cannot
satisfy a new request.

The Workstation discovers requests within approximately one minute, then
waits for its execution queue and builds the requested revision. It enables
CUDA and all four language APIs and runs all nine CPU, GPU, and Mathematica
workflows. Missing or skipped tests fail the check. This validates the V100
backend; separate hardware validation is still needed for Ampere kernels and
multi-GPU behavior.

If the Workstation is offline, the status remains pending. The hosted job
waits up to 40 minutes, then fails with a timeout rather than reporting a test
pass. The same limit includes time spent queued or building. Bring the
Workstation online and use **Re-run failed jobs** to issue a new request.

Pull requests, including those from forks, run the hosted CPU checks. They do
not submit code to the Workstation: GPU requests are limited to this
repository's `main` branch, and the public workflow does not check out code or
run on a self-hosted machine. The Workstation runs accepted revisions through
a separately managed execution service.

With Actions enabled, the workflow files handle future runs automatically.
You can additionally make the three hosted job checks **required** in GitHub's
branch rules for `main` after their first successful run. This is a separate
repository setting: adding a workflow does not itself prevent merging failed
changes. Required checks provide that protection when configured. The regression
workflow requests read-only repository access and does not publish packages or
create releases.
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
