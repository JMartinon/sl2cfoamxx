// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include <sl2cfoamxx.hpp>
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

template <class Function>
auto timed(Function&& function)
{
    const auto begin = Clock::now();
    auto value = std::forward<Function>(function)();
    const double seconds = std::chrono::duration<double>(
        Clock::now() - begin).count();
    return std::make_pair(std::move(value), seconds);
}

double relative_error(std::complex<double> a, std::complex<double> b)
{
    const double numerator = abs(a - b);
    const double denominator = std::max({abs(a), abs(b), 1.0e-300});
    return numerator / denominator;
}

struct TensorError {
    double maximum_scaled = 0.0;
    double relative_l2 = 0.0;
};

class TemporaryDirectory {
public:
    TemporaryDirectory()
    {
        const auto stamp = Clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("sl2cfoamxx_public_api_" + std::to_string(stamp));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_{};
};

class ScopedEnvironment {
public:
    ScopedEnvironment(const char* name, const char* value) : name_(name)
    {
        if (const char* previous = std::getenv(name)) {
            had_value_ = true;
            previous_ = previous;
        }
        if ((value ? ::setenv(name, value, 1) : ::unsetenv(name)) != 0)
            throw std::runtime_error("cannot set test environment: " + std::string(name));
    }

    ~ScopedEnvironment()
    {
        if (had_value_) (void)::setenv(name_, previous_.c_str(), 1);
        else (void)::unsetenv(name_);
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

private:
    const char* name_;
    std::string previous_;
    bool had_value_ = false;
};

TensorError tensor_error(std::span<const double> actual,
                         std::span<const double> expected)
{
    if (actual.size() != expected.size() || actual.empty()) {
        return {std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()};
    }

    long double squared_error = 0.0L;
    long double squared_reference = 0.0L;
    double maximum_error = 0.0;
    double maximum_reference = 0.0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double difference = actual[i] - expected[i];
        maximum_error = std::max(maximum_error, std::abs(difference));
        maximum_reference = std::max(maximum_reference, std::abs(expected[i]));
        squared_error += static_cast<long double>(difference) * difference;
        squared_reference +=
            static_cast<long double>(expected[i]) * expected[i];
    }

    constexpr long double floor = 1.0e-300L;
    return {
        static_cast<double>(static_cast<long double>(maximum_error) /
                            std::max(static_cast<long double>(maximum_reference),
                                     floor)),
        static_cast<double>(std::sqrt(squared_error) /
                            std::max(std::sqrt(squared_reference), floor))};
}

void check_general_spin_tensor_storage(
    const sl2cfoamxx::LibraryConfig& initial,
    sl2cfoamxx::Backend candidate_backend)
{
    using namespace sl2cfoamxx;
    // The first boundary preserves reflection 15432. Changing only j14 in
    // the second destroys that symmetry and gives unequal boundary intertwiner extents;
    // it must use the complete logical-q GPU schedule and rectangular output.
    const std::array<DSpins, 2> boundaries{{
        {{2, 1, 1, 2, 1, 1, 2, 1, 1, 1}},
        {{2, 1, 3, 2, 1, 1, 2, 1, 1, 1}},
    }};
    LibraryConfig config = initial;
    config.cpu_threads = 1;
    config.verbosity = Verbosity::silent;
    config.progress_bar = false;
    config.log_stream = nullptr;
    config.tensors.reuse_existing = false;
    std::size_t checks = 0;
    double maximum_scaled = 0.0;
    double maximum_relative_l2 = 0.0;
    for (std::size_t fixture = 0; fixture < boundaries.size(); ++fixture) {
        for (const auto simplicity : {SimplicityConstraint::gamma_j,
                                      SimplicityConstraint::gamma_j_p1}) {
            config.simplicity_constraint = simplicity;
            for (std::size_t cutoff = 0; cutoff <= 2; ++cutoff) {
                config.tensors.storage = TensorStorage::in_memory_dense;
                config.tensors.output_path.clear();
                configure(config);
                const FullTensor reference = compute_full_tensor(
                    boundaries[fixture], cutoff, 1.2, Backend::CPU);
                for (const auto storage : {TensorStorage::in_memory_dense,
                                          TensorStorage::in_memory_sym,
                                          TensorStorage::on_disk_dense,
                                          TensorStorage::on_disk_sym}) {
                    const bool mapped = storage == TensorStorage::on_disk_dense ||
                                        storage == TensorStorage::on_disk_sym;
                    const bool requests_symmetry =
                        storage == TensorStorage::in_memory_sym ||
                        storage == TensorStorage::on_disk_sym;
                    const bool compressed = fixture == 0 && requests_symmetry;
                    const TensorStorage expected_storage = compressed ? storage :
                        (mapped ? TensorStorage::on_disk_dense :
                                  TensorStorage::in_memory_dense);
                    config.tensors.storage = storage;
                    config.tensors.output_path = mapped
                        ? config.tensors.directory /
                              ("general_spin_" + std::to_string(checks) + ".sl2cxx")
                        : std::filesystem::path{};
                    configure(config);
                    const FullTensor actual = compute_full_tensor(
                        boundaries[fixture], cutoff, 1.2, candidate_backend);
                    const FullTensor canonical = actual.dense_copy();
                    const TensorError error = tensor_error(
                        canonical.span(), reference.span());
                    const bool metadata_matches =
                        actual.shape() == reference.shape() &&
                        actual.storage() == expected_storage &&
                        actual.is_mapped() == mapped &&
                        actual.path().empty() == !mapped &&
                        actual.is_symmetry_compressed() == compressed &&
                        (compressed ||
                         (actual.layout() == TensorLayout::CanonicalDenseColumnMajor &&
                          actual.stored_size() == actual.logical_size()));
                    const bool finite = std::all_of(
                        canonical.begin(), canonical.end(),
                        [](double value) { return std::isfinite(value); });
                    if (!metadata_matches || !finite ||
                        !(error.maximum_scaled <= 5.0e-10) ||
                        !(error.relative_l2 <= 5.0e-10)) {
                        std::ostringstream message;
                        message << "general-spin full tensor mismatch: fixture="
                                << fixture << " Dl=" << cutoff
                                << " simplicity=" << static_cast<int>(simplicity)
                                << " requested_storage=" << static_cast<int>(storage)
                                << " metadata=" << metadata_matches
                                << " max-scaled=" << error.maximum_scaled
                                << " relative-L2=" << error.relative_l2;
                        throw std::runtime_error(message.str());
                    }
                    maximum_scaled = std::max(maximum_scaled, error.maximum_scaled);
                    maximum_relative_l2 = std::max(
                        maximum_relative_l2, error.relative_l2);
                    ++checks;
                }
            }
        }
    }
    configure(initial);
    std::cout << "general-spin full tensor "
              << (candidate_backend == Backend::GPU ? "GPU" : "CPU")
              << " storage checks=" << checks
              << " max-scaled=" << maximum_scaled
              << " relative-L2=" << maximum_relative_l2 << '\n';
}

const sl2cfoamxx::BoundaryAngles euclidean_angles{{
    {{0.000000000000000,  0.000000000000000}}, // n12
    {{1.910633236249019,  0.000000000000000}}, // n13
    {{1.910633236249019,  2.094395102393196}}, // n14
    {{1.910633236249019, -2.094395102393196}}, // n15
    {{3.141592653589793,  0.000000000000000}}, // n21
    {{1.230959417340775, -1.823476581936975}}, // n23
    {{1.230959417340775,  0.270918520456220}}, // n24
    {{1.230959417340775,  2.365313622849416}}, // n25
    {{1.230959417340775,  3.141592653589793}}, // n31
    {{1.910633236249019,  1.318116071652818}}, // n32
    {{2.411864997362827, -1.435337066566786}}, // n34
    {{0.729727656226966, -0.388139515370189}}, // n35
    {{1.230959417340775, -1.047197551196598}}, // n41
    {{1.910633236249019, -2.870674133133573}}, // n42
    {{0.729727656226966,  1.706255587023007}}, // n43
    {{2.411864997362827,  0.659058035826409}}, // n45
    {{1.230959417340775,  1.047197551196598}}, // n51
    {{1.910633236249019, -0.776279030740378}}, // n52
    {{2.411864997362827,  2.753453138219605}}, // n53
    {{0.729727656226966, -2.482534617763384}}, // n54
}};

// Exact arbitrary normals from the formerly failing 0.4% coherent case.
const sl2cfoamxx::BoundaryAngles asymmetric_angles{{
    {{0.31, 0.13}}, {{0.77, -0.42}}, {{1.29, 1.07}}, {{2.11, 2.21}},
    {{2.39, 2.48}}, {{1.47, -1.14}}, {{0.43, -0.27}}, {{0.91, 0.61}},
    {{1.66, 1.52}}, {{2.52, -2.31}}, {{0.58, 0.36}}, {{1.04, -0.83}},
    {{1.18, 0.94}}, {{1.83, -1.73}}, {{2.67, 2.69}}, {{0.69, -0.48}},
    {{0.82, 0.57}}, {{1.96, 1.89}}, {{2.78, -2.52}}, {{1.33, -1.06}},
}};

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void require_booster_threads(const std::string& diagnostics, int expected_max)
{
    bool found = false;
    std::istringstream lines(diagnostics);
    for (std::string line; std::getline(lines, line);) {
        if (line.find("booster diagnostics harvest_calls=") == std::string::npos)
            continue;
        const auto field = [&](const std::string& key) {
            std::uint64_t value = 0;
            const auto position = line.find(key);
            if (position != std::string::npos) {
                std::istringstream input(line.substr(position + key.size()));
                input >> value;
            }
            return value;
        };
        const auto minimum = field("threads_min=");
        const auto maximum = field("threads_max=");
        require(field("harvest_calls=") > 0 && minimum > 0 && minimum <= maximum &&
                    maximum == static_cast<std::uint64_t>(expected_max),
                "booster harvesting did not use the requested OpenMP team: " + line);
        found = true;
    }
    require(found, "detailed coherent output omitted actual booster thread diagnostics");
}

template<class Exception, class Function>
void require_throws(Function&& function, const char* message)
{
    try { function(); }
    catch (const Exception&) { return; }
    throw std::runtime_error(message);
}

std::complex<double> contract_boundary(
    const sl2cfoamxx::FullTensor& tensor,
    const sl2cfoamxx::DSpins& spins,
    const sl2cfoamxx::BoundaryAngles& angles)
{
    using namespace sl2cfoamxx;
    constexpr std::array<std::array<std::size_t, 4>, 5> spin_rows{{
        {{0,1,2,3}}, {{4,5,6,0}}, {{7,8,1,4}}, {{9,2,5,7}}, {{3,6,8,9}}
    }};
    constexpr std::array<std::array<std::size_t, 4>, 5> angle_rows{{
        {{0,1,2,3}}, {{5,6,7,4}}, {{10,11,8,9}}, {{15,12,13,14}}, {{16,17,18,19}}
    }};
    std::array<CoherentCoefficients, 5> coefficients;
    for (std::size_t node = 0; node < 5; ++node) {
        TetrahedronSpins node_spins{};
        TetrahedronAngles node_angles{};
        for (std::size_t leg = 0; leg < 4; ++leg) {
            node_spins[leg] = spins[spin_rows[node][leg]];
            node_angles[leg] = angles[angle_rows[node][leg]];
        }
        coefficients[node] = compute_coherent_coefficients(node_spins, node_angles);
        require(coefficients[node].size() == tensor.shape()[node],
                "coherent coefficient axis does not match full tensor");
    }
    std::complex<double> result{};
    for (std::size_t flat = 0; flat < tensor.logical_size(); ++flat) {
        std::size_t remainder = flat;
        FullTensor::shape_type indices{};
        std::complex<double> product{1.0, 0.0};
        for (std::size_t axis = 0; axis < 5; ++axis) {
            indices[axis] = remainder % tensor.shape()[axis];
            remainder /= tensor.shape()[axis];
            product *= coefficients[axis][indices[axis]];
        }
        result += tensor.at(indices) * product;
    }
    return result;
}

void check_coherent_workflows(const sl2cfoamxx::LibraryConfig& initial)
{
    using namespace sl2cfoamxx;
    LibraryConfig config = initial;
    config.cpu_threads = 1;
    config.verbosity = Verbosity::silent;
    config.coherent_amplitude.pruning_ratio = 0.0;
    config.tensors.storage = TensorStorage::in_memory_dense;
    config.tensors.output_path.clear();
    config.tensors.reuse_existing = false;
    // Retain the former 0.4% discrepancy fixture, its rectangular variant,
    // and half-integer labels without compiling private contraction kernels.
    const std::array<DSpins, 3> boundaries{{
        {{2,1,1,2,1,1,2,1,1,1}},
        {{2,1,3,2,1,1,2,1,1,1}},
        {{1,1,1,1,1,1,1,1,1,1}}
    }};
    for (const auto simplicity : {SimplicityConstraint::gamma_j,
                                  SimplicityConstraint::gamma_j_p1}) {
        config.simplicity_constraint = simplicity;
        configure(config);
        for (const auto& spins : boundaries) {
            for (std::size_t cutoff = 0; cutoff <= 2; ++cutoff) {
                const auto tensor = compute_full_tensor(spins, cutoff, 1.2, Backend::CPU);
                const auto expected = contract_boundary(tensor, spins, asymmetric_angles);
                for (const auto backend : {Backend::CPU, Backend::GPU}) {
#if !SL2CFOAMXX_TEST_WITH_CUDA
                    if (backend == Backend::GPU) continue;
#endif
                    std::vector<ProgressEvent> events;
                    const auto value = compute_coherent_amplitude(
                        spins, cutoff, 1.2, asymmetric_angles, backend,
                        [&](const ProgressEvent& event) { events.push_back(event); });
                    require(relative_error(value, expected) <= 5.0e-10,
                            "coherent amplitude disagrees with contracted tensor");
                    require(events.size() >= 2 &&
                            events.front().status == ProgressStatus::started &&
                            events.back().status == ProgressStatus::completed &&
                            events.back().backend == backend &&
                            events.back().total > 0 &&
                            events.back().completed == events.back().total,
                            "coherent progress lifecycle did not complete");
                }
            }
        }
    }

    // Preserve independent sl2cfoam-next values from the former coherent suite.
    // These constants are data, and introduce no runtime oracle dependency.
    config.simplicity_constraint = SimplicityConstraint::gamma_j_p1;
    configure(config);
    const DSpins mixed{{1,1,0,0,1,0,0,0,0,0}};
    const DSpins half{{1,1,1,1,1,1,1,1,1,1}};
    const DSpins asymmetric{{2,4,6,8,4,6,8,10,12,14}};
    for (const auto backend : {Backend::CPU, Backend::GPU}) {
#if !SL2CFOAMXX_TEST_WITH_CUDA
        if (backend == Backend::GPU) continue;
#endif
        require(relative_error(compute_coherent_amplitude(
                    half, 2, 1.2, euclidean_angles, backend),
                    {-1.6748256353857123e-10, 1.2973143587300214e-10}) <= 5.0e-9,
                "independent half-integer coherent reference mismatch");
        require(relative_error(compute_coherent_amplitude(
                    asymmetric, 2, 1.2, euclidean_angles, backend),
                    {-3.5404738057802644e-19, -3.4513048451694775e-19}) <= 5.0e-9,
                "independent asymmetric coherent reference mismatch");
        require(relative_error(compute_full_tensor(mixed, 0, 1.2, backend).at(0,0,0,0,0),
                    8.549823132047453e-8) <= 1.0e-5,
                "independent mixed-parity tensor reference mismatch");
    }
    require(check_closure(half, euclidean_angles).all_closed,
            "known closed boundary failed closure");
    configure(initial);
    std::cout << "coherent contraction, references, closure, and progress: PASS\n";
}

void check_tensor_and_errors(const sl2cfoamxx::LibraryConfig& initial)
{
    using namespace sl2cfoamxx;
    FullTensor original(FullTensor::shape_type{{2,2,1,1,1}});
    original.at(1,1,0,0,0) = 3.0;
    auto copy = original.clone();
    copy.at(1,1,0,0,0) = 4.0;
    require(original.at(1,1,0,0,0) == 3.0, "tensor clone aliases its source");
    const auto file = initial.tensors.directory / "roundtrip.sl2cxx";
    original.save(file);
    const auto mapped = FullTensor::map(file);
    auto loaded = FullTensor::load(file);
    loaded.at(1,1,0,0,0) = 5.0;
    require(mapped.is_mapped() && mapped.is_read_only() &&
            mapped.at(1,1,0,0,0) == 3.0 && !loaded.is_mapped(),
            "tensor persistence or independent ownership failed");
    require_throws<std::out_of_range>([&] { (void)original.at(2,0,0,0,0); },
                                     "out-of-bounds tensor index accepted");
    DSpins bad{{1,1,1,1,1,1,1,1,1,1}};
    bad[0] = -1;
    require_throws<std::invalid_argument>([&] {
        (void)compute_full_tensor(bad, 0, 1.2, Backend::CPU);
    }, "negative spin accepted");
    const DSpins valid_spins{{1,1,1,1,1,1,1,1,1,1}};
    const DIntertwiners valid_intertwiners{{0,0,0,0,0}};
    for (const double gamma : {-1.2, -std::numeric_limits<double>::min()}) {
        require_throws<std::invalid_argument>([&] {
            (void)compute_amplitude(valid_spins, valid_intertwiners, 0, gamma);
        }, "negative gamma accepted by fixed amplitude");
        require_throws<std::invalid_argument>([&] {
            (void)compute_amplitude(valid_spins, valid_intertwiners, 0, gamma,
                                   AmplitudeCachePolicy::prefer_full_tensor);
        }, "negative gamma accepted by cached amplitude");
        for (const auto backend : {Backend::CPU, Backend::GPU}) {
            std::size_t progress_events = 0;
            const auto callback = [&](const ProgressEvent&) { ++progress_events; };
            require_throws<std::invalid_argument>([&] {
                (void)compute_full_tensor(valid_spins, 0, gamma, backend, callback);
            }, "negative gamma accepted by full tensor");
            require_throws<std::invalid_argument>([&] {
                (void)compute_coherent_amplitude(
                    valid_spins, 0, gamma, asymmetric_angles, backend, callback);
            }, "negative gamma accepted by coherent amplitude");
            require(progress_events == 0,
                    "negative gamma started a computation before validation");
        }
    }
    require(std::isfinite(compute_amplitude(DSpins{}, DIntertwiners{}, 0, 0.0)),
            "zero gamma boundary rejected");
    LibraryConfig invalid = initial;
    invalid.coherent_amplitude.pruning_ratio = -1.0;
    require_throws<std::invalid_argument>([&] { configure(invalid); },
                                         "invalid configuration accepted");
    const auto broken = initial.tensors.directory / "truncated.sl2cxx";
    { std::ofstream stream(broken, std::ios::binary); stream << "invalid"; }
    require_throws<std::exception>([&] { (void)FullTensor::load(broken); },
                                  "truncated tensor file accepted");
    configure(initial);
    std::cout << "tensor ownership, persistence, and public errors: PASS\n";
}

} // namespace

int main() try
{
    TemporaryDirectory cache;
    sl2cfoamxx::LibraryConfig config = sl2cfoamxx::default_config();
    if (!config.progress_bar ||
        config.tensors.storage != sl2cfoamxx::TensorStorage::on_disk_sym ||
        !sl2cfoamxx::LibraryConfig{}.progress_bar ||
        sl2cfoamxx::TensorCacheOptions{}.storage !=
            sl2cfoamxx::TensorStorage::on_disk_sym) {
        throw std::runtime_error("unexpected public configuration defaults");
    }
    config.tensors.directory = cache.path();
    config.simplicity_constraint =
        sl2cfoamxx::SimplicityConstraint::gamma_j;
    config.cpu_threads = 2;
    config.coherent_amplitude.pruning_ratio = 0.0;
    sl2cfoamxx::configure(config);

    const sl2cfoamxx::LibraryConfig active = sl2cfoamxx::current_config();
    std::cout << "tensor cache = " << active.tensors.directory << "\n";

    check_tensor_and_errors(active);
    check_coherent_workflows(active);

    // A small complete public workflow; large benchmarks are external consumers.
    const std::array<int, 10> spins{{2,2,2,2,2,2,2,2,2,2}};
    constexpr std::size_t Dl = 1;
    constexpr double gamma = 1.2;

    // Canonical directed-normal order:
    // n12,n13,n14,n15, n21,n23,n24,n25, n31,n32,n34,n35,
    // n41,n42,n43,n45, n51,n52,n53,n54.
    const sl2cfoamxx::BoundaryAngles angles{{
        {{1.570796326795,  0.000000000000}}, // n12
        {{1.570796326795,  1.910633236249}}, // n13
        {{0.615479708670, -2.186276035465}}, // n14
        {{2.526112944919, -2.186276035465}}, // n15
        {{1.570796326795,  3.141592653590}}, // n21
        {{1.570796326795,  0.585685543457}}, // n23
        {{1.071607449444, -0.320246032512}}, // n24
        {{2.069985204146, -0.320246032512}}, // n25
        {{1.570796326795, -1.230959417341}}, // n31
        {{1.570796326795,  1.324947692792}}, // n32
        {{1.071607449444,  2.230879268761}}, // n34
        {{2.069985204146,  2.230879268761}}, // n35
        {{2.526112944919,  0.955316618125}}, // n41
        {{0.573540745391, -1.105670539592}}, // n42
        {{0.573540745391,  3.016303775841}}, // n43
        {{1.201165252128, -2.186276035465}}, // n45
        {{0.615479708670,  0.955316618125}}, // n51
        {{2.568051908199, -1.105670539592}}, // n52
        {{2.568051908199,  3.016303775841}}, // n53
        {{1.940427401462, -2.186276035465}}, // n54
    }};

    std::cout << std::scientific << std::setprecision(17);

    // ---------------------------------------------------------------------
    // Coherent amplitude: CPU versus final GPU route.
    // ---------------------------------------------------------------------
    std::complex<double> coherent_cpu{};
    double coherent_cpu_seconds = 0.0;
    {
        // Raw startup environment must not cap an explicit public thread request.
        ScopedEnvironment omp_environment("OMP_NUM_THREADS", "1");
        ScopedEnvironment booster_environment("B4_NUM_THREADS", nullptr);
        std::ostringstream booster_diagnostics;
        auto threaded = active;
        threaded.cpu_threads = 2;
        threaded.tensors.reuse_existing = false;
        threaded.progress_bar = false;
        threaded.verbosity = sl2cfoamxx::Verbosity::detailed;
        threaded.log_stream = &booster_diagnostics;
        sl2cfoamxx::configure(threaded);
        auto run = timed([&] {
            return sl2cfoamxx::compute_coherent_amplitude(
                spins, Dl, gamma, angles, sl2cfoamxx::Backend::CPU);
        });
        coherent_cpu = run.first;
        coherent_cpu_seconds = run.second;
        sl2cfoamxx::configure(active);
        require_booster_threads(booster_diagnostics.str(),
                                std::min(2, omp_get_thread_limit()));

        // B4_NUM_THREADS remains an intentional booster-only ceiling.
        ScopedEnvironment booster_cap("B4_NUM_THREADS", "1");
        booster_diagnostics.str("");
        booster_diagnostics.clear();
        sl2cfoamxx::configure(threaded);
        const auto capped = sl2cfoamxx::compute_coherent_amplitude(
            spins, Dl, gamma, angles, sl2cfoamxx::Backend::CPU);
        sl2cfoamxx::configure(active);
        require_booster_threads(booster_diagnostics.str(), 1);
        require(relative_error(capped, coherent_cpu) <= 5.0e-10,
                "booster thread ceiling changed the coherent amplitude");
    }

    std::cout << "coherent CPU  = " << coherent_cpu
              << "  wall=" << coherent_cpu_seconds << " s\n";

#if SL2CFOAMXX_TEST_WITH_CUDA
    std::ostringstream coherent_gpu_tuning_diagnostics;
    sl2cfoamxx::LibraryConfig coherent_gpu_config = active;
    coherent_gpu_config.tensors.reuse_existing = false;
    coherent_gpu_config.low_level.coherent_gpu_pipeline_slots = 2;
    coherent_gpu_config.verbosity = sl2cfoamxx::Verbosity::detailed;
    coherent_gpu_config.log_stream = &coherent_gpu_tuning_diagnostics;
    sl2cfoamxx::configure(coherent_gpu_config);
    auto coherent_gpu_run = timed([&] {
        return sl2cfoamxx::compute_coherent_amplitude(
            spins, Dl, gamma, angles, sl2cfoamxx::Backend::GPU);
    });
    sl2cfoamxx::configure(active);
    const std::complex<double> coherent_gpu = coherent_gpu_run.first;
    const double coherent_error = relative_error(coherent_gpu, coherent_cpu);
    const std::string gpu_records = coherent_gpu_tuning_diagnostics.str();
    const auto unsigned_field = [](const std::string& line,
                                   const std::string& key) {
        const auto position = line.find(key);
        std::uint64_t value = 0;
        if (position != std::string::npos) {
            std::istringstream input(line.substr(position + key.size()));
            input >> value;
        }
        return value;
    };
    std::uint64_t reported_devices = 0, reported_slices = 0;
    std::uint64_t reported_workers = 0, observed_workers = 0, worker_slices = 0;
    bool placement_records_valid = true;
    std::istringstream gpu_lines(gpu_records);
    for (std::string line; std::getline(gpu_lines, line);) {
        if (line.find("GPU diagnostics devices=") != std::string::npos) {
            reported_devices = unsigned_field(line, "devices=");
            reported_slices = unsigned_field(line, "x_slices=");
            reported_workers = unsigned_field(line, "worker_records=");
        } else if (line.find("GPU worker device=") != std::string::npos) {
            ++observed_workers;
            worker_slices += unsigned_field(line, "x_slices=");
#if defined(__linux__)
            placement_records_valid &= unsigned_field(line, "thread_id=") > 0 &&
                line.find("cpus_allowed=unavailable") == std::string::npos &&
                line.find("mems_allowed=unavailable") == std::string::npos;
#endif
        }
    }
    const bool coherent_gpu_tuning_reported =
        gpu_records.find("coherent_gpu_pipeline_slots_effective=2") !=
            std::string::npos &&
        gpu_records.find("booster diagnostics harvest_calls=") !=
            std::string::npos &&
        reported_devices > 0 && reported_devices <= sl2cfoamxx::cuda_device_count() &&
        reported_workers >= reported_devices && observed_workers == reported_workers &&
        reported_slices > 0 && worker_slices == reported_slices &&
        placement_records_valid;

    std::cout << "coherent GPU  = " << coherent_gpu
              << "  wall=" << coherent_gpu_run.second << " s\n";
    std::cout << "coherent relative error = " << coherent_error << "\n\n";
#else
    const double coherent_error = 0.0;
    const bool coherent_gpu_tuning_reported = true;
    std::cout << "coherent GPU comparison = skipped (ENABLE_CUDA=OFF)\n\n";
#endif

    std::ostringstream default_backend_diagnostics;
    sl2cfoamxx::LibraryConfig default_backend_config = active;
    default_backend_config.verbosity = sl2cfoamxx::Verbosity::summary;
    default_backend_config.log_stream = &default_backend_diagnostics;
    sl2cfoamxx::configure(default_backend_config);
    auto coherent_default_run = timed([&] {
        return sl2cfoamxx::compute_coherent_amplitude(
            spins, Dl, gamma, angles);
    });
    sl2cfoamxx::configure(active);

#if SL2CFOAMXX_TEST_WITH_CUDA
    const std::complex<double> coherent_default_reference = coherent_gpu;
    const char* expected_default_backend = "started backend=GPU";
#else
    const std::complex<double> coherent_default_reference = coherent_cpu;
    const char* expected_default_backend = "started backend=CPU";
#endif
    const double coherent_default_error = relative_error(
        coherent_default_run.first, coherent_default_reference);
    const bool coherent_default_backend_selected =
        default_backend_diagnostics.str().find(expected_default_backend) !=
            std::string::npos &&
        default_backend_diagnostics.str().find("GPU worker device=") ==
            std::string::npos &&
        default_backend_diagnostics.str().find("GPU diagnostics devices=") ==
            std::string::npos &&
        default_backend_diagnostics.str().find("booster diagnostics harvest_calls=") ==
            std::string::npos;
    std::cout << "coherent configured-default = "
              << coherent_default_run.first
              << "  wall=" << coherent_default_run.second << " s\n";
    std::cout << "coherent configured-default relative error = "
              << coherent_default_error << "\n\n";

#if SL2CFOAMXX_TEST_WITH_CUDA
    std::ostringstream configured_cpu_diagnostics;
    sl2cfoamxx::LibraryConfig configured_cpu = active;
    configured_cpu.backend = sl2cfoamxx::Backend::CPU;
    configured_cpu.verbosity = sl2cfoamxx::Verbosity::summary;
    configured_cpu.log_stream = &configured_cpu_diagnostics;
    sl2cfoamxx::configure(configured_cpu);
    const std::complex<double> coherent_configured_cpu =
        sl2cfoamxx::compute_coherent_amplitude(
            spins, Dl, gamma, angles);
    sl2cfoamxx::configure(active);
    const double coherent_configured_cpu_error = relative_error(
        coherent_configured_cpu, coherent_cpu);
    const bool coherent_configured_cpu_selected =
        configured_cpu_diagnostics.str().find("started backend=CPU") !=
        std::string::npos;
    std::cout << "coherent config.backend=CPU relative error = "
              << coherent_configured_cpu_error << "\n\n";
#else
    const double coherent_configured_cpu_error = 0.0;
    const bool coherent_configured_cpu_selected = true;
#endif

    // ---------------------------------------------------------------------
    // One fixed-intertwiner amplitude.
    // ---------------------------------------------------------------------
    const std::array<int, 5> fixed{{2, 2, 2, 2, 2}};
    auto amplitude_run = timed([&] {
        return sl2cfoamxx::compute_amplitude(spins, fixed, Dl, gamma);
    });
    std::cout << "single amplitude = " << amplitude_run.first
              << "  wall=" << amplitude_run.second << " s\n\n";

    double maximum_rho_tile_error = 0.0;
    for (const std::uint32_t rho_tile : {4U, 8U, 16U}) {
        sl2cfoamxx::LibraryConfig tiled = active;
        tiled.low_level.booster_rho_tile = rho_tile;
        sl2cfoamxx::configure(tiled);
        const double tiled_amplitude =
            sl2cfoamxx::compute_amplitude(spins, fixed, Dl, gamma);
        require(std::isfinite(tiled_amplitude),
                "rho-tile amplitude is not finite");
        maximum_rho_tile_error = std::max(
            maximum_rho_tile_error,
            std::abs(tiled_amplitude - amplitude_run.first) /
                std::max({std::abs(tiled_amplitude),
                          std::abs(amplitude_run.first), 1.0e-300}));
    }
    sl2cfoamxx::configure(active);
    std::cout << "rho-tile maximum relative error = "
              << maximum_rho_tile_error << "\n\n";

    // ---------------------------------------------------------------------
    // Full tensor: optimized CPU versus final materialized GPU route.
    // ---------------------------------------------------------------------
    bool output_guard_rejected = false;
    try {
        sl2cfoamxx::LibraryConfig guarded = active;
        guarded.tensors.maximum_payload_size_bytes = sizeof(double);
        sl2cfoamxx::configure(guarded);
        (void)sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);
    } catch (const std::exception&) {
        output_guard_rejected = true;
    }
    sl2cfoamxx::configure(active);

    bool reserved_scratch_path_rejected = false;
    try {
        sl2cfoamxx::LibraryConfig invalid = active;
        invalid.tensors.storage = sl2cfoamxx::TensorStorage::on_disk_dense;
        invalid.tensors.output_path =
            active.tensors.directory / ".scratch" / "not-an-output.sl2cxx";
        sl2cfoamxx::configure(invalid);
        (void)sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);
    } catch (const std::exception&) {
        reserved_scratch_path_rejected = true;
    }
    sl2cfoamxx::configure(active);

    std::filesystem::create_directories(
        active.tensors.directory / ".scratch");
    std::filesystem::create_directory_symlink(
        active.tensors.directory / ".scratch",
        active.tensors.directory / "scratch_alias");
    bool symlinked_scratch_path_rejected = false;
    try {
        sl2cfoamxx::LibraryConfig invalid = active;
        invalid.tensors.storage = sl2cfoamxx::TensorStorage::on_disk_dense;
        invalid.tensors.output_path =
            active.tensors.directory / "scratch_alias" / "not-an-output.sl2cxx";
        sl2cfoamxx::configure(invalid);
        (void)sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);
    } catch (const std::exception&) {
        symlinked_scratch_path_rejected = true;
    }
    sl2cfoamxx::configure(active);

    const std::filesystem::path automatic_tensor_path =
        active.tensors.directory /
        "full_tensor" /
        "gamma_j" /
        "gamma_1.200" /
        "Dspins_2_2_2_2_2_2_2_2_2_2" /
        "Dl_1.sl2cxx";
    std::filesystem::create_directories(automatic_tensor_path.parent_path());

    sl2cfoamxx::LibraryConfig full_cpu_config = active;
    full_cpu_config.backend = sl2cfoamxx::Backend::CPU;
    full_cpu_config.tensors.storage =
        sl2cfoamxx::TensorStorage::on_disk_dense;
    full_cpu_config.tensors.output_path.clear();
    full_cpu_config.tensors.reuse_existing = false;
    sl2cfoamxx::configure(full_cpu_config);

    const std::filesystem::path stale_tensor_part =
        automatic_tensor_path.string() + ".part.stale";
    {
        std::ofstream stale(stale_tensor_part,
                            std::ios::binary | std::ios::trunc);
        stale.put('\0');
    }
    auto full_cpu_run = timed([&] {
        return sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);
    });

    full_cpu_config.tensors.reuse_existing = true;
    sl2cfoamxx::configure(full_cpu_config);
    sl2cfoamxx::FullTensor reused_cpu =
        sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);
    const bool mapped_cache_reused =
        reused_cpu.storage() == sl2cfoamxx::TensorStorage::on_disk_dense &&
        reused_cpu.is_mapped() &&
        reused_cpu.path() == full_cpu_run.first.path() &&
        reused_cpu.span().size() == full_cpu_run.first.span().size() &&
        std::equal(reused_cpu.begin(), reused_cpu.end(),
                   full_cpu_run.first.begin());
    const bool stale_tensor_part_removed =
        !std::filesystem::exists(stale_tensor_part);
    const bool automatic_storage_tree_restored =
        full_cpu_run.first.path() ==
            std::filesystem::weakly_canonical(automatic_tensor_path);

    std::cout << "full tensor CPU values = "
              << full_cpu_run.first.size()
              << "  wall=" << full_cpu_run.second << " s\n";

    // For doubled spins 2, each intertwiner axis is {0,2,4}; index 1 is the
    // fixed-intertwiner point used by compute_amplitude above.
    const double tensor_fixed_entry = full_cpu_run.first(1, 1, 1, 1, 1);
    const double fixed_entry_error =
        std::abs(tensor_fixed_entry - amplitude_run.first) /
        std::max({std::abs(tensor_fixed_entry),
                  std::abs(amplitude_run.first),
                  1.0e-300});
    std::cout << "full tensor fixed entry = " << tensor_fixed_entry
              << "  relative error=" << fixed_entry_error << "\n";

    // ---------------------------------------------------------------------
    // Alternative simplicity map: every public computation must use the
    // same configuration snapshot, and its full-tensor cache is independent.
    // ---------------------------------------------------------------------
    sl2cfoamxx::LibraryConfig gamma_j_p1_memory_config = active;
    gamma_j_p1_memory_config.simplicity_constraint =
        sl2cfoamxx::SimplicityConstraint::gamma_j_p1;
    gamma_j_p1_memory_config.backend = sl2cfoamxx::Backend::CPU;
    gamma_j_p1_memory_config.tensors.storage =
        sl2cfoamxx::TensorStorage::in_memory_dense;
    gamma_j_p1_memory_config.tensors.output_path.clear();
    sl2cfoamxx::configure(gamma_j_p1_memory_config);

    const double gamma_j_p1_amplitude =
        sl2cfoamxx::compute_amplitude(spins, fixed, Dl, gamma);
    const std::complex<double> gamma_j_p1_coherent =
        sl2cfoamxx::compute_coherent_amplitude(
            spins, Dl, gamma, angles, sl2cfoamxx::Backend::CPU);
    sl2cfoamxx::FullTensor gamma_j_p1_memory =
        sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);

    const bool memory_storage_reported =
        gamma_j_p1_memory.storage() ==
            sl2cfoamxx::TensorStorage::in_memory_dense &&
        !gamma_j_p1_memory.is_mapped() && gamma_j_p1_memory.path().empty();
    const double gamma_j_p1_fixed_entry =
        gamma_j_p1_memory(1, 1, 1, 1, 1);
    const double gamma_j_p1_fixed_entry_error =
        std::abs(gamma_j_p1_fixed_entry - gamma_j_p1_amplitude) /
        std::max({std::abs(gamma_j_p1_fixed_entry),
                  std::abs(gamma_j_p1_amplitude),
                  1.0e-300});

    // An exact output-path override cannot separate the two conventions by
    // directory name. With reuse enabled, its physics tag must reject a file
    // produced with the other convention without overwriting it.
    const std::filesystem::path explicit_collision_path =
        active.tensors.directory / "explicit_simplicity_collision.sl2cxx";
    full_cpu_run.first.save(explicit_collision_path);

    sl2cfoamxx::LibraryConfig explicit_gamma_j_p1_config =
        gamma_j_p1_memory_config;
    explicit_gamma_j_p1_config.tensors.storage =
        sl2cfoamxx::TensorStorage::on_disk_dense;
    explicit_gamma_j_p1_config.tensors.output_path = explicit_collision_path;
    explicit_gamma_j_p1_config.tensors.reuse_existing = true;
    sl2cfoamxx::configure(explicit_gamma_j_p1_config);
    bool explicit_collision_rejected = false;
    try {
        (void)sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);
    } catch (const std::exception& exception) {
        const std::string message = exception.what();
        explicit_collision_rejected =
            message.find("not reusable") != std::string::npos &&
            message.find("reuse_existing=false") != std::string::npos;
    }
    const sl2cfoamxx::FullTensor explicit_after_rejection =
        sl2cfoamxx::FullTensor::load(explicit_collision_path);
    const bool explicit_collision_preserved_gamma_j =
        explicit_after_rejection.size() == full_cpu_run.first.size() &&
        std::equal(explicit_after_rejection.begin(),
                   explicit_after_rejection.end(),
                   full_cpu_run.first.begin());

    // Explicitly authorize replacement, then restore reuse and require the
    // second gamma_j_p1 call to hit the newly written compatible file.
    explicit_gamma_j_p1_config.tensors.reuse_existing = false;
    sl2cfoamxx::configure(explicit_gamma_j_p1_config);
    sl2cfoamxx::FullTensor explicit_gamma_j_p1 =
        sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);

    explicit_gamma_j_p1_config.tensors.reuse_existing = true;
    sl2cfoamxx::configure(explicit_gamma_j_p1_config);
    sl2cfoamxx::FullTensor explicit_gamma_j_p1_reused =
        sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);

    const TensorError explicit_gamma_j_p1_error = tensor_error(
        explicit_gamma_j_p1.span(), gamma_j_p1_memory.span());
    const bool explicit_collision_replaced_and_reused =
        explicit_collision_rejected &&
        explicit_collision_preserved_gamma_j &&
        explicit_gamma_j_p1.storage() ==
            sl2cfoamxx::TensorStorage::on_disk_dense &&
        explicit_gamma_j_p1.path() ==
            std::filesystem::weakly_canonical(explicit_collision_path) &&
        explicit_gamma_j_p1.size() == full_cpu_run.first.size() &&
        !std::equal(explicit_gamma_j_p1.begin(), explicit_gamma_j_p1.end(),
                    full_cpu_run.first.begin()) &&
        explicit_gamma_j_p1_reused.path() == explicit_gamma_j_p1.path() &&
        explicit_gamma_j_p1_reused.size() == explicit_gamma_j_p1.size() &&
        std::equal(explicit_gamma_j_p1_reused.begin(),
                   explicit_gamma_j_p1_reused.end(),
                   explicit_gamma_j_p1.begin());

    const std::filesystem::path gamma_j_p1_automatic_path =
        active.tensors.directory /
        "full_tensor" /
        "gamma_j_p1" /
        "gamma_1.200" /
        "Dspins_2_2_2_2_2_2_2_2_2_2" /
        "Dl_1.sl2cxx";
    sl2cfoamxx::LibraryConfig gamma_j_p1_mapped_config =
        gamma_j_p1_memory_config;
    gamma_j_p1_mapped_config.tensors.storage =
        sl2cfoamxx::TensorStorage::on_disk_dense;
    gamma_j_p1_mapped_config.tensors.reuse_existing = false;
    sl2cfoamxx::configure(gamma_j_p1_mapped_config);
    sl2cfoamxx::FullTensor gamma_j_p1_mapped =
        sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);

    gamma_j_p1_mapped_config.tensors.reuse_existing = true;
    sl2cfoamxx::configure(gamma_j_p1_mapped_config);
    sl2cfoamxx::FullTensor gamma_j_p1_reused =
        sl2cfoamxx::compute_full_tensor(spins, Dl, gamma);

    const bool gamma_j_p1_tree_separated =
        gamma_j_p1_mapped.storage() ==
            sl2cfoamxx::TensorStorage::on_disk_dense &&
        gamma_j_p1_mapped.is_mapped() &&
        gamma_j_p1_mapped.path() ==
            std::filesystem::weakly_canonical(gamma_j_p1_automatic_path) &&
        gamma_j_p1_mapped.path() != full_cpu_run.first.path();
    const bool gamma_j_p1_cache_reused =
        gamma_j_p1_reused.storage() ==
            sl2cfoamxx::TensorStorage::on_disk_dense &&
        gamma_j_p1_reused.path() == gamma_j_p1_mapped.path() &&
        gamma_j_p1_reused.size() == gamma_j_p1_mapped.size() &&
        std::equal(gamma_j_p1_reused.begin(), gamma_j_p1_reused.end(),
                   gamma_j_p1_mapped.begin());
    const TensorError gamma_j_p1_storage_error = tensor_error(
        gamma_j_p1_mapped.span(), gamma_j_p1_memory.span());
    const bool simplicity_constraint_changed_results =
        gamma_j_p1_amplitude != amplitude_run.first &&
        gamma_j_p1_coherent != coherent_cpu &&
        gamma_j_p1_memory.size() == full_cpu_run.first.size() &&
        !std::equal(gamma_j_p1_memory.begin(), gamma_j_p1_memory.end(),
                    full_cpu_run.first.begin());

    std::cout << "gamma_j_p1 single amplitude = "
              << gamma_j_p1_amplitude << "\n";
    std::cout << "gamma_j_p1 coherent CPU = "
              << gamma_j_p1_coherent << "\n";
    std::cout << "gamma_j_p1 tensor fixed-entry relative error = "
              << gamma_j_p1_fixed_entry_error << "\n";

    // Restore the default-convention configuration before an optional GPU
    // comparison against the gamma_j CPU result.
    sl2cfoamxx::configure(full_cpu_config);

#if SL2CFOAMXX_TEST_WITH_CUDA
    std::ostringstream full_gpu_tuning_diagnostics;
    sl2cfoamxx::LibraryConfig full_gpu_config = active;
    // Deliberately leave the configured default at CPU: the explicit call
    // below must still select the GPU backend.
    full_gpu_config.backend = sl2cfoamxx::Backend::CPU;
    full_gpu_config.low_level.full_tensor_gpu_producer_waves_per_sm = 16;
    full_gpu_config.tensors.storage =
        sl2cfoamxx::TensorStorage::in_memory_dense;
    full_gpu_config.tensors.output_path.clear();
    full_gpu_config.verbosity = sl2cfoamxx::Verbosity::detailed;
    full_gpu_config.log_stream = &full_gpu_tuning_diagnostics;
    sl2cfoamxx::configure(full_gpu_config);
    auto full_gpu_run = timed([&] {
        return sl2cfoamxx::compute_full_tensor(
            spins, Dl, gamma, sl2cfoamxx::Backend::GPU);
    });

    const TensorError full_error = tensor_error(
        full_gpu_run.first.span(),
        full_cpu_run.first.span());
    const bool full_metadata_matches =
        full_gpu_run.first.shape() == full_cpu_run.first.shape() &&
        full_gpu_run.first.layout() == full_cpu_run.first.layout() &&
        full_gpu_run.first.storage() ==
            sl2cfoamxx::TensorStorage::in_memory_dense &&
        !full_gpu_run.first.is_mapped() && full_gpu_run.first.path().empty() &&
        full_gpu_run.first.layout() ==
            sl2cfoamxx::TensorLayout::CanonicalDenseColumnMajor;
    const bool full_gpu_tuning_reported =
        full_gpu_tuning_diagnostics.str().find(
            "started backend=GPU") != std::string::npos &&
        full_gpu_tuning_diagnostics.str().find(
            "full_tensor_gpu_producer_waves_per_sm_request=16") !=
            std::string::npos &&
        full_gpu_tuning_diagnostics.str().find(
            "full_tensor_gpu_producer_waves_per_sm_effective=16") !=
            std::string::npos;

    std::cout << "full tensor GPU values = "
              << full_gpu_run.first.size()
              << "  wall=" << full_gpu_run.second << " s\n";
    std::cout << "full tensor max-scaled error = "
              << full_error.maximum_scaled << "\n";
    std::cout << "full tensor relative-L2 error = "
              << full_error.relative_l2 << "\n";
#else
    const TensorError full_error{};
    const bool full_metadata_matches = true;
    const bool full_gpu_tuning_reported = true;
    std::cout << "full tensor GPU comparison = skipped (ENABLE_CUDA=OFF)\n";
#endif

#if SL2CFOAMXX_TEST_WITH_CUDA
    check_general_spin_tensor_storage(active, sl2cfoamxx::Backend::GPU);
#else
    check_general_spin_tensor_storage(active, sl2cfoamxx::Backend::CPU);
#endif

    constexpr double tolerance = 5.0e-10;
    const bool no_wigner_cache_created =
        !std::filesystem::exists(cache.path() / "3j_table") &&
        !std::filesystem::exists(
            cache.path() / ".locks" / "3j_table.lock");
    const bool cpu_results_valid =
        std::isfinite(abs(coherent_cpu)) &&
        no_wigner_cache_created &&
        std::isfinite(amplitude_run.first) &&
        !full_cpu_run.first.empty() &&
        full_cpu_run.first.is_mapped() &&
        mapped_cache_reused &&
        memory_storage_reported &&
        gamma_j_p1_tree_separated &&
        gamma_j_p1_cache_reused &&
        explicit_collision_replaced_and_reused &&
        simplicity_constraint_changed_results &&
        stale_tensor_part_removed &&
        automatic_storage_tree_restored &&
        output_guard_rejected &&
        reserved_scratch_path_rejected &&
        symlinked_scratch_path_rejected &&
        full_cpu_run.first.layout() ==
            sl2cfoamxx::TensorLayout::CanonicalDenseColumnMajor &&
        std::all_of(full_cpu_run.first.shape().begin(),
                    full_cpu_run.first.shape().end(),
                    [](std::size_t extent) { return extent > 0; }) &&
        full_cpu_run.first.size() ==
            full_cpu_run.first.shape()[0] *
                full_cpu_run.first.shape()[1] *
                full_cpu_run.first.shape()[2] *
                full_cpu_run.first.shape()[3] *
                full_cpu_run.first.shape()[4];

    const bool passed = cpu_results_valid &&
                        full_metadata_matches &&
                        fixed_entry_error <= tolerance &&
                        maximum_rho_tile_error <= tolerance &&
                        gamma_j_p1_fixed_entry_error <= tolerance &&
                        gamma_j_p1_storage_error.maximum_scaled <= tolerance &&
                        gamma_j_p1_storage_error.relative_l2 <= tolerance &&
                        explicit_gamma_j_p1_error.maximum_scaled <= tolerance &&
                        explicit_gamma_j_p1_error.relative_l2 <= tolerance &&
                        coherent_error <= tolerance &&
                        coherent_default_error <= tolerance &&
                        coherent_default_backend_selected &&
                        coherent_gpu_tuning_reported &&
                        coherent_configured_cpu_error <= tolerance &&
                        coherent_configured_cpu_selected &&
                        full_gpu_tuning_reported &&
                        full_error.maximum_scaled <= tolerance &&
                        full_error.relative_l2 <= tolerance;

#if SL2CFOAMXX_TEST_WITH_CUDA
    std::cout << "\npublic API CPU/GPU comparison: ";
#else
    std::cout << "\npublic API CPU smoke test: ";
#endif
    std::cout << (passed ? "PASS" : "FAIL") << "\n";
    return passed ? 0 : 1;
}

catch (const std::exception& error)
{
    std::cerr << "public API: FAIL: " << error.what() << '\n';
    return 1;
}
