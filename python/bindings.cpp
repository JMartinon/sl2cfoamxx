// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include <sl2cfoamxx.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <exception>
#include <memory>
#include <optional>

namespace nb = nanobind;
namespace s = sl2cfoamxx;
using namespace nb::literals;

void bind_tensor(nb::module_& m);

namespace {

std::optional<std::filesystem::path> optional_path(const std::filesystem::path& path)
{
    if (path.empty()) return std::nullopt;
    return path;
}

// The Python argument owns the callable until the synchronous C++ call and
// its reporter thread have finished. Copies of this callback contain no Python
// references, so worker teardown never decrements a refcount without the GIL.
s::ProgressCallback progress_callback(const nb::object& callable)
{
    if (callable.is_none()) return {};
    if (!PyCallable_Check(callable.ptr())) {
        throw nb::type_error("progress must be callable or None");
    }
    return [pointer = callable.ptr()](const s::ProgressEvent& event) {
        nb::gil_scoped_acquire acquire;
        try {
            // A callback may keep the event after this invocation returns.
            nb::borrow<nb::object>(pointer)(nb::cast(event, nb::rv_policy::copy));
        } catch (nb::python_error& error) {
            // Match the native API's non-fatal callback contract while making
            // the failure observable through Python's sys.unraisablehook.
            error.discard_as_unraisable("sl2cfoamxx progress callback");
        }
    };
}

void bind_errors(nb::module_& m)
{
    nb::enum_<s::ErrorCode>(m, "ErrorCode")
        .value("backend_unavailable", s::ErrorCode::backend_unavailable)
        .value("backend_failure", s::ErrorCode::backend_failure)
        .value("invalid_tensor_file", s::ErrorCode::invalid_tensor_file)
        .value("cache_failure", s::ErrorCode::cache_failure)
        .value("resource_limit", s::ErrorCode::resource_limit)
        .value("numerical_failure", s::ErrorCode::numerical_failure)
        .value("internal_failure", s::ErrorCode::internal_failure)
        .value("unsupported_operation", s::ErrorCode::unsupported_operation);

    nb::exception<s::Error> error_type(m, "Error", PyExc_RuntimeError);
    nb::register_exception_translator(
        [](const std::exception_ptr& exception, void* payload) {
            try {
                std::rethrow_exception(exception);
            } catch (const s::Error& error) {
                auto type = nb::handle(static_cast<PyObject*>(payload));
                nb::object instance = type(nb::str(error.what()));
                instance.attr("code") = nb::cast(error.code());
                PyErr_SetObject(type.ptr(), instance.ptr());
            } catch (const std::filesystem::filesystem_error& error) {
                nb::object instance = nb::handle(PyExc_OSError)(
                    error.code().value(), nb::str(error.what()),
                    error.path1().empty() ? nb::none() : nb::cast(error.path1()),
                    nb::none(),
                    error.path2().empty() ? nb::none() : nb::cast(error.path2()));
                PyErr_SetObject(PyExc_OSError, instance.ptr());
            } catch (const std::length_error& error) {
                PyErr_SetString(PyExc_OverflowError, error.what());
            }
        }, error_type.ptr());
}

void bind_enums(nb::module_& m)
{
    nb::enum_<s::Backend>(m, "Backend")
        .value("CPU", s::Backend::CPU).value("GPU", s::Backend::GPU);
    nb::enum_<s::TensorStorage>(m, "TensorStorage")
        .value("in_memory_sym", s::TensorStorage::in_memory_sym)
        .value("in_memory_dense", s::TensorStorage::in_memory_dense)
        .value("on_disk_sym", s::TensorStorage::on_disk_sym)
        .value("on_disk_dense", s::TensorStorage::on_disk_dense);
    nb::enum_<s::SimplicityConstraint>(m, "SimplicityConstraint")
        .value("gamma_j", s::SimplicityConstraint::gamma_j)
        .value("gamma_j_p1", s::SimplicityConstraint::gamma_j_p1);
    nb::enum_<s::AmplitudeCachePolicy>(m, "AmplitudeCachePolicy")
        .value("disabled", s::AmplitudeCachePolicy::disabled)
        .value("prefer_full_tensor", s::AmplitudeCachePolicy::prefer_full_tensor);
    nb::enum_<s::Verbosity>(m, "Verbosity")
        .value("silent", s::Verbosity::silent)
        .value("summary", s::Verbosity::summary)
        .value("timing", s::Verbosity::timing)
        .value("detailed", s::Verbosity::detailed);
    nb::enum_<s::PrintingStyle>(m, "PrintingStyle")
        .value("UTF8", s::PrintingStyle::UTF8)
        .value("ASCII", s::PrintingStyle::ASCII);
    m.attr("printing_style") = m.attr("PrintingStyle");
    nb::enum_<s::FullTensorCpuFinalTile>(m, "FullTensorCpuFinalTile")
        .value("automatic", s::FullTensorCpuFinalTile::automatic)
        .value("tile_4x4", s::FullTensorCpuFinalTile::tile_4x4)
        .value("tile_4x8", s::FullTensorCpuFinalTile::tile_4x8)
        .value("tile_8x4", s::FullTensorCpuFinalTile::tile_8x4);
    nb::enum_<s::TensorInit>(m, "TensorInit")
        .value("zero", s::TensorInit::zero)
        .value("uninitialized", s::TensorInit::uninitialized);
    nb::enum_<s::TensorAccess>(m, "TensorAccess")
        .value("read_only", s::TensorAccess::read_only)
        .value("read_write", s::TensorAccess::read_write);
    auto tensor_layout = nb::enum_<s::TensorLayout>(m, "TensorLayout");
    tensor_layout
        .value("generic_column_major", s::TensorLayout::generic_column_major)
        .value("canonical_dense_column_major", s::TensorLayout::canonical_dense_column_major)
        .value("full_tensor_reassociated_by_i1", s::TensorLayout::full_tensor_reassociated_by_i1);
    tensor_layout.attr("full_tensor_reassociated_by_a1") =
        tensor_layout.attr("full_tensor_reassociated_by_i1");
    auto reassociated_layout = nb::enum_<s::FullTensorReassociatedLayout>(
        m, "FullTensorReassociatedLayout");
    reassociated_layout
        .value("reflection_triangular_by_i1", s::FullTensorReassociatedLayout::reflection_triangular_by_i1)
        .value("rectangular_by_i1", s::FullTensorReassociatedLayout::rectangular_by_i1);
    reassociated_layout.attr("reflection_triangular_by_a1") =
        reassociated_layout.attr("reflection_triangular_by_i1");
    reassociated_layout.attr("rectangular_by_a1") =
        reassociated_layout.attr("rectangular_by_i1");
}

void bind_config(nb::module_& m)
{
    nb::class_<s::TensorCacheOptions>(m, "TensorCacheOptions")
        .def(nb::init<>())
        .def_prop_rw("directory", [](const s::TensorCacheOptions& options) {
            return optional_path(options.directory);
        }, [](s::TensorCacheOptions& options, std::optional<std::filesystem::path> path) {
            options.directory = path.value_or(std::filesystem::path{});
        })
        .def_rw("storage", &s::TensorCacheOptions::storage)
        .def_prop_rw("output_path", [](const s::TensorCacheOptions& options) {
            return optional_path(options.output_path);
        }, [](s::TensorCacheOptions& options, std::optional<std::filesystem::path> path) {
            options.output_path = path.value_or(std::filesystem::path{});
        })
        .def_rw("reuse_existing", &s::TensorCacheOptions::reuse_existing)
        .def_rw("maximum_payload_size_bytes", &s::TensorCacheOptions::maximum_payload_size_bytes);
    nb::class_<s::LowLevelOptions>(m, "LowLevelOptions")
        .def(nb::init<>())
        .def_rw("booster_rho_tile", &s::LowLevelOptions::booster_rho_tile)
        .def_rw("full_tensor_gpu_row_capacity", &s::LowLevelOptions::full_tensor_gpu_row_capacity)
        .def_rw("coherent_gpu_pipeline_slots", &s::LowLevelOptions::coherent_gpu_pipeline_slots)
        .def_rw("full_tensor_gpu_producer_waves_per_sm", &s::LowLevelOptions::full_tensor_gpu_producer_waves_per_sm)
        .def_rw("full_tensor_cpu_p_batch", &s::LowLevelOptions::full_tensor_cpu_p_batch)
        .def_rw("full_tensor_cpu_final_tile", &s::LowLevelOptions::full_tensor_cpu_final_tile);
    nb::class_<s::CoherentAmplitudeOptions>(m, "CoherentAmplitudeOptions")
        .def(nb::init<>())
        .def_rw("check_closure", &s::CoherentAmplitudeOptions::check_closure)
        .def_rw("pruning_ratio", &s::CoherentAmplitudeOptions::pruning_ratio);
    nb::class_<s::LibraryConfig>(m, "LibraryConfig",
        "Library settings. Publish changes with configure(); use per-call progress callbacks.")
        .def(nb::init<>())
        .def_rw("tensors", &s::LibraryConfig::tensors)
        .def_rw("backend", &s::LibraryConfig::backend)
        .def_rw("progress_bar", &s::LibraryConfig::progress_bar)
        .def_rw("simplicity_constraint", &s::LibraryConfig::simplicity_constraint)
        .def_rw("cpu_threads", &s::LibraryConfig::cpu_threads)
        .def_rw("memory_budget_bytes", &s::LibraryConfig::memory_budget_bytes)
        .def_rw("verbosity", &s::LibraryConfig::verbosity)
        .def_rw("printing_style", &s::LibraryConfig::printing_style)
        .def_rw("low_level", &s::LibraryConfig::low_level)
        .def_rw("coherent_amplitude", &s::LibraryConfig::coherent_amplitude);
    m.def("default_config", &s::default_config);
    m.def("current_config", &s::current_config, nb::call_guard<nb::gil_scoped_release>());
    m.def("configure", [](s::LibraryConfig config) {
        // Copy under the GIL before publishing. Another Python thread may
        // mutate the original config while filesystem work runs without it.
        nb::gil_scoped_release release;
        s::configure(config);
    }, "config"_a);
}

void bind_inspection(nb::module_& m)
{
    nb::class_<s::BuildInfo>(m, "BuildInfo")
        .def_ro("version", &s::BuildInfo::version)
        .def_ro("compiler_id", &s::BuildInfo::compiler_id)
        .def_ro("compiler_version", &s::BuildInfo::compiler_version)
        .def_ro("build_type", &s::BuildInfo::build_type)
        .def_ro("blas_vendor_selector", &s::BuildInfo::blas_vendor_selector)
        .def_ro("blas_provider", &s::BuildInfo::blas_provider)
        .def_ro("native_arch_enabled", &s::BuildInfo::native_arch_enabled)
        .def_ro("cuda_backend_built", &s::BuildInfo::cuda_backend_built)
        .def_ro("cuda_fast_math_requested", &s::BuildInfo::cuda_fast_math_requested)
        .def_ro("cuda_toolkit_version", &s::BuildInfo::cuda_toolkit_version)
        .def_ro("cuda_architectures", &s::BuildInfo::cuda_architectures)
        .def_ro("cutensor_version", &s::BuildInfo::cutensor_version);
    nb::class_<s::CudaDeviceInfo>(m, "CudaDeviceInfo")
        .def_ro("ordinal", &s::CudaDeviceInfo::ordinal)
        .def_ro("name", &s::CudaDeviceInfo::name)
        .def_ro("compute_capability_major", &s::CudaDeviceInfo::compute_capability_major)
        .def_ro("compute_capability_minor", &s::CudaDeviceInfo::compute_capability_minor)
        .def_ro("total_memory_bytes", &s::CudaDeviceInfo::total_memory_bytes)
        .def_ro("multiprocessor_count", &s::CudaDeviceInfo::multiprocessor_count)
        .def_ro("warp_size", &s::CudaDeviceInfo::warp_size)
        .def_ro("max_threads_per_block", &s::CudaDeviceInfo::max_threads_per_block)
        .def_ro("pci_bus_id", &s::CudaDeviceInfo::pci_bus_id);
    m.def("version", &s::version);
    m.def("build_info", &s::build_info);
    m.def("backend_available", &s::backend_available, "backend"_a,
          nb::call_guard<nb::gil_scoped_release>());
    m.def("cuda_device_count", &s::cuda_device_count,
          nb::call_guard<nb::gil_scoped_release>());
    m.def("cuda_device_info", &s::cuda_device_info, "ordinal"_a,
          nb::call_guard<nb::gil_scoped_release>());

    nb::enum_<s::ProgressOperation>(m, "ProgressOperation")
        .value("coherent_amplitude", s::ProgressOperation::coherent_amplitude)
        .value("full_tensor", s::ProgressOperation::full_tensor);
    nb::enum_<s::ProgressPhase>(m, "ProgressPhase")
        .value("contraction", s::ProgressPhase::contraction)
        .value("streamed_computation", s::ProgressPhase::streamed_computation);
    nb::enum_<s::ProgressStatus>(m, "ProgressStatus")
        .value("started", s::ProgressStatus::started)
        .value("running", s::ProgressStatus::running)
        .value("restarted", s::ProgressStatus::restarted)
        .value("completed", s::ProgressStatus::completed)
        .value("stopped", s::ProgressStatus::stopped);
    nb::class_<s::ProgressEvent>(m, "ProgressEvent")
        .def_ro("operation", &s::ProgressEvent::operation)
        .def_ro("phase", &s::ProgressEvent::phase)
        .def_ro("backend", &s::ProgressEvent::backend)
        .def_ro("status", &s::ProgressEvent::status)
        .def_ro("attempt", &s::ProgressEvent::attempt)
        .def_ro("completed", &s::ProgressEvent::completed)
        .def_ro("total", &s::ProgressEvent::total)
        .def_ro("elapsed_seconds", &s::ProgressEvent::elapsed_seconds)
        .def_ro("eta_seconds", &s::ProgressEvent::eta_seconds)
        .def_ro("eta_available", &s::ProgressEvent::eta_available);

    nb::class_<s::ClosureNodeResult>(m, "ClosureNodeResult")
        .def_prop_ro("vector", [](const s::ClosureNodeResult& node) {
            return nb::make_tuple(node.vector[0], node.vector[1], node.vector[2]);
        })
        .def_ro("residual_norm", &s::ClosureNodeResult::residual_norm)
        .def_ro("scale", &s::ClosureNodeResult::scale)
        .def_ro("relative_residual", &s::ClosureNodeResult::relative_residual)
        .def_ro("closed", &s::ClosureNodeResult::closed);
    nb::class_<s::ClosureCheckResult>(m, "ClosureCheckResult")
        .def_ro("nodes", &s::ClosureCheckResult::nodes)
        .def_ro("all_closed", &s::ClosureCheckResult::all_closed)
        .def("__bool__", [](const s::ClosureCheckResult& result) { return result.all_closed; });
}

void bind_computations(nb::module_& m)
{
    m.def("indices_from_intertwiners", [](const s::DSpins& spins,
                                           const s::DIntertwiners& intertwiners) {
        const auto indices = s::indices_from_intertwiners(spins, intertwiners);
        return nb::make_tuple(indices[0], indices[1], indices[2], indices[3], indices[4]);
    }, "spins"_a, "intertwiners"_a);
    m.def("check_closure", &s::check_closure,
          "spins"_a, "angles"_a, "absolute_tolerance"_a = 1e-12,
          "relative_tolerance"_a = 1e-12, nb::call_guard<nb::gil_scoped_release>());
    m.def("compute_coherent_coefficients", [](const s::TetrahedronDSpins& spins,
                                               const s::TetrahedronAngles& angles) {
        auto coefficients = std::make_unique<s::CoherentCoefficients>();
        {
            nb::gil_scoped_release release;
            *coefficients = s::compute_coherent_coefficients(spins, angles);
        }
        auto* owned = coefficients.get();
        nb::capsule owner(owned, [](void* pointer) noexcept {
            delete static_cast<s::CoherentCoefficients*>(pointer);
        });
        coefficients.release();
        return nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>, nb::device::cpu>(
            owned->data(), {owned->size()}, owner);
    }, "spins"_a, "angles"_a,
       "Return complex128 coefficients ordered by ascending admissible doubled intertwiner.");
    m.def("compute_amplitude",
          nb::overload_cast<const s::DSpins&, const s::DIntertwiners&,
                            std::size_t, double, s::AmplitudeCachePolicy>(&s::compute_amplitude),
          "spins"_a, "intertwiners"_a, "Dl"_a, "gamma"_a,
          "cache_policy"_a = s::AmplitudeCachePolicy::disabled,
          nb::call_guard<nb::gil_scoped_release>(),
          "Compute one CPU amplitude. Spins and intertwiners are doubled integers.");
    m.def("compute_coherent_amplitude", [](const s::DSpins& spins,
        std::size_t Dl, double gamma, const s::BoundaryAngles& angles,
        std::optional<s::Backend> backend, const nb::object& progress) {
        auto callback = progress_callback(progress);
        const bool has_progress = !progress.is_none();
        if (has_progress && !backend) {
            throw nb::type_error("an explicit backend is required with progress");
        }
        nb::gil_scoped_release release;
        if (!has_progress) {
            return backend ? s::compute_coherent_amplitude(spins, Dl, gamma, angles, *backend)
                           : s::compute_coherent_amplitude(spins, Dl, gamma, angles);
        }
        return s::compute_coherent_amplitude(spins, Dl, gamma, angles, *backend, std::move(callback));
    }, "spins"_a, "Dl"_a, "gamma"_a, "angles"_a, "backend"_a = nb::none(),
       nb::kw_only(), "progress"_a = nb::none(),
       "Compute a coherent amplitude from 20 (theta, phi) rows in canonical boundary order.");
    m.def("compute_full_tensor", [](const s::DSpins& spins,
        std::size_t Dl, double gamma, std::optional<s::Backend> backend,
        const nb::object& progress) {
        auto callback = progress_callback(progress);
        const bool has_progress = !progress.is_none();
        if (has_progress && !backend) {
            throw nb::type_error("an explicit backend is required with progress");
        }
        nb::gil_scoped_release release;
        if (!has_progress) {
            return backend ? s::compute_full_tensor(spins, Dl, gamma, *backend)
                           : s::compute_full_tensor(spins, Dl, gamma);
        }
        return s::compute_full_tensor(spins, Dl, gamma, *backend, std::move(callback));
    }, "spins"_a, "Dl"_a, "gamma"_a, "backend"_a = nb::none(),
       nb::kw_only(), "progress"_a = nb::none(),
       "Compute a logical rank-five tensor using the configured storage policy.");
}

} // namespace

NB_MODULE(_sl2cfoamxx, m)
{
    m.doc() = "Python bindings for the public sl2cfoam++ C++ API.";
    bind_enums(m);
    bind_errors(m);
    bind_config(m);
    bind_inspection(m);
    bind_tensor(m);
    bind_computations(m);
    m.attr("__version__") = nb::cast(s::version());
}
