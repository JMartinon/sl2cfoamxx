// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include <sl2cfoamxx.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <array>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;
using namespace sl2cfoamxx;

namespace {

nb::tuple shape_tuple(const FullTensor::shape_type& shape)
{
    return nb::make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4]);
}

nb::object optional_path(const std::filesystem::path& path)
{
    return path.empty() ? nb::none() : nb::cast(path);
}

// Use Python's integer-index protocol: accept NumPy integer scalars, but never
// silently truncate a floating-point coordinate. Coordinates retain the C++
// API's nonnegative, zero-based convention.
std::size_t checked_index(nb::handle value)
{
    nb::object integer = nb::steal<nb::object>(PyNumber_Index(value.ptr()));
    if (!integer.is_valid()) {
        throw nb::python_error();
    }
    const unsigned long long index = PyLong_AsUnsignedLongLong(integer.ptr());
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw nb::index_error("full-tensor index is negative or too large");
    }
    if (index > std::numeric_limits<std::size_t>::max()) {
        throw nb::index_error("full-tensor index is too large");
    }
    return static_cast<std::size_t>(index);
}

template <typename Sequence>
FullTensor::shape_type checked_indices(const Sequence& indices)
{
    if (nb::len(indices) != FullTensor::rank) {
        throw nb::index_error("a full tensor requires exactly five indices");
    }
    FullTensor::shape_type result{};
    for (std::size_t axis = 0; axis < result.size(); ++axis) {
        result[axis] = checked_index(indices[axis]);
    }
    return result;
}

nb::object numpy_view(FullTensor& tensor, nb::handle owner, bool canonical)
{
    const std::array<std::size_t, 1> physical_shape{tensor.stored_size()};
    const std::size_t rank = canonical ? FullTensor::rank : 1;
    const std::size_t* shape = canonical
        ? tensor.shape().data() : physical_shape.data();

    // NumPy dimensions and DLPack strides are signed. Tensor already checks
    // the size_t shape product, but an empty tensor can still have a huge axis.
    for (std::size_t axis = 0; axis < rank; ++axis) {
        if (shape[axis] > static_cast<std::size_t>(
                std::numeric_limits<Py_ssize_t>::max())) {
            throw std::overflow_error("tensor dimension does not fit NumPy");
        }
    }
    if (tensor.stored_bytes() > static_cast<std::size_t>(
            std::numeric_limits<Py_ssize_t>::max())) {
        throw std::overflow_error("tensor payload does not fit NumPy");
    }
    if (canonical) {
        for (const std::size_t stride : tensor.strides()) {
            if (stride > static_cast<std::size_t>(
                    std::numeric_limits<Py_ssize_t>::max()) / sizeof(double)) {
                throw std::overflow_error("tensor byte stride does not fit NumPy");
            }
        }
    }

    // Array ownership pins the FullTensor object. The Python API intentionally
    // omits clear, swap, and assignment, which could invalidate these pointers.
    // Const scalar export also prevents NumPy from re-enabling write access to
    // a read-only file mapping.
    if (tensor.is_read_only()) {
        const FullTensor& constant = tensor;
        return nb::ndarray<nb::numpy, const double, nb::f_contig>(
            constant.data(), rank, shape, owner).cast();
    }
    return nb::ndarray<nb::numpy, double, nb::f_contig>(
        tensor.data(), rank, shape, owner).cast();
}

nb::object tensor_numpy(FullTensor& tensor, bool copy, bool canonical)
{
    if (canonical && !copy && !tensor.is_dense()) {
        throw nb::value_error(
            "reassociated full tensors have no canonical zero-copy NumPy "
            "view; use numpy(copy=True) or stored_numpy()");
    }
    if (!copy) {
        nb::object owner = nb::find(&tensor);
        if (!owner.is_valid()) {
            throw std::runtime_error("cannot find the Python tensor owner");
        }
        return numpy_view(tensor, owner, canonical);
    }

    // The copy is an independently owned C++ tensor, so expansion needs only
    // one payload allocation and preserves the existing canonical copy path.
    auto duplicate = std::make_unique<FullTensor>(
        canonical ? tensor.dense_copy() : tensor.clone());
    nb::capsule owner(duplicate.get(), [](void* pointer) noexcept {
        delete static_cast<FullTensor*>(pointer);
    });
    FullTensor& result = *duplicate.release();
    return numpy_view(result, owner, canonical);
}

nb::object tensor_array(
    FullTensor& tensor, nb::object dtype, std::optional<bool> copy)
{
    // NumPy's array protocol always describes the logical canonical tensor,
    // never the physical reassociated payload exposed by stored_numpy().
    const bool forbid_copy = copy.has_value() && !*copy;
    if (forbid_copy && !tensor.is_dense()) {
        throw nb::value_error(
            "copy=False cannot expand a reassociated full tensor; "
            "allow a copy or use stored_numpy() for its physical payload");
    }

    bool convert_dtype = false;
    if (!dtype.is_none()) {
        const nb::object numpy = nb::module_::import_("numpy");
        dtype = numpy.attr("dtype")(dtype);
        const nb::object native_dtype = numpy.attr("dtype")("float64");
        const int same_dtype = PyObject_RichCompareBool(
            dtype.ptr(), native_dtype.ptr(), Py_EQ);
        if (same_dtype < 0) throw nb::python_error();
        convert_dtype = same_dtype == 0;
        if (forbid_copy && convert_dtype) {
            throw nb::value_error(
                "copy=False cannot convert the full tensor's float64 dtype");
        }
    }

    // astype already allocates for a different dtype. Avoid making a second
    // full-sized copy of a dense tensor before that conversion. Compact data
    // still expands through the public canonical dense-copy implementation.
    nb::object array = tensor_numpy(
        tensor, !tensor.is_dense() || (copy.value_or(false) && !convert_dtype),
        true);
    if (convert_dtype) {
        return array.attr("astype")(dtype, "order"_a = "F", "copy"_a = false);
    }
    return array;
}

std::string format_tensor(
    const FullTensor& tensor, const FullTensorPrintOptions& options)
{
    std::ostringstream output;
    tensor.print(output, options);
    return output.str();
}

} // namespace

void bind_tensor(nb::module_& module)
{
    nb::class_<FullTensorScientificMetadata>(module, "FullTensorScientificMetadata")
        .def_ro("spins", &FullTensorScientificMetadata::spins)
        .def_ro("Dl", &FullTensorScientificMetadata::Dl)
        .def_ro("gamma", &FullTensorScientificMetadata::gamma)
        .def_ro("simplicity_constraint",
                &FullTensorScientificMetadata::simplicity_constraint);

    nb::class_<FullTensorMetadata>(module, "FullTensorMetadata")
        .def_ro("scientific", &FullTensorMetadata::scientific)
        .def_prop_ro("logical_shape", [](const FullTensorMetadata& metadata) {
            return shape_tuple(metadata.logical_shape);
        })
        .def_ro("logical_values", &FullTensorMetadata::logical_values)
        .def_ro("stored_values", &FullTensorMetadata::stored_values)
        .def_ro("storage", &FullTensorMetadata::storage)
        .def_ro("layout", &FullTensorMetadata::layout)
        .def_ro("reassociated_layout", &FullTensorMetadata::reassociated_layout)
        .def_ro("mapped", &FullTensorMetadata::mapped)
        .def_ro("writable", &FullTensorMetadata::writable)
        .def_prop_ro("path", [](const FullTensorMetadata& metadata) {
            return optional_path(metadata.path);
        });

    nb::class_<FullTensorPrintOptions>(module, "FullTensorPrintOptions")
        .def(nb::init<>())
        .def_rw("max_values", &FullTensorPrintOptions::max_values)
        .def_rw("precision", &FullTensorPrintOptions::precision)
        .def_rw("include_metadata", &FullTensorPrintOptions::include_metadata)
        .def_prop_ro_static("all_values", [](nb::handle) {
            return FullTensorPrintOptions::all_values;
        });

    nb::class_<FullTensor>(module, "FullTensor",
        "A rank-five logical full tensor. All indices are nonnegative and "
        "zero-based. Tuple indexing addresses canonical (i1,i2,i3,i4,i5) "
        "coordinates; at_flat() addresses physical stored values.")
        .def(nb::init<>())
        .def(nb::init<const FullTensor::shape_type&, TensorInit, TensorLayout>(),
             "shape"_a, "init"_a = TensorInit::zero,
             "layout"_a = TensorLayout::canonical_dense_column_major)
        .def_static("create_mapped", [](
            const std::filesystem::path& file,
            const FullTensor::shape_type& shape,
            TensorInit init, TensorLayout layout) {
                return FullTensor::create_mapped(file, shape, init, layout);
            }, "file"_a, "shape"_a, "init"_a = TensorInit::zero,
            "layout"_a = TensorLayout::canonical_dense_column_major)
        .def_static("create_file", [](
            const std::filesystem::path& file,
            const FullTensor::shape_type& shape,
            TensorInit init, TensorLayout layout) {
                return FullTensor::create_file(file, shape, init, layout);
            }, "file"_a, "shape"_a, "init"_a = TensorInit::zero,
            "layout"_a = TensorLayout::canonical_dense_column_major)
        .def_static("load", &FullTensor::load, "file"_a)
        .def_static("map", &FullTensor::map, "file"_a,
                    "access"_a = TensorAccess::read_only)
        .def_static("open_file_existing", &FullTensor::open_file_existing,
                    "file"_a, "read_only"_a = false)
        .def("save", &FullTensor::save, "file"_a)
        .def("flush", &FullTensor::flush)
        .def("convert_storage", &FullTensor::convert_storage,
             "destination_storage"_a, "destination_path"_a,
             "overwrite_existing"_a = false)
        .def("clone", &FullTensor::clone,
             "Return an independent writable copy in owned memory.")
        .def("dense_copy", &FullTensor::dense_copy,
             "Return an independent canonical dense tensor in owned memory.")
        .def("fill", &FullTensor::fill, "value"_a)
        .def("zero", &FullTensor::zero)
        .def("metadata", &FullTensor::metadata)
        .def("format", &format_tensor,
             "options"_a = FullTensorPrintOptions{},
             "Return the bounded canonical diagnostic dump as a string.")
        .def("print", [](const FullTensor& tensor,
                          const FullTensorPrintOptions& options) {
            nb::module_::import_("builtins").attr("print")(
                format_tensor(tensor, options), "end"_a = "");
        }, "options"_a = FullTensorPrintOptions{})
        .def("__getitem__", [](const FullTensor& tensor, nb::tuple indices) {
            return tensor.at(checked_indices(indices));
        }, "indices"_a)
        .def("__setitem__", [](
            FullTensor& tensor, nb::tuple indices, double value) {
            tensor.at(checked_indices(indices)) = value;
        }, "indices"_a, "value"_a)
        .def("at", [](const FullTensor& tensor, nb::sequence indices) {
            return tensor.at(checked_indices(indices));
        }, "indices"_a)
        .def("at", [](const FullTensor& tensor, nb::handle i1,
                      nb::handle i2, nb::handle i3,
                      nb::handle i4, nb::handle i5) {
            return tensor.at(FullTensor::shape_type{
                checked_index(i1), checked_index(i2), checked_index(i3),
                checked_index(i4), checked_index(i5)});
        }, "i1"_a, "i2"_a, "i3"_a, "i4"_a, "i5"_a)
        // Retain the former keyword spellings for existing Python callers.
        .def("at", [](const FullTensor& tensor, nb::handle i1,
                      nb::handle i2, nb::handle i3,
                      nb::handle i4, nb::handle i5) {
            return tensor.at(FullTensor::shape_type{
                checked_index(i1), checked_index(i2), checked_index(i3),
                checked_index(i4), checked_index(i5)});
        }, "a1"_a, "a2"_a, "a3"_a, "a4"_a, "a5"_a)
        .def("set_at", [](
            FullTensor& tensor, nb::sequence indices, double value) {
            tensor.at(checked_indices(indices)) = value;
        }, "indices"_a, "value"_a)
        .def("at_flat", [](const FullTensor& tensor, nb::handle index) {
            return tensor.at_flat(checked_index(index));
        }, "index"_a)
        .def("set_at_flat", [](
            FullTensor& tensor, nb::handle index, double value) {
            tensor.at_flat(checked_index(index)) = value;
        }, "index"_a, "value"_a)
        .def("__array__", &tensor_array,
             "dtype"_a = nb::none(), "copy"_a = nb::none(),
             "Convert to a canonical NumPy array for asarray, einsum, and "
             "other NumPy operations. copy=None shares dense float64 storage "
             "or expands reassociated storage; copy=True returns independent "
             "storage. copy=False rejects expansion and dtype conversion. "
             "Views retain their owner and read-only mapping status.")
        .def("numpy", [](FullTensor& tensor, bool copy) {
            return tensor_numpy(tensor, copy, true);
        }, "copy"_a = false,
             "Return a float64 array in canonical (i1,i2,i3,i4,i5) order, "
             "with i1 contiguous (Fortran order). Dense tensors support a "
             "zero-copy view; reassociated tensors require copy=True. "
             "Views retain the tensor and its read-only mapping status.")
        .def("stored_numpy", [](FullTensor& tensor, bool copy) {
            return tensor_numpy(tensor, copy, false);
        }, "copy"_a = false,
             "Return the physical float64 payload as a one-dimensional "
             "NumPy array. Reassociated payloads are not in canonical order. "
             "copy=True returns independent writable storage.")
        .def_prop_ro_static("rank", [](nb::handle) { return FullTensor::rank; })
        .def_prop_ro("shape", [](const FullTensor& tensor) {
            return shape_tuple(tensor.shape());
        })
        .def_prop_ro("dims", [](const FullTensor& tensor) {
            return shape_tuple(tensor.dims());
        })
        .def_prop_ro("strides", [](const FullTensor& tensor) {
            return shape_tuple(tensor.strides());
        }, "Logical canonical strides, measured in elements.")
        .def("extent", [](const FullTensor& tensor, nb::handle axis) {
            return tensor.extent(checked_index(axis));
        }, "axis"_a)
        .def_prop_ro("size", &FullTensor::size)
        .def_prop_ro("stored_size", &FullTensor::stored_size)
        .def_prop_ro("logical_size", &FullTensor::logical_size)
        .def_prop_ro("dense_size", &FullTensor::dense_size)
        .def_prop_ro("size_bytes", &FullTensor::size_bytes)
        .def_prop_ro("stored_bytes", &FullTensor::stored_bytes)
        .def_prop_ro("dense_bytes", &FullTensor::dense_bytes)
        .def_prop_ro("bytes", &FullTensor::bytes)
        .def_prop_ro("empty", &FullTensor::empty)
        .def_prop_ro("is_dense", &FullTensor::is_dense)
        .def_prop_ro("is_reassociated", &FullTensor::is_reassociated)
        .def_prop_ro("is_symmetry_compressed", &FullTensor::is_symmetry_compressed)
        .def_prop_ro("reassociated_layout", &FullTensor::reassociated_layout)
        .def_prop_ro("storage", &FullTensor::storage)
        .def_prop_ro("is_mapped", &FullTensor::is_mapped)
        .def_prop_ro("writable", &FullTensor::writable)
        .def_prop_ro("is_read_only", &FullTensor::is_read_only)
        .def_prop_ro("path", [](const FullTensor& tensor) {
            return optional_path(tensor.path());
        })
        .def_prop_ro("root_path", [](const FullTensor& tensor) {
            return optional_path(tensor.root_path());
        })
        .def_prop_ro("layout", &FullTensor::layout)
        .def_prop_ro("layout_code", &FullTensor::layout_code)
        .def_prop_ro("tag", [](const FullTensor& tensor) {
            return nb::bytes(tensor.tag().data(), tensor.tag().size());
        })
        .def("__repr__", [](const FullTensor& tensor) {
            std::ostringstream output;
            output << "FullTensor(shape=(";
            for (std::size_t axis = 0; axis < FullTensor::rank; ++axis) {
                if (axis != 0) output << ", ";
                output << tensor.extent(axis);
            }
            output << "), storage=" << tensor.storage()
                   << ", writable=" << (tensor.writable() ? "True" : "False")
                   << ')';
            return output.str();
        });
}
