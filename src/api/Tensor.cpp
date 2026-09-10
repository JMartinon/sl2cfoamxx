// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sl2cfoamxx/tensor.hpp"

#include <atomic>
#include <cerrno>
#include <fstream>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace sl2cfoamxx {
namespace {

[[nodiscard]] std::filesystem::filesystem_error system_failure(
    const std::string& operation,
    const std::filesystem::path& path)
{
    const int error_number = errno != 0 ? errno : EIO;
    return std::filesystem::filesystem_error(
        operation + " failed",
        path,
        std::error_code(error_number, std::generic_category()));
}

[[nodiscard]] std::filesystem::path temporary_sibling(
    const std::filesystem::path& destination)
{
    static std::atomic<std::uint64_t> sequence{0};
#if defined(_WIN32)
    const auto process = static_cast<std::uint64_t>(_getpid());
#else
    const auto process = static_cast<std::uint64_t>(::getpid());
#endif
    return destination.string() + ".tmp." + std::to_string(process) + "." +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

} // namespace

namespace tensor_disk {

TensorTreeHeaderV1 read_header_file(const std::filesystem::path& file)
{
    TensorTreeHeaderV1 header{};
    errno = 0;
    std::ifstream input(file, std::ios::binary);
    if (!input) {
        throw system_failure("open tensor file", file);
    }
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "tensor file does not contain a complete header: " +
            file.string());
    }
    if (!std::equal(magic.begin(), magic.end(), header.magic)) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "tensor file has invalid magic: " + file.string());
    }
    if (header.version != version) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "tensor file has an unsupported version: " + file.string());
    }
    return header;
}

} // namespace tensor_disk

namespace detail {

MappedRegion::~MappedRegion()
{
#if !defined(_WIN32)
    if (data_ != nullptr && size_ != 0) {
        if (writable_) (void)::msync(data_, size_, MS_ASYNC);
        (void)::munmap(data_, size_);
    }
    if (descriptor_ >= 0) (void)::close(descriptor_);
#endif
}

std::unique_ptr<MappedRegion> MappedRegion::create(
    const std::filesystem::path& file,
    std::size_t bytes)
{
#if defined(_WIN32)
    (void)file;
    (void)bytes;
    throw Error(
        ErrorCode::unsupported_operation,
        "memory-mapped tensors are not implemented on Windows; use save/load");
#else
    if (bytes > static_cast<std::size_t>(
                    std::numeric_limits<off_t>::max())) {
        throw std::overflow_error("mapped tensor file exceeds off_t");
    }
    if (const auto parent = file.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    auto region = std::make_unique<MappedRegion>();
    region->path_ = file;
    region->writable_ = true;
    region->descriptor_ = ::open(
        file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (region->descriptor_ < 0) {
        throw system_failure("open", file);
    }
    if (::ftruncate(region->descriptor_, static_cast<off_t>(bytes)) != 0) {
        throw system_failure("ftruncate", file);
    }
    region->size_ = bytes;
    if (bytes != 0) {
        region->data_ = ::mmap(
            nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
            region->descriptor_, 0);
        if (region->data_ == MAP_FAILED) {
            region->data_ = nullptr;
            throw system_failure("mmap", file);
        }
    }
    return region;
#endif
}

std::unique_ptr<MappedRegion> MappedRegion::open(
    const std::filesystem::path& file,
    TensorAccess access)
{
#if defined(_WIN32)
    (void)file;
    (void)access;
    throw Error(
        ErrorCode::unsupported_operation,
        "memory-mapped tensors are not implemented on Windows; use load");
#else
    const std::uint64_t bytes64 = checked_file_size(file);
    if (bytes64 > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("mapped tensor file does not fit size_t");
    }

    auto region = std::make_unique<MappedRegion>();
    region->path_ = file;
    region->writable_ = access == TensorAccess::read_write;
    region->descriptor_ = ::open(
        file.c_str(), region->writable_ ? O_RDWR : O_RDONLY);
    if (region->descriptor_ < 0) {
        throw system_failure("open", file);
    }
    region->size_ = static_cast<std::size_t>(bytes64);
    if (region->size_ != 0) {
        const int protection = region->writable_
            ? PROT_READ | PROT_WRITE
            : PROT_READ;
        region->data_ = ::mmap(
            nullptr, region->size_, protection, MAP_SHARED,
            region->descriptor_, 0);
        if (region->data_ == MAP_FAILED) {
            region->data_ = nullptr;
            throw system_failure("mmap", file);
        }
    }
    return region;
#endif
}

void MappedRegion::flush() const
{
#if defined(_WIN32)
    throw Error(
        ErrorCode::unsupported_operation,
        "memory-mapped tensors are not implemented on Windows");
#else
    if (data_ != nullptr && size_ != 0 && writable_ &&
        ::msync(data_, size_, MS_SYNC) != 0) {
        throw system_failure("msync", path_);
    }
#endif
}

void write_tensor_file_atomic(
    const std::filesystem::path& file,
    const tensor_disk::TensorTreeHeaderV1& header,
    std::uint64_t payload_offset,
    const void* payload,
    std::size_t payload_bytes)
{
    if (payload_offset < sizeof(header)) {
        throw std::invalid_argument("tensor payload overlaps its header");
    }
    if (payload_bytes != 0 && payload == nullptr) {
        throw std::invalid_argument("non-empty tensor has a null payload");
    }
    if (payload_offset > static_cast<std::uint64_t>(
                             std::numeric_limits<std::streamoff>::max())) {
        throw std::overflow_error("tensor payload offset exceeds streamoff");
    }
    if (payload_bytes > static_cast<std::size_t>(
                            std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error("tensor payload exceeds streamsize");
    }

    if (const auto parent = file.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    const std::filesystem::path temporary = temporary_sibling(file);

    try {
        std::ofstream output(
            temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::filesystem::filesystem_error(
                "failed to create tensor file",
                temporary,
                std::make_error_code(std::errc::io_error));
        }
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        output.seekp(static_cast<std::streamoff>(payload_offset));
        if (payload_bytes != 0) {
            output.write(
                static_cast<const char*>(payload),
                static_cast<std::streamsize>(payload_bytes));
        } else {
            // Keep the file length consistent with the stored payload offset.
            output.put('\0');
        }
        output.flush();
        if (!output) {
            throw std::filesystem::filesystem_error(
                "failed while writing tensor file",
                temporary,
                std::make_error_code(std::errc::io_error));
        }
        output.close();

        if (!output) {
            throw std::filesystem::filesystem_error(
                "failed while closing tensor file",
                temporary,
                std::make_error_code(std::errc::io_error));
        }

        std::error_code rename_error;
        std::filesystem::rename(temporary, file, rename_error);
        if (rename_error) {
#if defined(_WIN32)
            std::filesystem::remove(file, rename_error);
            rename_error.clear();
            std::filesystem::rename(temporary, file, rename_error);
#endif
            if (rename_error) {
                throw std::filesystem::filesystem_error(
                    "failed to install tensor file",
                    temporary,
                    file,
                    rename_error);
            }
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void read_tensor_file_payload(
    const std::filesystem::path& file,
    std::uint64_t payload_offset,
    void* payload,
    std::size_t payload_bytes)
{
    if (payload_bytes == 0) return;
    if (payload == nullptr) {
        throw std::invalid_argument("non-empty tensor has a null destination");
    }
    if (payload_offset > static_cast<std::uint64_t>(
                             std::numeric_limits<std::streamoff>::max()) ||
        payload_bytes > static_cast<std::size_t>(
                            std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error("tensor payload exceeds stream limits");
    }

    errno = 0;
    std::ifstream input(file, std::ios::binary);
    if (!input) {
        throw system_failure("open tensor file", file);
    }
    input.seekg(static_cast<std::streamoff>(payload_offset));
    input.read(static_cast<char*>(payload),
               static_cast<std::streamsize>(payload_bytes));
    if (input.gcount() != static_cast<std::streamsize>(payload_bytes)) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "tensor payload is truncated: " + file.string());
    }
}

std::uint64_t checked_file_size(const std::filesystem::path& file)
{
    std::error_code error;
    const std::uintmax_t bytes = std::filesystem::file_size(file, error);
    if (error) {
        throw std::filesystem::filesystem_error(
            "failed to inspect tensor file", file, error);
    }
    if (bytes > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("tensor file size exceeds uint64_t");
    }
    return static_cast<std::uint64_t>(bytes);
}

} // namespace detail
} // namespace sl2cfoamxx
