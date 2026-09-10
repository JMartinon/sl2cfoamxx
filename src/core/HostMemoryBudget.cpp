// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/HostMemoryBudget.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#  include <sys/types.h>
#  include <sys/sysctl.h>
#elif defined(__linux__)
#  include <sys/sysinfo.h>
#  include <unistd.h>
#endif

namespace b4support::host_memory {

void min_present(std::uint64_t value,
                        bool present,
                        std::uint64_t& aggregate,
                        bool& aggregate_present) noexcept
{
    if (!present) return;
    if (!aggregate_present || value < aggregate) aggregate = value;
    aggregate_present = true;
}

namespace detail {

inline std::optional<std::uint64_t> parse_u64(const std::string& text) noexcept
{
    const char* begin = text.c_str();
    while (*begin == ' ' || *begin == '\t' || *begin == '\n' || *begin == '\r') ++begin;
    if (*begin == '\0' || *begin == '-') return std::nullopt;
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(begin, &end, 10);
    if (errno != 0 || end == begin) return std::nullopt;
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') ++end;
    if (*end != '\0') return std::nullopt;
    return static_cast<std::uint64_t>(value);
}

inline std::optional<std::uint64_t> read_u64_file(const std::string& path)
{
    std::ifstream input(path);
    std::string text;
    if (!input || !std::getline(input, text) || text == "max") return std::nullopt;
    return parse_u64(text);
}

inline std::optional<std::string> read_first_line(const std::string& path)
{
    std::ifstream input(path);
    std::string text;
    if (!input || !std::getline(input, text)) return std::nullopt;
    return text;
}

// Linux exposes CPU and NUMA-node masks as comma-separated indexes and
// inclusive ranges (for example, "0-7,16-23"). Keep this parser private and
// bounded: procfs/sysfs data is advisory and must never make planning throw or
// spend unbounded time expanding a malformed range.
inline std::optional<std::vector<unsigned>> parse_index_list(
    std::string_view text)
{
    constexpr std::uint64_t maximum_index = 1u << 20;
    std::vector<unsigned> values;
    std::size_t cursor = 0;

    const auto skip_space = [&]() {
        while (cursor < text.size() &&
               (text[cursor] == ' ' || text[cursor] == '\t' ||
                text[cursor] == '\n' || text[cursor] == '\r')) {
            ++cursor;
        }
    };
    const auto parse_index = [&]() -> std::optional<std::uint64_t> {
        skip_space();
        if (cursor == text.size() || text[cursor] < '0' || text[cursor] > '9') {
            return std::nullopt;
        }
        std::uint64_t value = 0;
        while (cursor < text.size() && text[cursor] >= '0' && text[cursor] <= '9') {
            const unsigned digit = static_cast<unsigned>(text[cursor] - '0');
            if (value > (maximum_index - digit) / 10u) return std::nullopt;
            value = value * 10u + digit;
            ++cursor;
        }
        return value;
    };

    skip_space();
    if (cursor == text.size()) return std::nullopt;
    while (cursor < text.size()) {
        const auto first = parse_index();
        if (!first) return std::nullopt;
        std::uint64_t last = *first;
        skip_space();
        if (cursor < text.size() && text[cursor] == '-') {
            ++cursor;
            const auto parsed_last = parse_index();
            if (!parsed_last || *parsed_last < *first) return std::nullopt;
            last = *parsed_last;
        }
        if (last > maximum_index || last - *first > maximum_index) {
            return std::nullopt;
        }
        const std::uint64_t span = last - *first + 1;
        if (span > maximum_index + 1 -
                       static_cast<std::uint64_t>(values.size())) {
            return std::nullopt;
        }
        for (std::uint64_t value = *first; value <= last; ++value) {
            values.push_back(static_cast<unsigned>(value));
        }
        skip_space();
        if (cursor == text.size()) break;
        if (text[cursor] != ',') return std::nullopt;
        ++cursor;
        skip_space();
        if (cursor == text.size()) return std::nullopt;
    }

    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

struct NumaNodeMemory {
    unsigned index = 0;
    std::uint64_t total_bytes = 0;
    std::uint64_t available_bytes = 0;
    bool has_available = false;
};

struct NumaAffinityCapacity {
    std::uint64_t total_bytes = 0;
    std::uint64_t available_bytes = 0;
    bool has_available = false;
};

inline bool sorted_contains(const std::vector<unsigned>& values,
                            unsigned value) noexcept
{
    return std::binary_search(values.begin(), values.end(), value);
}

// Apply the calling context's cpuset memory-node constraint. We deliberately
// do not use Cpus_allowed_list here: Linux reports the calling thread's CPU
// mask, and an OpenMP master may already be pinned to one place while workers
// legitimately span several NUMA nodes. Mems_allowed_list is the reliable
// memory-placement constraint for the calling execution context and is
// therefore a conservative hard cap for automatic planning.
inline std::optional<NumaAffinityCapacity> select_numa_affinity_capacity(
    const std::vector<NumaNodeMemory>& nodes,
    const std::optional<std::vector<unsigned>>& allowed_mems) noexcept
{
    NumaAffinityCapacity result;
    result.has_available = true;
    for (const NumaNodeMemory& node : nodes) {
        if (node.total_bytes == 0) continue;
        if (allowed_mems &&
            !sorted_contains(*allowed_mems, node.index)) {
            continue;
        }
        result.total_bytes = sat_add(result.total_bytes, node.total_bytes);
        if (node.has_available) {
            result.available_bytes = sat_add(
                result.available_bytes, node.available_bytes);
        } else {
            result.has_available = false;
        }
    }
    if (result.total_bytes == 0) return std::nullopt;
    if (!result.has_available) result.available_bytes = 0;
    return result;
}

inline std::string unescape_mount_field(const std::string& text)
{
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\\' && i + 3 < text.size() &&
            text[i + 1] >= '0' && text[i + 1] <= '7' &&
            text[i + 2] >= '0' && text[i + 2] <= '7' &&
            text[i + 3] >= '0' && text[i + 3] <= '7') {
            const unsigned value =
                static_cast<unsigned>(text[i + 1] - '0') * 64u +
                static_cast<unsigned>(text[i + 2] - '0') * 8u +
                static_cast<unsigned>(text[i + 3] - '0');
            out.push_back(static_cast<char>(value));
            i += 3;
        } else {
            out.push_back(text[i]);
        }
    }
    return out;
}

inline bool controller_list_contains(const std::string& controllers,
                                     const std::string& wanted)
{
    std::size_t begin = 0;
    while (begin <= controllers.size()) {
        const std::size_t end = controllers.find(',', begin);
        const std::size_t count = end == std::string::npos
            ? controllers.size() - begin
            : end - begin;
        if (controllers.compare(begin, count, wanted) == 0) return true;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return false;
}

struct CgroupMembership {
    std::string v2_path;
    std::string v1_memory_path;
};

inline CgroupMembership read_cgroup_membership()
{
    CgroupMembership membership;
    std::ifstream input("/proc/self/cgroup");
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t first = line.find(':');
        if (first == std::string::npos) continue;
        const std::size_t second = line.find(':', first + 1);
        if (second == std::string::npos) continue;
        const std::string controllers = line.substr(first + 1, second - first - 1);
        const std::string path = line.substr(second + 1);
        if (controllers.empty()) membership.v2_path = path;
        if (controller_list_contains(controllers, "memory")) {
            membership.v1_memory_path = path;
        }
    }
    return membership;
}

struct CgroupMount {
    std::string root;
    std::string mount_point;
    bool v2 = false;
};

inline std::vector<CgroupMount> read_cgroup_mounts()
{
    std::vector<CgroupMount> mounts;
    std::ifstream input("/proc/self/mountinfo");
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t separator = line.find(" - ");
        if (separator == std::string::npos) continue;

        std::istringstream left(line.substr(0, separator));
        std::vector<std::string> fields;
        std::string field;
        while (left >> field) fields.push_back(field);
        if (fields.size() < 5) continue;

        std::istringstream right(line.substr(separator + 3));
        std::string fs_type;
        std::string source;
        std::string super_options;
        right >> fs_type >> source >> super_options;
        if (fs_type != "cgroup2" && fs_type != "cgroup") continue;
        if (fs_type == "cgroup" &&
            !controller_list_contains(super_options, "memory") &&
            source.find("memory") == std::string::npos) {
            continue;
        }
        mounts.push_back(CgroupMount{
            unescape_mount_field(fields[3]),
            unescape_mount_field(fields[4]),
            fs_type == "cgroup2"});
    }
    return mounts;
}

inline bool path_has_prefix(const std::string& path, const std::string& root)
{
    if (root == "/") return !path.empty() && path.front() == '/';
    if (path.size() < root.size() || path.compare(0, root.size(), root) != 0) return false;
    return path.size() == root.size() || path[root.size()] == '/';
}

inline std::string resolve_cgroup_path(const CgroupMount& mount,
                                       const std::string& membership_path)
{
    // Outside a cgroup namespace, membership_path normally contains mount.root
    // and we subtract it. Inside a namespace, /proc/self/cgroup is relative to
    // the namespace root (often just "/") while mountinfo retains a non-root
    // internal root. In that case membership_path is already relative to the
    // visible mount and must not be rejected.
    std::string relative;
    if (path_has_prefix(membership_path, mount.root)) {
        relative = membership_path.substr(mount.root == "/" ? 1 : mount.root.size());
    } else {
        relative = membership_path;
    }
    while (!relative.empty() && relative.front() == '/') relative.erase(relative.begin());
    if (relative.empty()) return mount.mount_point;
    return mount.mount_point + "/" + relative;
}

inline std::string parent_path(const std::string& path)
{
    if (path.empty() || path == "/") return path;
    const std::size_t end = path.find_last_not_of('/');
    if (end == std::string::npos) return "/";
    const std::size_t slash = path.find_last_of('/', end);
    if (slash == std::string::npos) return {};
    return slash == 0 ? std::string("/") : path.substr(0, slash);
}

inline bool finite_cgroup_limit(std::uint64_t value,
                                std::uint64_t physical_total) noexcept
{
    // cgroup v1 represents "unlimited" with a page-rounded value close to
    // INT64_MAX. No real node addressed by this program is remotely this large.
    if (value >= (1ull << 60)) return false;
    if (physical_total != 0 && value > sat_mul(physical_total, 2)) return false;
    return true;
}

inline void add_cgroup_constraint(std::uint64_t limit,
                                  const std::optional<std::uint64_t>& usage,
                                  std::uint64_t physical_total,
                                  Snapshot& snapshot) noexcept
{
    if (!finite_cgroup_limit(limit, physical_total)) return;
    min_present(limit, true, snapshot.cgroup_limit_bytes, snapshot.has_cgroup_limit);
    if (usage) {
        const std::uint64_t headroom = *usage < limit ? limit - *usage : 0;
        min_present(headroom, true,
                    snapshot.cgroup_headroom_bytes,
                    snapshot.has_cgroup_headroom);
    }
}

inline void probe_cgroups(Snapshot& snapshot)
{
    const CgroupMembership membership = read_cgroup_membership();
    const std::vector<CgroupMount> mounts = read_cgroup_mounts();
    for (const CgroupMount& mount : mounts) {
        const std::string& member_path = mount.v2
            ? membership.v2_path
            : membership.v1_memory_path;
        if (member_path.empty()) continue;
        std::string current = resolve_cgroup_path(mount, member_path);
        if (current.empty()) continue;

        while (path_has_prefix(current, mount.mount_point)) {
            if (mount.v2) {
                const std::optional<std::uint64_t> usage =
                    read_u64_file(current + "/memory.current");
                if (const auto limit = read_u64_file(current + "/memory.max")) {
                    add_cgroup_constraint(*limit, usage,
                                          snapshot.physical_total_bytes, snapshot);
                }
            } else {
                const std::optional<std::uint64_t> usage =
                    read_u64_file(current + "/memory.usage_in_bytes");
                if (const auto limit = read_u64_file(current + "/memory.limit_in_bytes")) {
                    add_cgroup_constraint(*limit, usage,
                                          snapshot.physical_total_bytes, snapshot);
                }
            }

            if (current == mount.mount_point) break;
            const std::string parent = parent_path(current);
            if (parent.empty() || parent == current) break;
            current = parent;
        }
    }
}

inline std::optional<std::uint64_t> positive_env_mib(const char* name) noexcept
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return std::nullopt;
    const auto parsed = parse_u64(value);
    if (!parsed || *parsed == 0) return std::nullopt;
    return sat_mul(*parsed, 1024ull * 1024ull);
}

inline std::optional<std::uint64_t> positive_env_count(const char* name) noexcept
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return std::nullopt;
    const auto parsed = parse_u64(value);
    if (!parsed || *parsed == 0) return std::nullopt;
    return parsed;
}

inline void probe_slurm(Snapshot& snapshot) noexcept
{
    std::uint64_t limit = 0;
    bool present = false;
    if (const auto per_node = positive_env_mib("SLURM_MEM_PER_NODE")) {
        min_present(*per_node, true, limit, present);
    }
    if (const auto per_cpu = positive_env_mib("SLURM_MEM_PER_CPU")) {
        const std::optional<std::uint64_t> cpus =
            positive_env_count("SLURM_CPUS_ON_NODE");
        if (cpus) min_present(sat_mul(*per_cpu, *cpus), true, limit, present);
    }
    if (const auto per_gpu = positive_env_mib("SLURM_MEM_PER_GPU")) {
        const std::optional<std::uint64_t> gpus =
            positive_env_count("SLURM_GPUS_ON_NODE");
        if (gpus) min_present(sat_mul(*per_gpu, *gpus), true, limit, present);
    }
    if (present) {
        snapshot.slurm_limit_bytes = limit;
        snapshot.has_slurm_limit = true;
    }
}

inline void parse_meminfo(std::istream& input, Snapshot& snapshot)
{
    std::uint64_t mem_free = 0;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        std::uint64_t value_kib = 0;
        if (!(fields >> key >> value_kib)) continue;
        const std::uint64_t bytes = sat_mul(value_kib, 1024ull);
        if (key == "MemTotal:") snapshot.physical_total_bytes = bytes;
        else if (key == "MemAvailable:") {
            snapshot.system_available_bytes = bytes;
            snapshot.has_system_available = true;
        } else if (key == "MemFree:") {
            mem_free = bytes;
        }
    }
    if (!snapshot.has_system_available && mem_free != 0) {
        // MemFree is deliberately conservative on older kernels; unlike an
        // improvised cache sum, it cannot promise reclaimable memory twice.
        snapshot.system_available_bytes = mem_free;
        snapshot.has_system_available = true;
    }
}

inline void parse_numa_node_meminfo(std::istream& input,
                                    NumaNodeMemory& node)
{
    std::uint64_t mem_free_bytes = 0;
    bool saw_mem_free = false;
    std::string line;
    while (std::getline(input, line)) {
        // The sysfs form is "Node N MemTotal: VALUE kB". Some kernels also
        // expose MemAvailable here; use it only when explicitly present rather
        // than inventing a reclaimability estimate from MemFree/FilePages.
        std::istringstream fields(line);
        std::string node_word;
        unsigned node_index = 0;
        std::string key;
        std::uint64_t value_kib = 0;
        if (!(fields >> node_word >> node_index >> key >> value_kib) ||
            node_word != "Node" || node_index != node.index) {
            continue;
        }
        const std::uint64_t bytes = sat_mul(value_kib, 1024ull);
        if (key == "MemTotal:") {
            node.total_bytes = bytes;
        } else if (key == "MemAvailable:") {
            node.available_bytes = bytes;
            node.has_available = true;
        } else if (key == "MemFree:") {
            mem_free_bytes = bytes;
            saw_mem_free = true;
        }
    }
    if (!node.has_available && saw_mem_free) {
        // Most kernels omit per-node MemAvailable. MemFree is less permissive
        // than estimating reclaimable page cache and therefore a safe fallback
        // for avoiding plans that fit globally but not on the selected node.
        node.available_bytes = mem_free_bytes;
        node.has_available = true;
    }
}

#if defined(__linux__)
inline std::optional<std::vector<unsigned>> read_allowed_mems_from_status(
    const char* path)
{
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        constexpr std::string_view mem_key = "Mems_allowed_list:";
        if (line.rfind(mem_key, 0) == 0) {
            return parse_index_list(
                std::string_view(line).substr(mem_key.size()));
        }
    }
    return std::nullopt;
}

inline std::optional<std::vector<unsigned>> read_calling_context_allowed_mems()
{
    // thread-self is explicit about threaded-cgroup membership. The self path
    // remains the portable procfs fallback for older Linux kernels.
    if (auto mems = read_allowed_mems_from_status(
            "/proc/thread-self/status")) {
        return mems;
    }
    return read_allowed_mems_from_status("/proc/self/status");
}

inline std::vector<NumaNodeMemory> read_numa_nodes()
{
    std::optional<std::string> online = read_first_line(
        "/sys/devices/system/node/online");
    if (!online) {
        online = read_first_line("/sys/devices/system/node/possible");
    }
    if (!online) return {};
    const auto indexes = parse_index_list(*online);
    if (!indexes) return {};

    std::vector<NumaNodeMemory> nodes;
    nodes.reserve(indexes->size());
    for (const unsigned index : *indexes) {
        const std::string root =
            "/sys/devices/system/node/node" + std::to_string(index);
        NumaNodeMemory node;
        node.index = index;
        std::ifstream meminfo(root + "/meminfo");
        if (meminfo) parse_numa_node_meminfo(meminfo, node);
        if (node.total_bytes != 0) nodes.push_back(std::move(node));
    }
    return nodes;
}

inline void probe_numa_affinity(Snapshot& snapshot)
{
    const std::vector<NumaNodeMemory> nodes = read_numa_nodes();
    if (nodes.empty()) return;
    const auto allowed_mems = read_calling_context_allowed_mems();
    if (!allowed_mems) return;

    // On an unrestricted host the global MemAvailable estimate is both more
    // accurate and less pessimistic than summing per-node MemFree. Apply the
    // NUMA-specific cap only when the cpuset excludes at least one online node.
    const bool restricted = std::any_of(
        nodes.begin(), nodes.end(), [&](const NumaNodeMemory& node) {
            return !sorted_contains(*allowed_mems, node.index);
        });
    if (!restricted) return;

    const auto selected = select_numa_affinity_capacity(
        nodes, allowed_mems);
    if (!selected) return;
    snapshot.numa_affinity_capacity_bytes = selected->total_bytes;
    snapshot.has_numa_affinity_capacity = true;
    if (selected->has_available) {
        snapshot.numa_affinity_available_bytes = selected->available_bytes;
        snapshot.has_numa_affinity_available = true;
    }
}
#endif

inline std::uint64_t read_process_rss()
{
    std::ifstream input("/proc/self/status");
    std::string line;
    std::uint64_t vm_rss = 0;
    std::uint64_t rss_anon = 0;
    std::uint64_t rss_shmem = 0;
    bool saw_accounted_rss = false;
    while (std::getline(input, line)) {
        std::size_t prefix = std::string::npos;
        std::uint64_t* destination = nullptr;
        if (line.rfind("VmRSS:", 0) == 0) {
            prefix = 6;
            destination = &vm_rss;
        } else if (line.rfind("RssAnon:", 0) == 0) {
            prefix = 8;
            destination = &rss_anon;
            saw_accounted_rss = true;
        } else if (line.rfind("RssShmem:", 0) == 0) {
            prefix = 9;
            destination = &rss_shmem;
            saw_accounted_rss = true;
        }
        if (destination == nullptr) continue;
        std::istringstream values(line.substr(prefix));
        std::uint64_t kib = 0;
        values >> kib;
        *destination = sat_mul(kib, 1024ull);
    }
    // File-backed code and shared libraries are reclaimable and absent from the
    // planner's fixed-payload model. Credit only anonymous/shared resident data
    // when the kernel reports it; fall back to VmRSS on older kernels.
    return saw_accounted_rss ? sat_add(rss_anon, rss_shmem) : vm_rss;
}

} // namespace detail

Snapshot detect_snapshot() noexcept
{
    Snapshot snapshot;
    try {
#if defined(__APPLE__)
        std::uint64_t mem = 0;
        std::size_t len = sizeof(mem);
        if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
            snapshot.physical_total_bytes = mem;
        }
#elif defined(__linux__)
        {
            std::ifstream meminfo("/proc/meminfo");
            if (meminfo) detail::parse_meminfo(meminfo, snapshot);
        }
        if (snapshot.physical_total_bytes == 0 ||
            !snapshot.has_system_available) {
            struct sysinfo info;
            if (sysinfo(&info) == 0) {
                const std::uint64_t unit = static_cast<std::uint64_t>(info.mem_unit);
                if (snapshot.physical_total_bytes == 0) {
                    snapshot.physical_total_bytes = sat_mul(
                        static_cast<std::uint64_t>(info.totalram), unit);
                }
                if (!snapshot.has_system_available) {
                    snapshot.system_available_bytes = sat_mul(
                        static_cast<std::uint64_t>(info.freeram), unit);
                    snapshot.has_system_available = true;
                }
            }
        }
        snapshot.process_rss_bytes = detail::read_process_rss();
        detail::probe_numa_affinity(snapshot);
        detail::probe_cgroups(snapshot);
        detail::probe_slurm(snapshot);
#endif
    } catch (...) {
        // A memory probe is advisory. Keep whatever was detected and let the
        // conservative fallback below handle inaccessible procfs/cgroup files.
    }
    return snapshot;
}

std::uint64_t hard_capacity_bytes(const Snapshot& snapshot) noexcept
{
    std::uint64_t capacity = 0;
    bool present = false;
    min_present(snapshot.physical_total_bytes,
                snapshot.physical_total_bytes != 0, capacity, present);
    min_present(snapshot.cgroup_limit_bytes,
                snapshot.has_cgroup_limit, capacity, present);
    min_present(snapshot.slurm_limit_bytes,
                snapshot.has_slurm_limit, capacity, present);
    min_present(snapshot.numa_affinity_capacity_bytes,
                snapshot.has_numa_affinity_capacity, capacity, present);
    return present ? capacity : 0;
}

std::uint64_t additional_headroom_bytes(const Snapshot& snapshot) noexcept
{
    std::uint64_t headroom = 0;
    bool present = false;
    min_present(snapshot.system_available_bytes,
                snapshot.has_system_available, headroom, present);
    min_present(snapshot.cgroup_headroom_bytes,
                snapshot.has_cgroup_headroom, headroom, present);
    min_present(snapshot.numa_affinity_available_bytes,
                snapshot.has_numa_affinity_available, headroom, present);
    if (snapshot.has_slurm_limit) {
        const std::uint64_t slurm_headroom =
            snapshot.process_rss_bytes < snapshot.slurm_limit_bytes
                ? snapshot.slurm_limit_bytes - snapshot.process_rss_bytes
                : 0;
        min_present(slurm_headroom, true, headroom, present);
    }
    return present ? headroom : 0;
}

std::uint64_t reachable_process_peak_bytes(const Snapshot& snapshot) noexcept
{
    const bool has_headroom = snapshot.has_system_available ||
                              snapshot.has_cgroup_headroom ||
                              snapshot.has_slurm_limit ||
                              snapshot.has_numa_affinity_available;
    const std::uint64_t capacity = hard_capacity_bytes(snapshot);
    if (!has_headroom) return capacity;
    const std::uint64_t reachable = sat_add(
        snapshot.process_rss_bytes, additional_headroom_bytes(snapshot));
    const bool capacity_known = snapshot.physical_total_bytes != 0 ||
                                snapshot.has_cgroup_limit ||
                                snapshot.has_slurm_limit ||
                                snapshot.has_numa_affinity_capacity;
    return capacity_known ? std::min(capacity, reachable) : reachable;
}

std::uint64_t automatic_budget_bytes(const Snapshot& snapshot,
                                            double ram_fraction) noexcept
{
    if (!(ram_fraction > 0.0)) return 0;
    ram_fraction = std::min(1.0, ram_fraction);
    const std::uint64_t capacity = hard_capacity_bytes(snapshot);
    const std::uint64_t reachable = reachable_process_peak_bytes(snapshot);

    std::uint64_t fractional_capacity = 0;
    if (capacity != 0) {
        const long double scaled = static_cast<long double>(capacity) *
                                   static_cast<long double>(ram_fraction);
        fractional_capacity = scaled >= static_cast<long double>(
            std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : static_cast<std::uint64_t>(scaled);
    }
    const bool capacity_known = has_hard_capacity(snapshot);
    const bool reachable_known = has_headroom_constraint(snapshot);
    if (capacity_known && reachable_known) {
        return std::min(fractional_capacity, reachable);
    }
    if (capacity_known) return fractional_capacity;
    if (reachable_known) {
        return static_cast<std::uint64_t>(
            static_cast<long double>(reachable) * ram_fraction);
    }
    return 0;
}

} // namespace b4support::host_memory
