// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include <sl2cfoamxx.hpp>

#include <iostream>
#include <stdexcept>

int main()
{
    using namespace sl2cfoamxx;

    LibraryConfig config = default_config();
    config.cpu_threads = 2;
    // Relative paths are resolved from the working directory by configure().
    // Clear this field to use SL2CFOAMXX_CACHE_DIR, or ./data_sl2cfoamxx when
    // that variable is unset or empty.
    config.tensors.directory = "./test_data";
    // This shared backend selects coherent-amplitude and full-tensor execution.
    // default_config() chooses GPU in a CUDA build and CPU otherwise; an
    // explicit backend argument can still override one coherent call.
    config.backend = Backend::CPU;
    // Choose one of on_disk_sym (the default), on_disk_dense,
    // in_memory_sym, or in_memory_dense.
    config.tensors.storage = TensorStorage::on_disk_sym;
    config.tensors.output_path.clear(); // use the automatic cache path
    config.tensors.reuse_existing = true;
    config.progress_bar = true;
    config.simplicity_constraint = SimplicityConstraint::gamma_j;
    config.verbosity = Verbosity::timing;
    configure(config);

    DSpins spins;
    spins.fill(2);

    DIntertwiners intertwiners;
    intertwiners.fill(2);
    // Set these to 30 to see the amplitude call rejected while the independent
    // full-tensor call below still runs.

    constexpr std::size_t Dl = 1;
    constexpr double gamma = 1.2;

    std::cout << "Data directory: "
              << current_config().tensors.directory << '\n';

    try {
        const double amplitude =
            compute_amplitude(spins, intertwiners, Dl, gamma);
        std::cout << "Amplitude: " << amplitude << '\n';
    } catch (const std::invalid_argument& error) {
        std::cerr << "Amplitude skipped: " << error.what() << '\n';
    }

    // Keep calls in independent try blocks: invalid fixed intertwiners do not
    // prevent a full tensor from being computed from otherwise physical spins.
    try {
        FullTensor tensor = compute_full_tensor(spins, Dl, gamma);
        std::cout << "Tensor storage: " << tensor.storage() << '\n';
        std::cout << "Stored values: " << tensor.stored_size() << '\n';
        std::cout << "Logical values: " << tensor.logical_size() << '\n';
        if (tensor.is_mapped()) {
            std::cout << "Tensor file: " << tensor.path() << '\n';
        }
        const auto indices =
            indices_from_intertwiners(spins, intertwiners);
        std::cout << "Selected entry: " << tensor.at(indices) << '\n';

        // Because this example selected on_disk_sym, the explicit cache policy
        // finds the compatible tensor just computed above. A miss would simply
        // run the normal fixed-intertwiner contraction.
        const double reused_amplitude = compute_amplitude(
            spins, intertwiners, Dl, gamma,
            AmplitudeCachePolicy::prefer_full_tensor);
        std::cout << "Reused amplitude: " << reused_amplitude << '\n';

        // The mapping is writable. Uncomment to persist a deliberate change:
        // tensor.at(indices) *= 2.0;
        // tensor.flush();
    } catch (const std::invalid_argument& error) {
        std::cerr << "Full tensor skipped: " << error.what() << '\n';
    }
}
