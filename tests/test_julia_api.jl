# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
# Exercise coexistence with Julia's own BLAS before loading the native library.
using LinearAlgebra
using SL2CfoamXX
const sf = SL2CfoamXX
const BACKEND = isempty(ARGS) || ARGS[1] == "CPU" ? sf.Backend.CPU : sf.Backend.GPU
if !sf.backend_available(BACKEND)
    println("SKIP: requested Julia GPU backend is unavailable")
    exit(77)
end

const SPINS = (2, 1, 1, 2, 1, 1, 2, 1, 1, 1)
const ANGLES = [
    .31 .13; .77 -.42; 1.29 1.07; 2.11 2.21;
    2.39 2.48; 1.47 -1.14; .43 -.27; .91 .61;
    1.66 1.52; 2.52 -2.31; .58 .36; 1.04 -.83;
    1.18 .94; 1.83 -1.73; 2.67 2.69; .69 -.48;
    .82 .57; 1.96 1.89; 2.78 -2.52; 1.33 -1.06]
const NODE_SPINS = ((1,2,3,4), (5,6,7,1), (8,9,2,5), (10,3,6,8), (4,7,9,10))
const NODE_ROWS = ((1,2,3,4), (6,7,8,5), (11,12,9,10), (16,13,14,15), (17,18,19,20))
const INTERTWINERS = (3, 2, 0, 2, 2)

coefficients(spins) = [sf.compute_coherent_coefficients(
    [spins[k] for k in NODE_SPINS[n]], ANGLES[collect(NODE_ROWS[n]), :]) for n in 1:5]
contract(tensor, cs) = sum(tensor[I] * prod(cs[n][I[n]] for n in 1:5)
                           for I in CartesianIndices(tensor))
numerically_equal(a, b) = isfinite(a) && isfinite(b) &&
    abs(a - b) <= 5e-11 * max(abs(b), 1e-300)

mktempdir() do directory
    config = sf.default_config()
    config.backend = BACKEND
    config.cpu_threads = 1
    config.progress_bar = false
    config.verbosity = sf.Verbosity.silent
    config.coherent_amplitude.pruning_ratio = 0.0
    config.tensors.directory = joinpath(directory, "cache")
    config.tensors.storage = sf.TensorStorage.in_memory_dense
    config.tensors.reuse_existing = false
    sf.configure(config)

    @testset "Library, defaults, configuration snapshots and BLAS" begin
        info = sf.build_info()
        @test sf.version() == info.version
        @test !isempty(info.compiler_id)
        @test !isempty(info.blas_provider)
        @test sf.backend_available(sf.Backend.CPU)
        if BACKEND == sf.Backend.GPU
            @test sf.cuda_device_count() > 0
            @test !isempty(sf.cuda_device_info(0).name)
        end
        defaults = sf.default_config()
        @test defaults.progress_bar
        @test defaults.cpu_threads == 0
        @test defaults.memory_budget_bytes == 0
        @test defaults.tensors.storage == sf.TensorStorage.on_disk_sym
        @test defaults.tensors.reuse_existing
        @test defaults.tensors.maximum_payload_size_bytes == 107374182400
        @test defaults.simplicity_constraint == sf.SimplicityConstraint.gamma_j
        @test sf.LibraryConfig().backend == sf.Backend.CPU
        @test sf.LibraryConfig().progress_bar
        @test sf.TensorCacheOptions().storage == sf.TensorStorage.on_disk_sym
        @test sf.TensorCacheOptions().directory === nothing
        snapshot = sf.current_config()
        snapshot.low_level.booster_rho_tile = 8
        snapshot.low_level.full_tensor_cpu_p_batch = 128
        snapshot.low_level.full_tensor_cpu_final_tile = sf.FullTensorCpuFinalTile.tile_4x8
        snapshot.coherent_amplitude.pruning_ratio = .25
        snapshot.printing_style = sf.PrintingStyle.ASCII
        sf.configure(snapshot)
        snapshot.low_level.booster_rho_tile = 4
        @test sf.current_config().low_level.booster_rho_tile == 8
        @test sf.current_config().coherent_amplitude.pruning_ratio == .25
        snapshot.low_level.booster_rho_tile = 3
        @test_throws ArgumentError sf.configure(snapshot)
        @test sf.current_config().low_level.booster_rho_tile == 8
        sf.configure(config)
        matrix = reshape(collect(1.0:400), 20, 20)
        @test matrix * transpose(matrix) ≈
            [sum(matrix[i,k] * matrix[j,k] for k in 1:20) for i in 1:20, j in 1:20]
    end

    @testset "Independent coefficient and closure oracles" begin
        cs = coefficients(SPINS)
        @test cs[2] ≈ [0.1282584174511503 + 0.1343652056345688im,
                       0.2484734822500689 + 0.06033324720018829im] rtol=5e-13 atol=2e-15
        @test cs[5] ≈ [0.02331902555825567 - 0.1698302920514597im,
                       0.002828506333065372 - 0.2918289545554249im] rtol=5e-13 atol=2e-15
        @test cs[1] isa Vector{ComplexF64}
        closure = sf.check_closure(SPINS, zeros(20,2))
        @test !closure.all_closed
        for (node, labels) in zip(closure.nodes, NODE_SPINS)
            scale = sum(SPINS[k] for k in labels) / 2
            @test collect(node.vector) == [0, 0, scale]
            @test node.residual_norm == node.scale == scale
            @test node.relative_residual == 1
        end
        @test sf.check_closure(zeros(Int,10), zeros(20,2)).all_closed
        @test sf.check_closure(SPINS, zeros(20,2); absolute_tolerance=4.0).all_closed
    end

    @testset "Canonical tensor and coherent contractions ($BACKEND)" begin
        @test sf.indices_from_intertwiners(SPINS, INTERTWINERS) == (2,2,1,2,2)
        for spins in (SPINS, (2,1,3,2,1,1,2,1,1,1)),
            simplicity in (sf.SimplicityConstraint.gamma_j, sf.SimplicityConstraint.gamma_j_p1),
            cutoff in 0:2
            config.simplicity_constraint = simplicity
            sf.configure(config)
            tensor = sf.compute_full_tensor(spins, cutoff, 1.2; backend=BACKEND)
            array = Array(tensor)
            cs = coefficients(spins)
            expected = contract(array, cs)
            @test abs(expected) > 1e-24
            actual = sf.compute_coherent_amplitude(spins, cutoff, 1.2, ANGLES; backend=BACKEND)
            @test numerically_equal(actual, expected)
            if BACKEND == sf.Backend.GPU
                @test numerically_equal(sf.compute_coherent_amplitude(
                    spins, cutoff, 1.2, ANGLES; backend=sf.Backend.CPU), expected)
            end
            if spins == SPINS
                @test numerically_equal(sf.compute_amplitude(spins, INTERTWINERS, cutoff, 1.2),
                                        tensor[2,2,1,2,2])
            end
            @test tensor isa AbstractArray{Float64,5}
            @test size(tensor) == (spins == SPINS ? (2,2,2,2,2) : (2,2,2,1,2))
            @test axes(tensor,1) == 1:2
            @test collect(tensor) == array
            @test tensor[2,:,:,:,:] == array[2,:,:,:,:]
            @test tensor .* 2 == array .* 2
            @test collect(reshape(tensor, 2, :)) == reshape(array, 2, :)
            @test tensor[end] == array[end]
            @test sf.metadata(tensor).scientific.spins == spins
            @test sf.metadata(tensor).scientific.Dl == cutoff
            @test sf.metadata(tensor).scientific.gamma == 1.2
            @test sf.metadata(tensor).logical_values == length(tensor)
            # Canonical copying and compact storage must preserve every entry.
            config.tensors.storage = sf.TensorStorage.in_memory_sym
            sf.configure(config)
            compact = sf.compute_full_tensor(spins, cutoff, 1.2)
            @test Array(compact) ≈ array rtol=5e-11 atol=0
            @test length(sf.physical_view(compact)) == compact.stored_size
            if spins == SPINS
                @test compact.stored_size == 20 < length(compact)
            end
            config.tensors.storage = sf.TensorStorage.in_memory_dense
            sf.configure(config)
        end
    end

    @testset "Storage, native array mutation, descendant ownership and files" begin
        tensor = sf.FullTensor((2,3,1,1,2))
        tensor .= reshape(collect(1.0:12), size(tensor))
        expected = Array(tensor)
        independent = Array(tensor)
        independent[1] = -99
        @test tensor[1] == 1
        cloned = copy(tensor)
        cloned[1] = 77
        @test tensor[1] == 1
        deep = deepcopy(tensor)
        deep[1] = 33
        @test tensor[1] == 1
        @test isempty(sf.FullTensor())
        @test Array{Float32}(tensor) == Float32.(expected)
        # Drop every direct FullTensor reference. Both logical and physical
        # reshape/view descendants must retain the actual native allocation.
        physical, logical = let parent = copy(tensor)
            reshape(view(sf.physical_view(parent), :), 2, 6), view(reshape(parent,2,6), :, :)
        end
        GC.gc(true)
        @test collect(physical) == reshape(expected,2,6)
        physical[2,3] = 41
        @test logical[2,3] == 41
        @test_throws BoundsError tensor[0,1,1,1,1]
        @test_throws BoundsError tensor[3,1,1,1,1]
        file = joinpath(directory, "tensor.sl2cxx")
        sf.save_tensor(tensor, file)
        @test Array(sf.load_tensor(file)) == expected
        @test sf.metadata(file).logical_shape == size(tensor)
        mapped = sf.map_tensor(file)
        @test mapped.is_mapped && mapped.is_read_only
        @test Array(mapped) == expected
        @test_throws Exception setindex!(mapped, 3.0, 1)
        @test_throws Exception fill!(mapped, 3.0)
        @test_throws Exception setindex!(reshape(sf.physical_view(mapped),2,6), 3.0, 1,1)
        @test Array(mapped) == expected
        writable = sf.map_tensor(file; access=sf.TensorAccess.read_write)
        writable[2] = 13.0
        flush(writable)
        @test sf.load_tensor(file)[2] == 13
        created = sf.create_mapped(joinpath(directory,"created.sl2cxx"), size(tensor))
        fill!(created, 5)
        flush(created)
        @test all(==(5), sf.load_tensor(created.path))
        @test length(tensor.tag) == 1024
        @test !isempty(sf.format_tensor(tensor))
        config.tensors.storage = sf.TensorStorage.on_disk_sym
        config.tensors.output_path = joinpath(directory,"scientific.sl2cxx")
        sf.configure(config)
        scientific = sf.compute_full_tensor(SPINS, 1, 1.2)
        @test scientific.is_mapped && scientific.is_symmetry_compressed
        dense_file = joinpath(directory,"dense.sl2cxx")
        converted = sf.convert_storage(scientific, sf.TensorStorage.on_disk_dense, dense_file)
        @test Array(converted) == Array(scientific)
        @test Array(sf.dense_copy(scientific)) == Array(scientific)
        @test sf.metadata(converted).scientific.gamma == 1.2
        @test !converted.is_symmetry_compressed
        config.tensors.reuse_existing = true
        sf.configure(config)
        cache_events = sf.ProgressEvent[]
        reused = sf.compute_full_tensor(SPINS,1,1.2; backend=BACKEND,
                                        progress=e->push!(cache_events,e))
        @test Array(reused) == Array(scientific)
        @test isempty(cache_events)
        @test numerically_equal(sf.compute_amplitude(SPINS,INTERTWINERS,1,1.2;
            cache_policy=sf.AmplitudeCachePolicy.prefer_full_tensor), reused[2,2,1,2,2])
        config.tensors.output_path = nothing
        config.tensors.storage = sf.TensorStorage.in_memory_dense
        config.tensors.reuse_existing = false
        sf.configure(config)
    end

    @testset "Input validation and native per-call errors" begin
        for invalid in ((-1,SPINS[2:end]...), Float64.(SPINS), fill(true,10), SPINS[1:9])
            @test_throws ArgumentError sf.compute_full_tensor(invalid,0,1.2)
        end
        @test_throws OverflowError sf.compute_full_tensor(fill(big(2)^40,10),0,1.2)
        for gamma in (NaN, Inf, -Inf, -1.2, -floatmin(Float64))
            @test_throws ArgumentError sf.compute_full_tensor(SPINS,0,gamma)
        end
        @test_throws ArgumentError sf.compute_amplitude(SPINS,INTERTWINERS,0,-1.2)
        @test_throws ArgumentError sf.compute_coherent_amplitude(SPINS,0,-1.2,ANGLES)
        @test_throws ArgumentError sf.compute_full_tensor(SPINS,-1,1.2)
        @test_throws ArgumentError sf.compute_coherent_amplitude(SPINS,0,1.2,zeros(2,20))
        badangles = copy(ANGLES); badangles[1] = NaN
        @test_throws ArgumentError sf.compute_coherent_amplitude(SPINS,0,1.2,badangles)
        # Non-contiguous Julia matrices are packed in native row order.
        padded = zeros(20,4); padded[:,1:2:4] .= ANGLES
        @test sf.compute_coherent_amplitude(SPINS,0,1.2,view(padded,:,1:2:4)) ==
              sf.compute_coherent_amplitude(SPINS,0,1.2,ANGLES)
        invalidfile = joinpath(directory,"invalid.sl2cxx")
        write(invalidfile,"not a tensor")
        caught = try sf.load_tensor(invalidfile); nothing catch error; error end
        @test caught isa sf.Error
        @test caught.code == sf.ErrorCode.invalid_tensor_file
        if !sf.backend_available(sf.Backend.GPU)
            caught = try sf.compute_full_tensor(SPINS,0,1.2; backend=sf.Backend.GPU)
                nothing catch error; error end
            @test caught isa sf.Error
            @test caught.code == sf.ErrorCode.backend_unavailable
        end
        good = @async sf.compute_amplitude(SPINS,INTERTWINERS,1,1.2)
        bad = @async try sf.load_tensor(invalidfile); nothing catch error; error end
        @test isfinite(fetch(good))
        @test fetch(bad) isa sf.Error
    end

    @testset "Progress on the Julia caller task, exceptions and native display" begin
        events = sf.ProgressEvent[]
        caller = current_task()
        function report(event)
            @test current_task() === caller
            @test event.backend == BACKEND
            push!(events,event)
        end
        tensor = sf.compute_full_tensor(SPINS,1,1.2; backend=BACKEND, progress=report)
        @test !isempty(events)
        @test first(events).status == sf.ProgressStatus.started
        @test last(events).status == sf.ProgressStatus.completed
        @test last(events).completed == last(events).total
        @test all(e.operation == sf.ProgressOperation.full_tensor for e in events)
        empty!(events)
        sf.compute_coherent_amplitude(SPINS,1,1.2,ANGLES; backend=BACKEND, progress=report)
        @test first(events).status == sf.ProgressStatus.started
        @test last(events).status == sf.ProgressStatus.completed
        @test all(e.operation == sf.ProgressOperation.coherent_amplitude for e in events)
        callback_failure = ErrorException("Julia callback failure")
        caught = try sf.compute_full_tensor(SPINS,1,1.2; backend=BACKEND, progress=e->throw(callback_failure))
            nothing catch error; error end
        @test caught === callback_failure
        @test Array(sf.compute_full_tensor(SPINS,1,1.2)) == Array(tensor)
        @test_throws ArgumentError sf.compute_full_tensor(SPINS,1,1.2; progress=report)
        # Changing the process configuration from a callback cannot alter the
        # configuration snapshot of the native computation emitting the event.
        changed = copy(config)
        changed.simplicity_constraint = config.simplicity_constraint == sf.SimplicityConstraint.gamma_j ?
            sf.SimplicityConstraint.gamma_j_p1 : sf.SimplicityConstraint.gamma_j
        captured = sf.compute_full_tensor(SPINS,1,1.2; backend=BACKEND,progress=e->sf.configure(changed))
        @test Array(captured) == Array(tensor)
        @test sf.current_config().simplicity_constraint == changed.simplicity_constraint
        sf.configure(config)
        # The C++ display writes to the actual stderr file descriptor.
        config.progress_bar = true
        sf.configure(config)
        progress_file = joinpath(directory,"progress.txt")
        open(progress_file,"w") do io
            redirect_stderr(io) do
                sf.compute_full_tensor(SPINS,1,1.2)
                sf.compute_coherent_amplitude(SPINS,1,1.2,ANGLES)
            end
        end
        display = read(progress_file,String)
        @test occursin("100", display)
        @test occursin("full", lowercase(display))
        @test occursin("coherent", lowercase(display))
        config.progress_bar = false
        sf.configure(config)
    end
end
println("Julia public API validated on ", BACKEND)
