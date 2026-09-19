# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later


#=
All basic Julia API usage, with optional native-array contractions.

From the repository root after building SL2CFOAMXX_BUILD_JULIA=ON:
  JULIA_LOAD_PATH="$PWD/build-julia/julia:@stdlib" julia examples/julia_example.jl
  JULIA_LOAD_PATH="$PWD/build-julia/julia:@stdlib" julia examples/julia_example.jl --contract
  JULIA_LOAD_PATH="$PWD/build-julia/julia:@stdlib" julia examples/julia_example.jl --backend GPU --contract

For a longer visible progress bar, add --two-j 12 --Dl 6.
No external Julia packages are required.
=#
using SL2CfoamXX
using LinearAlgebra
const sf = SL2CfoamXX

# Euclidean equilateral boundary, canonical oriented-normal order:
# n12,n13,n14,n15,n21,n23,n24,n25,n31,n32,n34,n35,
# n41,n42,n43,n45,n51,n52,n53,n54. Columns are theta,phi in radians.
const ANGLES = [
    0.000000000000000  0.000000000000000;
    1.910633236249019  0.000000000000000;
    1.910633236249019  2.094395102393196;
    1.910633236249019 -2.094395102393196;
    3.141592653589793  0.000000000000000;
    1.230959417340775 -1.823476581936975;
    1.230959417340775  0.270918520456220;
    1.230959417340775  2.365313622849416;
    1.230959417340775  3.141592653589793;
    1.910633236249019  1.318116071652818;
    2.411864997362827 -1.435337066566786;
    0.729727656226966 -0.388139515370189;
    1.230959417340775 -1.047197551196598;
    1.910633236249019 -2.870674133133573;
    0.729727656226966  1.706255587023007;
    2.411864997362827  0.659058035826409;
    1.230959417340775  1.047197551196598;
    1.910633236249019 -0.776279030740378;
    2.411864997362827  2.753453138219605;
    0.729727656226966 -2.482534617763384]

# The coefficient vectors must use each node's tensor recoupling basis.
# All positions here are Julia one-based indices, not physical spin labels.
const NODE_SPINS = ((1,2,3,4), (5,6,7,1), (8,9,2,5), (10,3,6,8), (4,7,9,10))
const NODE_ROWS = ((1,2,3,4), (6,7,8,5), (11,12,9,10), (16,13,14,15), (17,18,19,20))

function show_contractions(tensor, cs, coherent)
    # FullTensor supports Julia's AbstractArray interface directly. Array(tensor)
    # expands compressed storage once for an efficient dense contraction.
    array = Array(tensor)
    contracted = sum(array[I] * prod(cs[n][I[n]] for n in 1:5)
                     for I in CartesianIndices(array))
    @assert isapprox(contracted, coherent; rtol=5e-9, atol=5e-21)

    # Contract T[a,b,c,d,e] with another tensor U[f,b,c,d,e], leaving (a,f).
    # transpose deliberately does not conjugate; use adjoint for conjugation.
    other = Array(copy(tensor))
    rows = reshape(array, size(array,1), :)
    other_rows = reshape(other, size(other,1), :)
    matrix = rows * transpose(other_rows)
    println("Coherent array contraction: ", contracted)
    println("Tensor/tensor contraction shape: ", size(matrix))
end

function main(args)
    backend_name, two_j, Dl, threads, contraction = "CPU", 1, 2, 2, false
    i = 1
    while i <= length(args)
        option = args[i]
        if option == "--contract"
            contraction = true
        elseif option in ("--backend", "--two-j", "--Dl", "--threads")
            i += 1
            i <= length(args) || error("$option requires a value")
            if option == "--backend"
                backend_name = args[i]
            else
                value = parse(Int,args[i])
                value >= 0 || error("$option must be nonnegative")
                option == "--two-j" ? (two_j=value) : option == "--Dl" ? (Dl=value) : (threads=value)
            end
        elseif option == "--help"
            println("Options: --backend CPU|GPU (CPU), --two-j N (1), --Dl N (2), --threads N (2), --contract")
            return
        else
            error("Unknown option $option; use --help")
        end
        i += 1
    end
    backend_name in ("CPU","GPU") || error("backend must be CPU or GPU")
    backend = backend_name == "CPU" ? sf.Backend.CPU : sf.Backend.GPU
    sf.backend_available(backend) || error("Requested backend unavailable; GPU needs a CUDA build and device")

    mktempdir() do directory
        config = sf.default_config()
        config.backend = backend
        config.cpu_threads = threads
        config.progress_bar = true
        config.verbosity = sf.Verbosity.summary
        config.tensors.directory = directory
        config.tensors.storage = sf.TensorStorage.in_memory_sym
        config.tensors.reuse_existing = false
        # Explicit convention for the reference below; default is gamma_j.
        config.simplicity_constraint = sf.SimplicityConstraint.gamma_j_p1
        config.coherent_amplitude.pruning_ratio = 0.0
        sf.configure(config)

        # Doubled spins: 1 means physical j=1/2; intertwiners are also doubled.
        spins = fill(two_j,10)
        intertwiners = zeros(Int,5)
        events = sf.ProgressEvent[]
        report = event -> push!(events,event)
        tensor = sf.compute_full_tensor(spins,Dl,1.2; backend=backend,progress=report)
        amplitude = sf.compute_amplitude(spins,intertwiners,Dl,1.2)
        indices = sf.indices_from_intertwiners(spins,intertwiners)
        @assert isapprox(tensor[indices...], amplitude; rtol=5e-10)
        closure = sf.check_closure(spins,ANGLES)
        @assert closure.all_closed
        cs = [sf.compute_coherent_coefficients(
            [spins[k] for k in NODE_SPINS[n]], ANGLES[collect(NODE_ROWS[n]),:]) for n in 1:5]
        C1, C2, C3, C4, C5 = cs
        coherent = sf.compute_coherent_amplitude(spins,Dl,1.2,ANGLES; backend=backend,progress=report)
        # Supply any finite vectors in the same intertwiner bases. Scaling C1
        # by 2im scales the amplitude by 2im: no normalization or conjugation.
        weighted = sf.compute_coherent_amplitude(spins,Dl,1.2,2im .* C1,C2,C3,C4,C5;
                                                 backend=backend,progress=report)
        @assert isapprox(weighted,2im * coherent; rtol=5e-9,atol=5e-21)
        if two_j == 1 && Dl == 2
            expected = -1.6748256353857123e-10 + 1.2973143587300214e-10im
            @assert isapprox(coherent,expected; rtol=5e-9,atol=5e-21)
        end

        # FullTensor itself supports indexing, views and broadcasting.
        first_slice = tensor[1,:,:,:,:]
        scaled = 2 .* tensor
        @assert isapprox(scaled[indices...],2amplitude;rtol=5e-10)
        array = Array(tensor)  # independent Julia-owned canonical dense copy
        owned = sf.dense_copy(tensor)
        shared = view(owned,:,:,:,:,:)
        shared[indices...] = 0.0
        @assert owned[indices...] == 0

        path = joinpath(directory,"vertex.sl2cxx")
        sf.save_tensor(tensor,path)
        loaded = sf.load_tensor(path) # independent native owned copy
        mapped = sf.map_tensor(path) # protected read-only mapping by default
        @assert Array(loaded) == array == Array(mapped)
        println("sl2cfoam++ ", sf.version(), "; backend: ", backend_name)
        println("Fixed amplitude: ", amplitude, "; coherent amplitude: ", coherent)
        println("Amplitude with custom coefficients: ", weighted)
        println("Closure: ", closure.all_closed, "; C1: ", C1)
        println("Tensor shape: ",size(tensor),"; stored values: ",tensor.stored_size)
        println("Julia element type: ",eltype(tensor),"; first slice: ",size(first_slice))
        println("Progress events: ",length(events),"; mapped read-only: ",mapped.is_read_only)
        contraction && show_contractions(tensor,cs,coherent)
    end
end

main(ARGS)
