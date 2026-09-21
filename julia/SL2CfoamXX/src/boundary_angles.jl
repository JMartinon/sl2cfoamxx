# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

"""
    lorentzian_boundary_angles

Immutable tuple of twenty `(theta, phi)` pairs, in radians and canonical
oriented-normal order, copied from the native Lorentzian preset. This boundary
closes for doubled spins `(10,10,10,10,4,4,4,4,4,4)` and proportional scalings.
Pass it directly to `compute_coherent_amplitude` or `check_closure`.
"""
global lorentzian_boundary_angles::NTuple{20,NTuple{2,Float64}}

"""
    euclidean_boundary_angles

Immutable tuple of twenty `(theta, phi)` pairs, in radians and canonical
oriented-normal order, copied from the native equilateral Euclidean preset.
This boundary closes for equal spins. Pass it directly to
`compute_coherent_amplitude` or `check_closure`.
"""
global euclidean_boundary_angles::NTuple{20,NTuple{2,Float64}}

"""Compatibility spelling of `lorentzian_boundary_angles`."""
global lorentzian_bounday_angles::NTuple{20,NTuple{2,Float64}}
"""Compatibility spelling of `euclidean_boundary_angles`."""
global euclidean_bounday_angles::NTuple{20,NTuple{2,Float64}}

function _boundary_angles_preset(preset::Int32)
    values = Vector{Float64}(undef, 40)
    _check_error(ccall(_sym(:sjx_boundary_angles), Ptr{Cvoid},
                      (Int32, Ptr{Float64}), preset, values))
    return ntuple(row -> (values[2row-1], values[2row]), 20)
end

function _initialize_boundary_angles!()
    # __init__ runs after the native library is loaded, including when loading
    # a precompiled or relocated package. No foreign call runs at compile time.
    global lorentzian_boundary_angles = _boundary_angles_preset(Int32(0))
    global euclidean_boundary_angles = _boundary_angles_preset(Int32(1))
    global lorentzian_bounday_angles = lorentzian_boundary_angles
    global euclidean_bounday_angles = euclidean_boundary_angles
    return nothing
end
