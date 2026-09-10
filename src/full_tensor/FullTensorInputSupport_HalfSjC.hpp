// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "amplitude/AmplitudeInputSupport.hpp"

namespace b4support {


// Metadata for the half-stored sjC tensor.
//
// The logical sjC block is still indexed by (x,l24,l34,l35) and viewed as
// a matrix over (k3,k4).  The payload, however, stores only a canonical
// representative under
//
//   sjC(x,k4,k3,l24,l34,l35) = sjC(x,k3,k4,l35,l34,l24).
//
// For non-canonical logical blocks, `storage_block` points to the stored
// swapped block and `transpose_from_storage=true` means the contraction must
// read that stored matrix transposed back into the logical (k3,k4) view.
struct FullTensorSjCHalfBlockMap {
    Range2D logical = Range2D{};
    std::uint32_t storage_block = 0u;
    bool transpose_from_storage = false;

    bool empty() const noexcept {
        return logical.empty();
    }
};

// ============================================================================
// Full-tensor input support plan
// ============================================================================
//
// This is the full-output analogue of AmplitudeInputSupportPlan.
//
// Single amplitude mode fixed i1..i5, so the boundary intertwiner labels were constants.
// Full tensor mode keeps i1..i5 alive, so the support blocks simply acquire
// boundary intertwiner outer axes where the corresponding tensor depends on them.
//
// Packed layouts used below:
//
//   sjA: block (i1,x,l25)          -> k5 range
//   sjB: block (x,k4,l35,l45)      -> k5 range
//   sjC: half-stored block (x,l24,l34,l35) -> k3 x k4 rectangle
//   sjD: block (x,k3,l23,l24)      -> k2 range
//   sjE: block (i1,x,l25)          -> k2 range
//
//   B2 : block (i2,l23,l24,l25)    -> k2 range
//   B3 : block (i3,k3,l34,l35)     -> l23 range
//   B4 : block (i4,k4,l24,l34)     -> l45 range
//   B5 : block (i5,l25,l35,l45)    -> k5 range
//
// The sjB/sjC/sjD supports are unchanged relative to single-amplitude mode.
// The only 6j support change is that sjA/sjE carry i1 explicitly.
// ============================================================================


int full_tensor_half_sjC_max_value_in_vector(const std::vector<int>& v);

std::vector<int> full_tensor_half_sjC_make_value_to_index_map(
    const std::vector<int>& values,
    int max_value);

struct FullTensorInputSupportPlan {
    FullTensorChannels channels;

    // The compact reflection output is a physical symmetry only when the
    // boundary spins are invariant under 2 <-> 5 and 3 <-> 4.  Matching boundary intertwiner
    // channel dimensions alone is not sufficient (asymmetric spin data can
    // accidentally have identical channel sets).
    bool reflection_15432_preserves_spins = false;

    std::vector<int> l23;
    std::vector<int> l24;
    std::vector<int> l25;
    std::vector<int> l34;
    std::vector<int> l35;
    std::vector<int> l45;

    RangeBlockSupport1D sjA;
    RangeBlockSupport1D sjB;

    // Half-stored sjC payload.  plan.sjC.blocks contains only storage blocks.
    // Use sjC_map[block] to obtain the logical block and the storage block.
    RangeBlockSupport2D sjC;
    std::vector<FullTensorSjCHalfBlockMap> sjC_map;
    std::vector<int> sjC_k3_to_k4;
    std::vector<int> sjC_k4_to_k3;

    RangeBlockSupport1D sjD;
    RangeBlockSupport1D sjE;

    RangeBlockSupport1D B2;
    RangeBlockSupport1D B3;
    RangeBlockSupport1D B4;
    RangeBlockSupport1D B5;

    std::uint64_t total_payload_bytes() const noexcept;

    std::uint64_t total_metadata_bytes() const noexcept;

    void print(std::ostream& os) const;
};

FullTensorInputSupportPlan build_full_tensor_input_support_plan(
    const B4Spins& s,
    int Dl,
    const FullTensorChannels& ch);

} // namespace b4support
