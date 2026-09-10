// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The production booster builder uses a 64-point Gauss-Legendre rule split
// over [-1,0] and [0,1].  This is the only part of the former integration
// header that belongs to the live computation.

inline constexpr int GL64_POS_N = 32;

inline constexpr double GL64_pos_x[GL64_POS_N] = {
    0.0243502926634244, 0.0729931217877990, 0.1214628192961206,
    0.1696444204239928, 0.2174236437400071, 0.2646871622087674,
    0.3113228719902110, 0.3572201583376681, 0.4022701579639916,
    0.4463660172534641, 0.4894031457070530, 0.5312794640198946,
    0.5718956462026340, 0.6111553551723933, 0.6489654712546573,
    0.6852363130542333, 0.7198818501716109, 0.7528199072605319,
    0.7839723589433414, 0.8132653151227975, 0.8406292962525803,
    0.8659993981540928, 0.8893154459951141, 0.9105221370785028,
    0.9295691721319396, 0.9464113748584028, 0.9610087996520538,
    0.9733268277899110, 0.9833362538846260, 0.9910133714767443,
    0.9963401167719553, 0.9993050417357722};

inline constexpr double GL64_pos_w[GL64_POS_N] = {
    0.0486909570091397, 0.0485754674415034, 0.0483447622348030,
    0.0479993885964583, 0.0475401657148303, 0.0469681828162100,
    0.0462847965813144, 0.0454916279274181, 0.0445905581637566,
    0.0435837245293235, 0.0424735151236536, 0.0412625632426235,
    0.0399537411327203, 0.0385501531786156, 0.0370551285402400,
    0.0354722132568824, 0.0338051618371416, 0.0320579283548516,
    0.0302346570724025, 0.0283396726142595, 0.0263774697150547,
    0.0243527025687109, 0.0222701738083833, 0.0201348231535302,
    0.0179517157756973, 0.0157260304760247, 0.0134630478967186,
    0.0111681394601311, 0.0088467598263639, 0.0065044579689784,
    0.0041470332605625, 0.0017832807216964};

inline constexpr int GL64_N = 2 * GL64_POS_N;
inline constexpr int GL64_SPLIT_N = 2 * GL64_N;

struct SplitGL64Rule {
    alignas(64) double x[GL64_SPLIT_N];
    alignas(64) double w[GL64_SPLIT_N];

    SplitGL64Rule()
    {
        for (int i = 0; i < GL64_POS_N; ++i) {
            const double u = GL64_pos_x[i];
            const double weight = 0.5 * GL64_pos_w[i];
            const int offset = 4 * i;

            x[offset] = -0.5 - 0.5 * u;
            x[offset + 1] = -0.5 + 0.5 * u;
            x[offset + 2] = 0.5 - 0.5 * u;
            x[offset + 3] = 0.5 + 0.5 * u;

            w[offset] = weight;
            w[offset + 1] = weight;
            w[offset + 2] = weight;
            w[offset + 3] = weight;
        }
    }
};

[[nodiscard]] inline const SplitGL64Rule& get_split_gl64_rule()
{
    static const SplitGL64Rule rule;
    return rule;
}
