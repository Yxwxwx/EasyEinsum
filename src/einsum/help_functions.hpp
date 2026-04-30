#pragma once
#ifndef EASY_EINSUM_HELP_HPP
#define EASY_EINSUM_HELP_HPP

#include <array>
#include <stdexcept>
#include <string>

namespace EasyEinsum::detail {
template <int RA, int RB, int RC>
inline auto parse_einsum(const std::string& equation) {
  std::string idxA, idxB, idxC;
  idxA.reserve(RA);
  idxB.reserve(RB);
  idxC.reserve(RC);

  int state = 0;  // 0: A, 1: B, 2: C

  for (size_t i = 0; i < equation.size(); ++i) {
    unsigned char c = equation[i];

    if (std::isspace(c)) continue;

    if (c == ',') {
      if (state != 0) throw std::invalid_argument("Unexpected ','");
      state = 1;
      continue;
    }

    if (c == '-' && i + 1 < equation.size() && equation[i + 1] == '>') {
      if (state != 1) throw std::invalid_argument("Unexpected '->'");
      state = 2;
      ++i;
      continue;
    }

    if (!std::isalpha(c)) {
      throw std::invalid_argument("Invalid index character");
    }

    if (state == 0) {
      if (idxA.size() >= RA)
        throw std::invalid_argument("Too many indices for A");
      idxA.push_back(c);
    } else if (state == 1) {
      if (idxB.size() >= RB)
        throw std::invalid_argument("Too many indices for B");
      idxB.push_back(c);
    } else {
      if (idxC.size() >= RC)
        throw std::invalid_argument("Too many indices for C");
      idxC.push_back(c);
    }
  }

  if (idxA.size() != RA || idxB.size() != RB || idxC.size() != RC) {
    throw std::invalid_argument("Index count does not match ranks");
  }

  return std::make_tuple(idxA, idxB, idxC);
}

template <int RA, int RB, int RC>
inline auto validate_einsum(const std::string& idxA, const std::string& idxB,
                            const std::string& idxC,
                            const std::array<size_t, RA>& shapeA,
                            const std::array<size_t, RB>& shapeB) {
  std::array<size_t, 256> dim_map;
  dim_map.fill(static_cast<size_t>(-1));

  // A
  for (int i = 0; i < RA; ++i) {
    unsigned char c = idxA[i];
    if (dim_map[c] == static_cast<size_t>(-1))
      dim_map[c] = shapeA[i];
    else if (dim_map[c] != shapeA[i])
      throw std::invalid_argument("Dimension mismatch for index " +
                                  std::string(1, c));
  }

  // B
  for (int i = 0; i < RB; ++i) {
    unsigned char c = idxB[i];
    if (dim_map[c] == static_cast<size_t>(-1))
      dim_map[c] = shapeB[i];
    else if (dim_map[c] != shapeB[i])
      throw std::invalid_argument("Dimension mismatch for index " +
                                  std::string(1, c));
  }

  // C shape
  std::array<size_t, RC> shapeC;
  for (int i = 0; i < RC; ++i) {
    unsigned char c = idxC[i];
    if (dim_map[c] == static_cast<size_t>(-1))
      throw std::invalid_argument("Output index not in inputs");
    shapeC[i] = dim_map[c];
  }

  return shapeC;
}
}  // namespace EasyEinsum::detail

#endif  // EASY_EINSUM_HELP_HPP
