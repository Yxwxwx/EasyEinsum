#pragma once
#ifndef EASY_EINSUM_TBLIS_EINSUM_HPP
#define EASY_EINSUM_TBLIS_EINSUM_HPP

#include "../tensor/cpu_tensor.hpp"
#include "help_functions.hpp"

namespace EasyEinsum {

template <typename T, int RA, int RB, int RC>
void run_tblis_einsum(const tblis::varray_view<const T>& A,
                      const std::string& idxA,
                      const tblis::varray_view<const T>& B,
                      const std::string& idxB, tblis::varray_view<T>& C,
                      const std::string& idxC) {
  tblis::mult(static_cast<T>(1.0), A, idxA.c_str(), B, idxB.c_str(),
              static_cast<T>(0.0), C, idxC.c_str());
}
template <typename T, int RA, int RB, int RC>
auto tblis_einsum(const std::string& eq,
                  const std::shared_ptr<CpuTensor<T, RA>>& A,
                  const std::shared_ptr<CpuTensor<T, RB>>& B) {
  auto [idxA, idxB, idxC] = detail::parse_einsum<RA, RB, RC>(eq);
  auto shapeA = A->shape();
  auto shapeB = B->shape();
  auto shapeC =
      detail::validate_einsum<RA, RB, RC>(idxA, idxB, idxC, shapeA, shapeB);

  auto C = CpuTensor<T, RC>::zeros(shapeC);

  auto va = A->varray_const();
  auto vb = B->varray_const();
  auto vc = C->varray();

  run_tblis_einsum<T, RA, RB, RC>(va, idxA, vb, idxB, vc, idxC);
  return C;
}
}  // namespace EasyEinsum
#endif  // EASY_EINSUM_TBLIS_EINSUM_HPP
