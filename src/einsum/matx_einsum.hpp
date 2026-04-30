#pragma once
#ifndef EASY_EINSUM_MATX_EINSUM_HPP
#define EASY_EINSUM_MATX_EINSUM_HPP

#include "../tensor/cuda_tensor.hpp"
#include "help_functions.hpp"
namespace EasyEinsum {

#if defined(_USE_CUDA_)
template <typename T, int RA, int RB, int RC>
auto matx_einsum(const std::string& eq,
                 const std::shared_ptr<CudaTensor<T, RA>>& A,
                 const std::shared_ptr<CudaTensor<T, RB>>& B) {
  auto [idxA, idxB, idxC] = detail::parse_einsum<RA, RB, RC>(eq);
  auto shapeA = A->shape();
  auto shapeB = B->shape();
  auto shapeC =
      detail::validate_einsum<RA, RB, RC>(idxA, idxB, idxC, shapeA, shapeB);

  auto C = CudaTensor<T, RC>::zeros(shapeC);

  auto exec = matx::cudaExecutor{};

  (C->tensor() = matx::cutensor::einsum(eq, A->tensor(), B->tensor()))
      .run(exec);

  return C;
}

#endif
}  // namespace EasyEinsum

#endif  // EASY_EINSUM_MATX_EINSUM_HPP
