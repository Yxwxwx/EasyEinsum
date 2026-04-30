#pragma once
#ifndef EASY_EINSUM_CUDA_TENSOR_HPP
#define EASY_EINSUM_CUDA_TENSOR_HPP

#include <random>

#include "base_tensor.hpp"
#include "memory.hpp"
#include "type.hpp"

#if defined(_USE_CUDA_)

#include <cuda_runtime.h>
#include <matx.h>

namespace EasyEinsum {
namespace detail {

template <SupportedType T, int Rank>
inline matx::tensor_t<T, Rank> make_matx_tensor(
    T* ptr, const std::array<size_t, Rank>& shape,
    const std::array<size_t, Rank>& strides) {
  matx::index_t s[Rank];
  matx::index_t st[Rank];
  for (int i = 0; i < Rank; ++i) {
    s[i] = static_cast<matx::index_t>(shape[i]);
    st[i] = static_cast<matx::index_t>(strides[i]);
  }
  return matx::make_tensor<T, Rank>(ptr, s, st);
}

template <typename T>
inline constexpr bool is_complex_v = false;

template <typename T>
inline constexpr bool is_complex_v<std::complex<T>> = true;

}  // namespace detail

template <SupportedType T, int Rank>
class CudaTensor : public TensorBase<CudaTensor<T, Rank>, T, Rank> {
 public:
  using shape_type = std::array<size_t, Rank>;
  using Base = TensorBase<CudaTensor<T, Rank>, T, Rank>;

  CudaTensor(shape_type shape)
      : Base(shape),
        ptr_(static_cast<T*>(
            mem_malloc<Device::CUDA>(this->total_elements_ * sizeof(T)))),
        is_owner_(true),
        tensor_(detail::make_matx_tensor<T, Rank>(ptr_, this->shape_,
                                                  this->strides_)) {
    if (!ptr_) {
      throw std::runtime_error("CUDA malloc failed");
    }
  }

  CudaTensor(shape_type shape, T* external_ptr)
      : Base(shape),
        ptr_(external_ptr),
        is_owner_(false),
        tensor_(detail::make_matx_tensor<T, Rank>(ptr_, this->shape_,
                                                  this->strides_)) {
    if (!ptr_) {
      throw std::runtime_error("External CUDA pointer is null");
    }
  }

  ~CudaTensor() {
    if (ptr_ && is_owner_) {
      mem_free<Device::CUDA>(ptr_);
    }
    ptr_ = nullptr;
  }

  static std::shared_ptr<CudaTensor> empty(shape_type dims) {
    return std::make_shared<CudaTensor<T, Rank>>(std::move(dims));
  }

  static std::shared_ptr<CudaTensor> zeros(shape_type dims) {
    auto t = empty(std::move(dims));
    if (t->ptr_) {
      auto exec = matx::cudaExecutor{};
      (t->tensor_ = T(0)).run(exec);
    }
    return t;
  }

  static std::shared_ptr<CudaTensor> random(shape_type dims,
                                            T min_val = static_cast<T>(0),
                                            T max_val = static_cast<T>(1)) {
    auto t = empty(std::move(dims));
    if (!t->ptr_) return t;

    auto exec = matx::cudaExecutor{};

    if constexpr (std::is_floating_point_v<T>) {
      auto randOp = matx::random<T>(t->tensor_.Shape(), matx::NORMAL);
      (t->tensor_ = min_val + (max_val - min_val) * randOp).run(exec);
    } else if constexpr (detail::is_complex_v<T>) {
      using Real = typename T::value_type;
      std::vector<T> host(t->total_elements_);

      thread_local std::mt19937 gen(std::random_device{}());
      const Real lo = static_cast<Real>(min_val.real());
      const Real hi = static_cast<Real>(max_val.real());
      std::uniform_real_distribution<Real> dist(lo, hi);

      for (auto& x : host) {
        x = T(dist(gen), dist(gen));
      }

      cudaMemcpy(t->ptr_, host.data(), sizeof(T) * host.size(),
                 cudaMemcpyHostToDevice);
    } else {
      static_assert(std::is_same_v<T, void>,
                    "Unsupported type for CudaTensor::random()");
    }

    return t;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }

  auto& tensor() { return tensor_; }
  const auto& tensor() const { return tensor_; }

  const auto print() const {
    matx::set_print_format_type(matx::MATX_PRINT_FORMAT_PYTHON);
    matx::print(tensor_);
  }

 private:
  T* ptr_ = nullptr;
  bool is_owner_ = false;
  matx::tensor_t<T, Rank> tensor_;
};

}  // namespace EasyEinsum

#else  // !_USE_CUDA_

namespace EasyEinsum {
template <SupportedType T, int Rank>
class CudaTensor;  // CPU build only forward declare
}  // namespace EasyEinsum

#endif  // _USE_CUDA_

#endif  // EASY_EINSUM_CUDA_TENSOR_HPP