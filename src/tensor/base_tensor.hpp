#pragma once
#ifndef EASY_EINSUM_BASE_TENSOR_HEADER
#define EASY_EINSUM_BASE_TENSOR_HEADER

#include <array>
#include <numeric>

#include "type.hpp"

namespace EasyEinsum {
template <typename Derived, SupportedType T, int Rank>
class TensorBase /*Only Row-Major*/ {
 protected:
  std::array<size_t, Rank> shape_;
  std::array<size_t, Rank> strides_;
  size_t total_elements_;

  void compute_strides() {
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
  }

 public:
  TensorBase(std::array<size_t, Rank> dims)
      : shape_(std::move(dims)),
        total_elements_(std::accumulate(shape_.begin(), shape_.end(), 1ULL,
                                        std::multiplies<size_t>())) {
    compute_strides();
  }
  auto shape() const { return shape_; }
  auto strides() const { return strides_; }
  auto size() const { return total_elements_; }

  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  auto print() const { return derived().print(); }

};  // TensorBase class

}  // namespace EasyEinsum

#endif  // EASY_EINSUM_BASE_TENSOR_HEADER
