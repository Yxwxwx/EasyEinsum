#pragma once

#ifndef EASY_EINSUM_CPU_TENSOR_HEADER
#define EASY_EINSUM_CPU_TENSOR_HEADER

#include <format>
#include <memory>
#include <random>
#include <string>

#include <tblis/tblis.h>

#include "base_tensor.hpp"
#include "memory.hpp"

namespace EasyEinsum {
template <SupportedType T, int Rank>
class CpuTensor : public TensorBase<CpuTensor<T, Rank>, T, Rank> {
  T* ptr_ = nullptr;
  bool is_owner_ = false;
  tblis::varray_view<T> tensor_;

  using Base = TensorBase<CpuTensor<T, Rank>, T, Rank>;

 public:
  using shape_type = std::array<size_t, Rank>;
  CpuTensor(shape_type shape)
      : Base(std::move(shape)),  // Initialize base class
        ptr_(static_cast<T*>(mem_malloc<Device::CPU>(
            this->total_elements_ * sizeof(T)))),    // Allocate memory
        is_owner_(true),                             // We own the memory
        tensor_(this->shape_, ptr_, this->strides_)  // Initialize tensor view
  {
    if (!ptr_) throw std::runtime_error("CPU memory allocation failed");
  }
  ~CpuTensor() {
    if (ptr_ && is_owner_) {
      mem_free<Device::CPU>(ptr_);
    }
    ptr_ = nullptr;
  }
  // View constructor:
  CpuTensor(shape_type shape, T* external_ptr)
      : Base(std::move(shape)),                      // Initialize base class
        ptr_(external_ptr),                          // External pointer
        is_owner_(false),                            // We do not own the memory
        tensor_(this->shape_, ptr_, this->strides_)  // Initialize tensor view
  {
    if (!ptr_) throw std::runtime_error("External pointer is null");
  }
  // --- Factory: Empty ---
  static auto empty(shape_type dims) {
    return std::make_shared<CpuTensor<T, Rank>>(std::move(dims));
  }
  // --- Factory: Zeros ---
  static auto zeros(shape_type dims) {
    auto t = empty(std::move(dims));
    if (t->ptr_) {
      std::fill_n(t->ptr_, t->total_elements_, static_cast<T>(0));
    }
    return t;
  }
  // --- Factory: Random ---
  static auto random(shape_type dims, T min_val = static_cast<T>(0),
                     T max_val = static_cast<T>(1)) {
    auto t = empty(std::move(dims));
    if (t->ptr_) {
      thread_local std::mt19937 gen(std::random_device{}());
      if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < t->total_elements_; ++i) t->ptr_[i] = dist(gen);
      } else {
        std::uniform_int_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < t->total_elements_; ++i) t->ptr_[i] = dist(gen);
      }
    }
    return t;
  }

  void print() const {
    auto fmt_container = [](const auto& container) {
      std::string s = "[";
      for (size_t i = 0; i < container.size(); ++i) {
        s += std::to_string(container[i]) +
             (i == container.size() - 1 ? "" : ", ");
      }
      s += "]";
      return s;
    };

    std::string type_name = "unknown";
    if constexpr (std::is_same_v<T, float>)
      type_name = "float";
    else if constexpr (std::is_same_v<T, double>)
      type_name = "double";
    else if constexpr (std::is_same_v<T, int32_t>)
      type_name = "int32";
    else if constexpr (std::is_same_v<T, int64_t>)
      type_name = "int64";

    std::cout << std::format("Tensor{{{}}} Rank: {}, Sizes:{}, Strides:{}\n",
                             type_name, Rank, fmt_container(this->shape_),
                             fmt_container(this->strides_));

    std::cout << tensor_ << std::endl;
  }
  T* data() { return ptr_; }
  const T* data() const { return ptr_; }

  auto& tensor() { return tensor_; }
  const auto& tensor() const { return tensor_; }

  auto varray() const {
    std::vector<tblis::len_type> lengths(this->shape_.begin(),
                                         this->shape_.end());
    std::vector<tblis::stride_type> strides(this->strides_.begin(),
                                            this->strides_.end());
    return tblis::varray_view<T>(lengths, static_cast<T*>(ptr_), strides);
  }
  auto varray_const() const {
    std::vector<tblis::len_type> lengths(this->shape_.begin(),
                                         this->shape_.end());
    std::vector<tblis::stride_type> strides(this->strides_.begin(),
                                            this->strides_.end());
    return tblis::varray_view<const T>(lengths, static_cast<const T*>(ptr_),
                                       strides);
  }
};  // CpuTensor class

}  // namespace EasyEinsum
#endif  // EASY_EINSUM_CPU_TENSOR_HEADER