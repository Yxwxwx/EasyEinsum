#pragma once
#ifndef EASY_EINSUM_TYPE_HPP
#define EASY_EINSUM_TYPE_HPP

#include <complex>
#include <type_traits>

namespace EasyEinsum {
enum class DeviceType : int { CPU = 0, CUDA = 1 };  // Device type
enum class DType : int {
  Float32 = 0,
  Float64 = 1,
  Complex64 = 2,
  Complex128 = 3,
};  // Data type

template <typename T>
concept SupportedType =
    std::is_floating_point_v<T> || std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

template <typename T>
consteval DType dtype_of() {
  if constexpr (std::is_same_v<T, float>)
    return DType::Float32;
  else if constexpr (std::is_same_v<T, double>)
    return DType::Float64;
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    return DType::Complex64;
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    return DType::Complex128;
  else
    static_assert(sizeof(T) == 0, "Unsupported data type for EasyEinsum");
}

}  // namespace EasyEinsum
#endif  // EASY_EINSUM_TYPE_HPP
