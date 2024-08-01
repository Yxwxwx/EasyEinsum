#pragma once
// Eigen head files
#define EIGEN_USE_THREADS
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>
// fmt library
#include "fmt/base.h"
#include <fmt/core.h>
#include <fmt/format.h>
// C++ std
#include <string>
#include <vector>

// some useful function
template <typename TensorType> void print_tensor(const TensorType &tensor);

template <int num_contractions, typename TensorType, int Dim1, int Dim2,
          int ResultDim>
Eigen::Tensor<TensorType, ResultDim>
einsum(const std::string &einsum_str,
       const Eigen::Tensor<TensorType, Dim1> &input1,
       const Eigen::Tensor<TensorType, Dim2> &input2);
template <typename T>
bool tensor_equal(const T &tensor1, const T &tensor2, const double tol = 1e-10);
/**
 * 判断两个张量（tensor）是否相等
 *
 * @param tensor1 第一个张量
 * @param tensor2 第二个张量
 * @param tol 容差，用于判断维度大小的差异。默认为 1e-10
 * @return 如果张量的维度相同，并且每个维度的大小差异都在容差范围内，则返回
 * true，否则返回 false
 */
template <typename TensorType>
bool tensor_equal(const TensorType &tensor1, const TensorType &tensor2,
                  const double tol) {
  // Check if the dimensions are the same
  if (tensor1.dimensions() != tensor2.dimensions()) {
    return false;
  }
  // Compare each element with the given tolerance
  for (auto i = 0; i < tensor1.size(); ++i) {
    if (std::abs(tensor1.data()[i] - tensor2.data()[i]) > tol) {
      return false;
    }
  }
  return true;
}
/**
 * 打印格式化
 *
 * 这个模板函数 print_formatted
 * 用于打印不同类型的值，包括整型和浮点型。对于整型，它将打印一个以零填充的数字。
 * 对于浮点型，它将根据数字是否为零来采取不同的打印策略。如果数字为零，它将使用精度为
 * 8 的格式化字符串来打印一个以零填充的浮点数。
 * 如果数字不为零，它将使用精度为 8 的格式化字符串来打印浮点数，并显示 8
 * 位小数。
 *
 * @tparam T 要打印的值的类型
 * @param value 要打印的值
 */
template <typename T> void print_formatted(const T &value) {
  if constexpr (std::is_integral_v<T>) {
    fmt::print("{:12}", value); // Zero-padded integers
  } else if constexpr (std::is_floating_point_v<T>) {

    fmt::print("{:0.10f}", value);
  }
}

/**
 * 递归打印给定张量（Tensor）的元素，显示其形状结构
 *
 * 此函数用于递归地打印出给定张量的所有元素，以及它们在多维空间中的位置索引。它以一种人类可读的方式显示张量的结构，
 * 对每个元素，它会打印出其索引和值。对于多维张量，它会用方括号表示维度，并在每个维度级别上递归，直到打印完所有元素。
 *
 * @param tensor 要打印的张量对象，类型为 TensorType
 * @param shape 表示张量形状的尺寸向量，类型为 typename TensorType::Dimensions
 * @param dim 表示当前递归访问的维度索引，从
 * 0开始，表示最外层维度，逐渐向内递归
 * @param indices 一个大小为 dim
 * 的索引向量，表示当前访问的元素在每个维度上的索引位置
 * @param indent
 * 表示当前递归层级的缩进字符串，用于显示层次感，随着递归的深入，每一级都比上一级多两个空格
 * @param is_last 一个布尔值，表示当前元素是否是所在维度的最后一个元素
 */
template <typename TensorType>
void print_recursive(const TensorType &tensor,
                     const typename TensorType::Dimensions &shape, size_t dim,
                     std::vector<size_t> indices, const std::string &indent,
                     bool is_last) {
  if (dim == shape.size()) {
    Eigen::array<Eigen::Index, TensorType::NumDimensions> eigen_indices;
    for (int i = 0; i < TensorType::NumDimensions; ++i) {
      eigen_indices[i] = static_cast<Eigen::Index>(indices[i]);
    }
    print_formatted(tensor(eigen_indices));
  } else {
    fmt::print("[");
    std::string new_indent = indent + " ";
    for (size_t i = 0; i < shape[dim]; ++i) {
      indices.push_back(i);
      print_recursive(tensor, shape, dim + 1, indices, new_indent,
                      i == shape[dim] - 1);
      indices.pop_back();
      if (i < shape[dim] - 1) {
        fmt::print(" "); // Space between elements in the same dimension
      }
    }
    fmt::print("]");
    if (!is_last) {
      fmt::print("\n{}", indent);
    }
  }
}

/**
 * @brief 打印张量的函数
 *
 * @tparam TensorType 模板参数，表示张量的类型
 * @param tensor 需要打印的张量
 */
template <typename TensorType> void print_tensor(const TensorType &tensor) {
  const auto &shape = tensor.dimensions();
  fmt::print(" ");
  print_recursive(tensor, shape, 0, std::vector<size_t>(), "", true);
  fmt::print("\n");
}

std::string find_different_element(const std::string &str1,
                                   const std::string &str2,
                                   std::vector<size_t> &left,
                                   std::vector<size_t> &right) {

  std::vector<char> set_I(str1.begin(), str1.end());
  std::vector<char> set_D(str2.begin(), str2.end());

  std::vector<char> difference;
  // 找到str1中不在str2中的字符，并记录它们的索引
  for (size_t i = 0; i < str1.size(); ++i) {
    char c = str1[i];
    if (str2.find(c) == std::string::npos) {
      difference.push_back(c);
      left.push_back(i); // 记录索引到left中
    }
  }

  // 找到str2中不在str1中的字符，并记录它们的索引
  for (size_t i = 0; i < str2.size(); ++i) {
    char c = str2[i];
    if (str1.find(c) == std::string::npos) {
      difference.push_back(c);
      right.push_back(i); // 记录索引到right中
    }
  }

  // 将不同元素组合成一个字符串
  std::string different_elements(difference.begin(), difference.end());
  return different_elements;
}

std::vector<size_t> find_indices(const std::string &result_indices,
                                 const std::string &different_elements) {
  std::vector<size_t> indices;
  if (result_indices == different_elements) {
    return indices;
  }

  for (char c : different_elements) {
    auto pos = result_indices.find(c);
    if (pos != std::string::npos) {
      indices.push_back(pos);
    }
  }

  return indices;
}

template <typename TensorType>
std::vector<Eigen::IndexPair<int>>
parse_einsum_string(const std::string &einsum_str, std::string &result_indices,
                    std::vector<size_t> &shuffle_indexs,
                    std::vector<size_t> &left, std::vector<size_t> &right) {
  std::vector<Eigen::IndexPair<int>> idx_pairs;

  // Find "->" position
  auto arrow_pos = einsum_str.find("->");
  if (arrow_pos == std::string::npos) {
    throw std::invalid_argument("Invalid einsum string format: missing '->'");
  }
  // Extract left and right parts
  auto left_part = einsum_str.substr(0, arrow_pos);
  result_indices = einsum_str.substr(arrow_pos + 2);

  // Check if result_indices is empty after "->"
  if (result_indices.empty()) {
    throw std::invalid_argument(
        "Unsupport this code syntax, please set an index for output");
  }

  // Split left part by commas
  std::vector<std::string> input_parts;
  auto comma_pos = left_part.find(',');
  while (comma_pos != std::string::npos) {
    input_parts.push_back(left_part.substr(0, comma_pos));
    left_part = left_part.substr(comma_pos + 1);
    comma_pos = left_part.find(',');
  }
  input_parts.push_back(left_part);

  // Check if exactly two input tensors are present
  if (input_parts.size() != 2) {
    throw std::invalid_argument(
        "Invalid number of input tensors in einsum string");
  }

  const auto &I_indices = input_parts[0];
  const auto &D_indices = input_parts[1];

  // Generate index pairs
  for (int i = 0; i < I_indices.size(); ++i) {
    for (int j = 0; j < D_indices.size(); ++j) {
      if (I_indices[i] == D_indices[j]) {
        idx_pairs.emplace_back(i, j);
      }
    }
  }
  auto different_elements =
      find_different_element(I_indices, D_indices, left, right);
  shuffle_indexs = find_indices(result_indices, different_elements);

  return idx_pairs;
}

template <int num_contractions, typename TensorType, int Dim1, int Dim2,
          int ResultDim>
Eigen::Tensor<TensorType, ResultDim>
einsum(const std::string &einsum_str,
       const Eigen::Tensor<TensorType, Dim1> &input1,
       const Eigen::Tensor<TensorType, Dim2> &input2) {
  std::vector<size_t> left_idx;
  std::vector<size_t> right_idx;

  std::vector<size_t> shuffle_idx;
  std::string result_indices;

  auto idx_pairs = parse_einsum_string<TensorType>(
      einsum_str, result_indices, shuffle_idx, left_idx, right_idx);

  if (num_contractions != idx_pairs.size()) {
    throw std::invalid_argument("Invalid number of contractions");
  }

  Eigen::array<Eigen::IndexPair<int>, num_contractions> contract_dims;
  std::copy(idx_pairs.begin(), idx_pairs.end(), contract_dims.begin());
  Eigen::Tensor<TensorType, ResultDim> result;
  Eigen::array<Eigen::Index, ResultDim> result_dimensions;

  if (ResultDim != left_idx.size() + right_idx.size()) {
    throw std::invalid_argument("Invalid number of dimensions in result");
  }
  for (size_t i = 0; i < left_idx.size(); ++i) {
    result_dimensions[i] = input1.dimension(left_idx[i]);
  }

  for (size_t i = 0; i < right_idx.size(); ++i) {
    result_dimensions[left_idx.size() + i] = input2.dimension(right_idx[i]);
  }
  result.resize(result_dimensions);

  Eigen::ThreadPool pool(12 /* number of threads in pool */);

  Eigen::ThreadPoolDevice my_device(&pool, 12 /* number of threads to use */);
  result.device(my_device) = input1.contract(input2, contract_dims);

  if (shuffle_idx.empty()) {
    return result;
  } else {
    Eigen::array<int, ResultDim> shuffle_array;
    for (int i = 0; i < ResultDim; ++i) {
      shuffle_array[i] = shuffle_idx[i];
    }
    return result.shuffle(shuffle_array);
  }
}
