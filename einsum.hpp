#define EIGEN_USE_THREADS
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <array>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <algorithm>

// NDArray 模板类定义
template<typename T, std::size_t Dim>
class NDArray {
public:
    using TensorType = Eigen::Tensor<T, Dim>;

    NDArray(const std::array<std::size_t, Dim>& shape) : shape_(shape) {
        Eigen::array<Eigen::Index, Dim> eigen_shape;
        for (std::size_t i = 0; i < Dim; ++i) {
            eigen_shape[i] = static_cast<Eigen::Index>(shape[i]);
        }
        tensor_ = TensorType(eigen_shape);
    }

    T& operator()(const std::array<size_t, Dim>& indices) {
        Eigen::array<Eigen::Index, Dim> eigen_indices;
        for (std::size_t i = 0; i < Dim; ++i) {
            eigen_indices[i] = static_cast<Eigen::Index>(indices[i]);
        }
        return tensor_(eigen_indices);
    }

    const T& operator()(const std::array<size_t, Dim>& indices) const {
        Eigen::array<Eigen::Index, Dim> eigen_indices;
        for (std::size_t i = 0; i < Dim; ++i) {
            eigen_indices[i] = static_cast<Eigen::Index>(indices[i]);
        }
        return tensor_(eigen_indices);
    }

    const TensorType& tensor() const {
        return tensor_;
    }

    TensorType& tensor() {
        return tensor_;
    }

    const std::array<std::size_t, Dim>& shape() const {
        return shape_;
    }

    void print() const {
        print_recursive(0, std::vector<size_t>(), true);
        std::cout << "\n";
    }

private:
    void print_recursive(size_t dim, std::vector<size_t> indices, bool is_outermost) const {
        if (dim == shape_.size()) {
            Eigen::array<Eigen::Index, Dim> eigen_indices;
            for (std::size_t i = 0; i < Dim; ++i) {
                eigen_indices[i] = static_cast<Eigen::Index>(indices[i]);
            }
            std::cout << tensor_(eigen_indices);
        } else {
            std::cout << "[";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                indices.push_back(i);
                print_recursive(dim + 1, indices, false);
                indices.pop_back();
                if (i < shape_[dim] - 1) {
                    std::cout << " ";
                }
            }
            std::cout << "]";
        }
    }

    std::array<std::size_t, Dim> shape_;
    TensorType tensor_;
};

std::string find_different_element(const std::string& str1, const std::string& str2,
                                   std::vector<size_t>& left, std::vector<size_t>& right) {

    std::vector<char> set_I(str1.begin(), str1.end());
    std::vector<char> set_D(str2.begin(), str2.end());

    std::vector<char> difference;
     // 找到str1中不在str2中的字符，并记录它们的索引
    for (size_t i = 0; i < str1.size(); ++i) {
        char c = str1[i];
        if (str2.find(c) == std::string::npos) {
            difference.push_back(c);
            left.push_back(i);  // 记录索引到left中
        }
    }
    
    // 找到str2中不在str1中的字符，并记录它们的索引
    for (size_t i = 0; i < str2.size(); ++i) {
        char c = str2[i];
        if (str1.find(c) == std::string::npos) {
            difference.push_back(c);
            right.push_back(i);  // 记录索引到right中
        }
    }
    
    // 将不同元素组合成一个字符串
    std::string different_elements(difference.begin(), difference.end());
    return different_elements; 
}

std::vector<size_t> find_indices(const std::string& result_indices, const std::string& different_elements) {
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
    for (auto &i : indices) {
        std::cout << i << " ";
    }
    return indices;
}

template<typename T, std::size_t Dim1, std::size_t Dim2, std::size_t ResultDim>
std::vector<Eigen::IndexPair<int>> parse_einsum_string(const std::string& einsum_str,
                                                       const std::array<std::size_t, Dim1>& shape1,
                                                       const std::array<std::size_t, Dim2>& shape2,
                                                       std::string& result_indices,
                                                       std::vector<size_t>& shuffle_indexs,
                                                       std::vector<size_t>& left,
                                                       std::vector<size_t>& right) {
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
        throw std::invalid_argument("Unsupport this code syntax, please set an index for output");;
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
        throw std::invalid_argument("Invalid number of input tensors in einsum string");
    }

    const auto& I_indices = input_parts[0];
    const auto& D_indices = input_parts[1];

    

    
    // Generate index pairs
    for (int i = 0; i < I_indices.size(); ++i) {
        for (int j = 0; j < D_indices.size(); ++j) {
            if (I_indices[i] == D_indices[j]) {
                idx_pairs.emplace_back(i, j);
            }
        }
    }

    auto different_elements = find_different_element(I_indices, D_indices, left, right);
    shuffle_indexs = find_indices(result_indices, different_elements);   

    return idx_pairs;
}

// 实现 einsum 函数
template<std::size_t num_contractions, typename T, std::size_t Dim1, std::size_t Dim2, std::size_t ResultDim>
NDArray<T, ResultDim> einsum(const std::string& einsum_str, 
                             const NDArray<T, Dim1>& input1,
                             const NDArray<T, Dim2>& input2) {
    
    const auto& I = input1.tensor();
    const auto& D = input2.tensor();

    std::vector<size_t> left_idx;
    std::vector<size_t> right_idx;

    std::vector<size_t> shuffle_idx;
    std::array<std::size_t, ResultDim> result_shape;
    std::string result_indices;

    auto idx_pairs = parse_einsum_string<T, Dim1, Dim2, ResultDim>(einsum_str, input1.shape(), input2.shape(),result_indices, shuffle_idx, left_idx, right_idx);
    
    if (num_contractions != idx_pairs.size()) {
        throw std::invalid_argument("Invalid number of contractions");
    }

    Eigen::array<Eigen::IndexPair<int>, num_contractions> contract_dims;
    std::copy(idx_pairs.begin(), idx_pairs.end(), contract_dims.begin());
    Eigen::Tensor<T, ResultDim> result;
    Eigen::array<Eigen::Index, ResultDim> result_dimensions;

    if (ResultDim != left_idx.size() + right_idx.size()) {
        throw std::invalid_argument("Invalid number of dimensions in result");
    }
    for (size_t i = 0; i < left_idx.size(); ++i) {
        result_dimensions[i] = I.dimension(left_idx[i]);
    }

    for (size_t i = 0; i < right_idx.size(); ++i) {
        result_dimensions[left_idx.size() + i] = D.dimension(right_idx[i]);
    }
    result.resize(result_dimensions);    

    Eigen::ThreadPool pool(8 /* number of threads in pool */);

    Eigen::ThreadPoolDevice my_device(&pool, 4 /* number of threads to use */);
    result.device(my_device) = I.contract(D, contract_dims);

    for (std::size_t i = 0; i < ResultDim; ++i) {
        result_shape[i] = result.dimension(i);
    }

    NDArray<T, ResultDim> result_ndarray(result_shape);
    
    if (shuffle_idx.empty()) {
        result_ndarray.tensor() = result;   
    }
    else {
        Eigen::array<int, ResultDim> shuffle_array;
        for (std::size_t i = 0; i < ResultDim; ++i) {
            shuffle_array[i] = shuffle_idx[i];
        }
        result_ndarray.tensor() = result.shuffle(shuffle_array);
    }

    return result_ndarray;
}
