#include <gtest/gtest.h>
#include "einsum.hpp"
#include <chrono>
void small_test() {
  // 创建一个 2x2x2x2 的张量 I 和 D
  std::array<std::size_t, 4> shape1 = {2, 2, 2, 2};
  std::array<std::size_t, 2> shape2 = {2, 2};
  NDArray<int, 4> I(shape1);
  NDArray<int, 2> D(shape2);

  // 填充张量
  for (size_t p = 0; p < 2; ++p) {
      for (size_t q = 0; q < 2; ++q) {
          for (size_t r = 0; r < 2; ++r) {
              for (size_t s = 0; s < 2; ++s) {
                  I({ p, q, r, s }) = p + q + r + s;
                  D({ r, s }) = r + s + p + q;  // Example modification for D
              }
          }
      }
  }

  // 计算 einsum
  auto J = einsum<1, int, 4, 2, 4>("pqrs,rk->qpks", I, D);  // 调整 einsum 字符串以匹配 I 和 D 的维度

  // 打印结果张量
  J.print();
}
void large_tensor_einsum() {
    // Create two 50x50x50x50 tensor objects
    std::array<std::size_t, 4> shape1 = {50, 50, 50, 50};
    std::array<std::size_t, 4> shape2 = {50, 50, 50, 50};
    NDArray<int, 4> I(shape1);
    NDArray<int, 4> D(shape2);

    // Fill tensors I and D
    for (size_t i = 0; i < 50; ++i) {
        for (size_t j = 0; j < 50; ++j) {
            for (size_t k = 0; k < 50; ++k) {
                for (size_t l = 0; l < 50; ++l) {
                    I({i, j, k, l}) = i + j + k + l;  // Fill rule for I
                    D({i, j, k, l}) = i * j * k * l;  // Fill rule for D
                }
            }
        }
    }

    // Compute einsum and measure time
    auto start = std::chrono::high_resolution_clock::now();
    auto J = einsum<2, int, 4, 4, 4>("ijpq,pqrs->ijrs", I, D);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Print execution time
    std::cout << "Einsum takes: " << duration.count() << " s" << std::endl;

    // Print or process result
    //J.print();  // Assuming there is a print function to print the contents of tensor J
}


void test_thread() {
    Eigen::Tensor<float, 3> A(50, 50, 50);
    Eigen::Tensor<float, 2> B(50,50);

    // Fill tensors with some values
    A.setRandom();
    B.setRandom();
    
    // Create the Eigen ThreadPool
    Eigen::ThreadPool pool(8 /* number of threads in pool */);
    // Create the Eigen ThreadPoolDevice.
    Eigen::ThreadPoolDevice my_device(&pool, 4 /* number of threads to use */);

    // Perform contraction
    Eigen::Tensor<float, 3> result(50, 50, 50);
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};
    result.device(my_device) = A.contract(B, contract_dims);
}
int main(int argc, char** argv) {
    large_tensor_einsum();

    
    return 0;
}
