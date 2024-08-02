#include "einsum.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <omp.h>

TEST(EinsumTest, SmallTest) {
  Eigen::Tensor<int, 4> I(2, 2, 2, 2);
  Eigen::Tensor<int, 2> D(2, 2);

// 填充张量
#pragma omp parallel for collapse(4)
  for (int p = 0; p < 2; ++p) {
    for (int q = 0; q < 2; ++q) {
      for (int r = 0; r < 2; ++r) {
        for (int s = 0; s < 2; ++s) {
          I(p, q, r, s) = p + q + r + s;
          D(r, s) = r + s + p + q; // Example modification for D
        }
      }
    }
  }

  auto result = YXTensor::einsum<1, int, 4, 2, 4>("pqrs,rk->qpks", I, D);

  Eigen::Tensor<int, 4> E(2, 2, 2, 2);

#pragma omp parallel for collapse(4)
  for (int q = 0; q < 2; ++q) {
    for (int p = 0; p < 2; ++p) {
      for (int k = 0; k < 2; ++k) {
        for (int s = 0; s < 2; ++s) {
          E(q, p, k, s) = 0;            // Initialize E element
          for (int r = 0; r < 2; ++r) { // Correct upper bound for r
            E(q, p, k, s) += I(p, q, r, s) * D(r, k);
          }
        }
      }
    }
  }

  ASSERT_TRUE(YXTensor::tensor_equal(result, E));
}

TEST(EinsumTest, LargeTest) {
  Eigen::Tensor<int, 4> I(20, 20, 20, 20);
  Eigen::Tensor<int, 3> D(20, 20, 20);
  Eigen::Tensor<int, 5> E(20, 20, 20, 20, 20);

// 填充张量 I 和 D
#pragma omp parallel for collapse(4)
  for (int p = 0; p < 20; ++p) {
    for (int q = 0; q < 20; ++q) {
      for (int r = 0; r < 20; ++r) {
        for (int s = 0; s < 20; ++s) {
          I(p, q, r, s) = p + q + r + s;
        }
      }
    }
  }

#pragma omp parallel for collapse(3)
  for (int r = 0; r < 20; ++r) {
    for (int s = 0; s < 20; ++s) {
      for (int k = 0; k < 20; ++k) {
        D(r, s, k) = r + s + k;
      }
    }
  }

  // 计算 einsum
  auto start = std::chrono::steady_clock::now();
  auto result = YXTensor::einsum<1, int, 4, 3, 5>("abcd,def->abcef", I, D);
  auto end = std::chrono::steady_clock::now();
  auto einsum_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

#pragma omp parallel for collapse(5)
  for (int a = 0; a < 20; ++a) {
    for (int b = 0; b < 20; ++b) {
      for (int c = 0; c < 20; ++c) {
        for (int e = 0; e < 20; ++e) {
          for (int f = 0; f < 20; ++f) {
            E(a, b, c, e, f) = 0;
            for (int d = 0; d < 20; ++d) {
              E(a, b, c, e, f) += I(a, b, c, d) * D(d, e, f);
            }
          }
        }
      }
    }
  }

  end = std::chrono::steady_clock::now();
  auto for_loop_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  ASSERT_TRUE(YXTensor::tensor_equal(result, E) && for_loop_time > einsum_time);
}

TEST(EinsumTest, LargeTensorTest) {
  int large_size = 100;
  Eigen::Tensor<double, 4> I(large_size, large_size, large_size, large_size);
  Eigen::Tensor<double, 2> D(large_size, large_size);
  Eigen::Tensor<double, 2> E(large_size, large_size);

// 填充张量 I 和 D
#pragma omp parallel for collapse(4)
  for (int p = 0; p < large_size; ++p) {
    for (int q = 0; q < large_size; ++q) {
      for (int r = 0; r < large_size; ++r) {
        for (int s = 0; s < large_size; ++s) {
          I(p, q, r, s) = p + q + r + s;
        }
      }
    }
  }

#pragma omp parallel for collapse(2)
  for (int r = 0; r < large_size; ++r) {
    for (int s = 0; s < large_size; ++s) {
      D(r, s) = r + s;
    }
  }

  // 计算 einsum
  auto start = std::chrono::steady_clock::now();
  auto result = YXTensor::einsum<2, double, 4, 2, 2>("prqs,rs->pq", I, D);
  auto end = std::chrono::steady_clock::now();
  auto einsum_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

#pragma omp parallel for collapse(2)
  for (int p = 0; p < large_size; ++p) {
    for (int q = 0; q < large_size; ++q) {
      E(p, q) = 0;
      for (int r = 0; r < large_size; ++r) {
        for (int s = 0; s < large_size; ++s) {
          E(p, q) += I(p, r, q, s) * D(r, s);
        }
      }
    }
  }

  end = std::chrono::steady_clock::now();
  auto for_loop_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  ASSERT_TRUE(YXTensor::tensor_equal(result, E) && for_loop_time > einsum_time);
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
