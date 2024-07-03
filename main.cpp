#include <gtest/gtest.h>
#include "einsum.hpp"
#include <chrono>
#include <omp.h>

template<typename T, std::size_t Dim>
bool ndarray_equal(const NDArray<T, Dim>& a, const NDArray<T, Dim>& b, double tol = 1e-10) {
    if (a.shape() != b.shape()) {
        return false;
    }

    auto a_tensor = a.tensor();
    auto b_tensor = b.tensor();
    for (std::size_t i = 0; i < a_tensor.size(); ++i) {
        if (std::abs(a_tensor(i) - b_tensor(i)) > tol) {
            return false;
        }
    }
    return true;
}

TEST(EinsumTest, SmallTest) {
// 创建一个 2x2x2x2 的张量 I 和 D
    std::array<std::size_t, 4> shape1 = {2, 2, 2, 2};
    std::array<std::size_t, 2> shape2 = {2, 2};
    std::array<std::size_t, 4> shape3 = {2, 2, 2, 2};
    NDArray<int, 4> I(shape1);
    NDArray<int, 2> D(shape2);
    NDArray<int, 4> E(shape3);

    // 填充张量
    #pragma omp parallel for collapse(4)
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

    #pragma omp parallel for collapse(4)
    for (size_t q = 0; q < 2; ++q) {
        for (size_t p = 0; p < 2; ++p) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t s = 0; s < 2; ++s) {
                    E({q, p, k, s}) = 0; // Initialize E element
                    for (size_t r = 0; r < 2; ++r) { // Correct upper bound for r
                        E({q, p, k, s}) += I({p, q, r, s}) * D({r, k});
                    }
                }
            }
        }
    }
    

    ASSERT_TRUE(ndarray_equal(J, E));
}
TEST(EinsumTest, LargeTest) {
    // 创建两个不同维度的大张量
    std::array<std::size_t, 4> shape1 = {20, 20, 20, 20};
    std::array<std::size_t, 3> shape2 = {20, 20, 20};
    std::array<std::size_t, 5> shape3 = {20, 20, 20, 20, 20}; // Correct shape for E

    NDArray<int, 4> I(shape1);
    NDArray<int, 3> D(shape2);
    NDArray<int, 5> E(shape3); // Correct shape for E

    // 填充张量 I 和 D
    #pragma omp parallel for collapse(4)
    for (size_t p = 0; p < 20; ++p) {
        for (size_t q = 0; q < 20; ++q) {
            for (size_t r = 0; r < 20; ++r) {
                for (size_t s = 0; s < 20; ++s) {
                    I({ p, q, r, s }) = p + q + r + s;
                }
            }
        }
    }

    #pragma omp parallel for collapse(3)
    for (size_t r = 0; r < 20; ++r) {
        for (size_t s = 0; s < 20; ++s) {
            for (size_t k = 0; k < 20; ++k) {
                D({ r, s, k }) = r + s + k;
            }
        }
    }

    // 计算 einsum
    auto start = std::chrono::steady_clock::now();
    auto J = einsum<1, int, 4, 3, 5>("abcd,def->abcef", I, D);
    auto end = std::chrono::steady_clock::now();
    auto einsum_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


    #pragma omp parallel for collapse(5)
    for (size_t a = 0; a < 20; a++) {
        for (size_t b = 0; b < 20; b++) {
            for (size_t c = 0; c < 20; c++) {
                for (size_t e = 0; e < 20; e++) {
                    for (size_t f = 0; f < 20; f++) {
                        E({a, b, c, e, f}) = 0;
                        for (size_t d = 0; d < 20; d++) {
                            E({a, b, c, e, f}) += I({a, b, c, d}) * D({d, e, f});
                        }                 
                    }
                }
            }
        }
    }
    end = std::chrono::steady_clock::now();
    auto for_loop_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    ASSERT_TRUE(ndarray_equal(J, E) && for_loop_time > einsum_time);
}
TEST(EinsumTest, LargeTensorTest) {
    // 创建两个不同维度的大张量
    std::size_t large_size = 100;
    std::array<std::size_t, 4> shape1 = {large_size, large_size, large_size, large_size};
    std::array<std::size_t, 2> shape2 = {large_size, large_size};
    std::array<std::size_t, 2> shape3 = {large_size, large_size}; 

    NDArray<double, 4> I(shape1);
    NDArray<double, 2> D(shape2);
    NDArray<double, 2> E(shape3); 
    // 填充张量 I 和 D
    #pragma omp parallel for collapse(4)
    for (size_t p = 0; p < large_size; ++p) {
        for (size_t q = 0; q < large_size; ++q) {
            for (size_t r = 0; r < large_size; ++r) {
                for (size_t s = 0; s < large_size; ++s) {
                    I({ p, q, r, s }) = p + q + r + s;
                }
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (size_t r = 0; r < 20; ++r) {
        for (size_t s = 0; s < 20; ++s) {
                D({ r, s }) = r + s;
        }
    }

    // 计算 einsum
    auto start = std::chrono::steady_clock::now();
    auto K =  einsum<2, double, 4, 2, 2>("prqs,rs->pq", I, D);
    auto end = std::chrono::steady_clock::now();
    auto einsum_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


    #pragma omp parallel for collapse(2)
    for (size_t p = 0; p < large_size; p++) {
        for (size_t q = 0; q < large_size; q++) {
            E({p, q}) = 0;
            for (size_t r = 0; r < large_size; r++) {
                for (size_t s = 0; s < large_size; s++) {
                    E({p, q}) += I({p, r, q, s}) * D({r, s});
                }  
            }
        }
    }
    end = std::chrono::steady_clock::now();
    auto for_loop_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    ASSERT_TRUE(ndarray_equal(K, E) && for_loop_time > einsum_time);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    return RUN_ALL_TESTS();
}
