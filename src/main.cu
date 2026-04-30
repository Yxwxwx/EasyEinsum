#include "einsum/matx_einsum.hpp"
using namespace EasyEinsum;

int main() {
  auto A_gpu = CudaTensor<float, 2>::random({2, 3});
  auto B_gpu = CudaTensor<float, 2>::random({3, 2});

  auto C_gpu = matx_einsum<float, 2, 2, 2>("ij,jk->ik", A_gpu, B_gpu);
  A_gpu->print();
  B_gpu->print();

  C_gpu->print();
}