#include "einsum/tblis_einsum.hpp"

using namespace EasyEinsum;
int main() {
  auto A_cpu = CpuTensor<float, 2>::random({2, 3});
  auto B_cpu = CpuTensor<float, 2>::random({3, 2});

  auto C_cpu = tblis_einsum<float, 2, 2, 2>("ij,jk->ik", A_cpu, B_cpu);
  C_cpu->print();
}