#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "einsum/tblis_einsum.hpp"
#include "tensor/cpu_tensor.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace EasyEinsum;

template <typename T, int Rank>
void bind_cpu_tensor(py::module_& m, const std::string& typestr) {
  std::string class_name = "CpuTensor_" + typestr + "_R" + std::to_string(Rank);

  py::class_<CpuTensor<T, Rank>, std::shared_ptr<CpuTensor<T, Rank>>>(
      m, class_name.c_str(), py::buffer_protocol())
      // bind constructor
      .def(py::init([](std::vector<size_t> shape_vec) {
        if (shape_vec.size() != Rank) {
          throw std::invalid_argument("Shape size must match Tensor Rank " +
                                      std::to_string(Rank));
        }
        std::array<size_t, Rank> shape_array;
        for (int i = 0; i < Rank; ++i) {
          shape_array[i] = shape_vec[i];
        }
        return std::make_shared<CpuTensor<T, Rank>>(shape_array);
      }))
      // bind static methods
      .def_static("random", &CpuTensor<T, Rank>::random,
                  "shape"_a,          // shape
                  "min_val"_a = 0.0,  // min_val
                  "max_val"_a = 1.0   // max_val
                  )
      .def_static("zeros", &CpuTensor<T, Rank>::zeros)
      // bind member methods
      .def("print", &CpuTensor<T, Rank>::print)
      .def_property_readonly("shape", &CpuTensor<T, Rank>::shape)
      // bind Buffer Protocol
      .def_buffer([](CpuTensor<T, Rank>& t) -> py::buffer_info {
        std::vector<ssize_t> shape_ssize, strides_ssize;
        for (int i = 0; i < Rank; ++i) {
          shape_ssize.push_back(static_cast<ssize_t>(t.shape()[i]));
          strides_ssize.push_back(
              static_cast<ssize_t>(t.strides()[i] * sizeof(T)));
        }
        return py::buffer_info(t.data(), sizeof(T),
                               py::format_descriptor<T>::format(), Rank,
                               shape_ssize, strides_ssize);
      });
}

// help template: unfold Result Rank (RC)
template <int RA, int RB, int RC>
struct EinsumBinder {
  static void bind(py::module_& m) {
    std::string func_name = "einsum_f32_r" + std::to_string(RA) +
                            std::to_string(RB) + std::to_string(RC);

    m.def(
        func_name.c_str(),
        [](const std::string& expr,
           const std::shared_ptr<CpuTensor<float, RA>>& A,
           const std::shared_ptr<CpuTensor<float, RB>>& B) {
          return tblis_einsum<float, RA, RB, RC>(expr, A, B);
        },
        "expr"_a, "A"_a, "B"_a);

    if constexpr (RC < 8) {
      EinsumBinder<RA, RB, RC + 1>::bind(m);
    }
  }
};
// help template: unfold Input Rank (RB)
template <int RA, int RB>
struct RBBinder {
  static void bind(py::module_& m) {
    EinsumBinder<RA, RB, 1>::bind(m);

    if constexpr (RB < 4) {
      RBBinder<RA, RB + 1>::bind(m);
    }
  }
};
// help template: unfold Input Rank (RA)
template <int RA>
struct RABinder {
  static void bind(py::module_& m) {
    RBBinder<RA, 1>::bind(m);

    if constexpr (RA < 4) {
      RABinder<RA + 1>::bind(m);
    }
  }
};
PYBIND11_MODULE(easyeinsum_cpu, m) {
  m.doc() = "EasyEinsum CPU backend powered by TBlis";

  bind_cpu_tensor<float, 1>(m, "f32");
  bind_cpu_tensor<float, 2>(m, "f32");
  bind_cpu_tensor<float, 3>(m, "f32");
  bind_cpu_tensor<float, 4>(m, "f32");
  bind_cpu_tensor<float, 5>(m, "f32");
  bind_cpu_tensor<float, 6>(m, "f32");
  bind_cpu_tensor<float, 7>(m, "f32");
  bind_cpu_tensor<float, 8>(m, "f32");

  RABinder<1>::bind(m);
}