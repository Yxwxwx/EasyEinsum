#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// 替换为你 GPU 版本的头文件
#include "einsum/matx_einsum.hpp"
#include "tensor/cuda_tensor.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace EasyEinsum;

template <typename T, int Rank>
void bind_cuda_tensor(py::module_& m, const std::string& typestr) {
  std::string class_name =
      "CudaTensor_" + typestr + "_R" + std::to_string(Rank);

  py::class_<CudaTensor<T, Rank>, std::shared_ptr<CudaTensor<T, Rank>>>(
      m, class_name.c_str(), py::buffer_protocol())
      // 构造函数
      .def(py::init([](std::vector<size_t> shape_vec) {
        if (shape_vec.size() != Rank) {
          throw std::invalid_argument("Shape size must match Tensor Rank " +
                                      std::to_string(Rank));
        }
        std::array<size_t, Rank> shape_array;
        for (int i = 0; i < Rank; ++i) {
          shape_array[i] = shape_vec[i];
        }
        return std::make_shared<CudaTensor<T, Rank>>(shape_array);
      }))
      // 静态方法
      .def_static("random", &CudaTensor<T, Rank>::random,
                  "shape"_a,          // shape
                  "min_val"_a = 0.0,  // min_val
                  "max_val"_a = 1.0   // max_val
                  )
      .def_static("zeros", &CudaTensor<T, Rank>::zeros)
      // 成员方法
      .def("print", &CudaTensor<T, Rank>::print)
      .def_property_readonly("shape", &CudaTensor<T, Rank>::shape)

      .def_property_readonly("__cuda_array_interface__", [](py::object self) {
        auto& t = self.cast<CudaTensor<T, Rank>&>();
        py::dict d;

        // 1. 形状
        py::list shape;
        for (int i = 0; i < Rank; ++i) shape.append(t.shape()[i]);
        d["shape"] = py::tuple(shape);

        // 2. 类型字符串 (例如 "<f4" 代表 float32)
        d["typestr"] = py::format_descriptor<T>::format();

        // 3. 数据指针：(地址, 是否只读)
        d["data"] = py::make_tuple(reinterpret_cast<size_t>(t.data()), false);

        // 4. 版本号 (目前固定为 3)
        d["version"] = 3;

        // 5. 步长 (Strides) - 以字节为单位
        py::list strides;
        for (int i = 0; i < Rank; ++i) {
          strides.append(t.strides()[i] * sizeof(T));
        }
        d["strides"] = py::tuple(strides);

        return d;
      });
}

// 辅助模板：展开 Result Rank (RC)
template <int RA, int RB, int RC>
struct EinsumBinderCUDA {
  static void bind(py::module_& m) {
    std::string func_name = "einsum_f32_r" + std::to_string(RA) +
                            std::to_string(RB) + std::to_string(RC);

    m.def(
        func_name.c_str(),
        [](const std::string& expr,
           const std::shared_ptr<CudaTensor<float, RA>>& A,
           const std::shared_ptr<CudaTensor<float, RB>>& B) {
          return matx_einsum<float, RA, RB, RC>(expr, A, B);
        },
        "expr"_a, "A"_a, "B"_a);

    if constexpr (RC < 4) {
      EinsumBinderCUDA<RA, RB, RC + 1>::bind(m);
    }
  }
};

// 辅助模板：展开 Input Rank (RB)
template <int RA, int RB>
struct RBBinderCUDA {
  static void bind(py::module_& m) {
    EinsumBinderCUDA<RA, RB, 1>::bind(m);

    if constexpr (RB < 4) {
      RBBinderCUDA<RA, RB + 1>::bind(m);
    }
  }
};

// 辅助模板：展开 Input Rank (RA)
template <int RA>
struct RABinderCUDA {
  static void bind(py::module_& m) {
    RBBinderCUDA<RA, 1>::bind(m);

    if constexpr (RA < 4) {
      RABinderCUDA<RA + 1>::bind(m);
    }
  }
};

PYBIND11_MODULE(easyeinsum_cuda, m) {
  m.doc() = "EasyEinsum CUDA backend powered by cuTENSOR/cuBLAS";

  bind_cuda_tensor<float, 1>(m, "f32");
  bind_cuda_tensor<float, 2>(m, "f32");
  bind_cuda_tensor<float, 3>(m, "f32");
  bind_cuda_tensor<float, 4>(m, "f32");

  RABinderCUDA<1>::bind(m);
}