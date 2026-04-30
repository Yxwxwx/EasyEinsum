import numpy as np

# 尝试导入 CuPy（GPU 必备）
try:
    import cupy as cp
except ImportError:
    cp = None

# 尝试导入两个后端
try:
    import easyeinsum_cpu as ee_cpu
except ImportError:
    ee_cpu = None

try:
    import easyeinsum_cuda as ee_cuda
except ImportError as e:
    print(f"Debug: Failed to import CUDA backend due to: {e}")
    ee_cuda = None


def _get_output_rank(subscripts):
    """解析 einsum 字符串以确定结果的 Rank"""
    if "->" in subscripts:
        return len(subscripts.split("->")[1].strip())
    inputs = subscripts.replace(",", "").replace(" ", "")
    out_indices = [c for c in set(inputs) if inputs.count(c) == 1]
    return len(out_indices)


def einsum(subscripts, a, b, device="cpu"):
    """
    EasyEinsum Python 接口
    device='cpu': 接收并返回 NumPy 数组
    device='cuda': 接收并返回 CuPy 数组
    """
    rc = _get_output_rank(subscripts)

    if device == "cpu":
        if ee_cpu is None:
            raise RuntimeError(
                "CPU backend (easyeinsum_cpu) is not installed or failed to import."
            )

        # 1. 强制转换为连续的 NumPy 数组
        a_arr = np.ascontiguousarray(a, dtype=np.float32)
        b_arr = np.ascontiguousarray(b, dtype=np.float32)
        ra, rb = a_arr.ndim, b_arr.ndim

        # 2. 寻找后端实例
        func_name = f"einsum_f32_r{ra}{rb}{rc}"
        if not hasattr(ee_cpu, func_name):
            raise NotImplementedError(
                f"未找到 CPU 后端实例: {func_name} (RA={ra}, RB={rb}, RC={rc})"
            )
        backend_func = getattr(ee_cpu, func_name)

        # 3. 构造 C++ CpuTensor
        cls_a = getattr(ee_cpu, f"CpuTensor_f32_R{ra}")
        cls_b = getattr(ee_cpu, f"CpuTensor_f32_R{rb}")
        tensor_a = cls_a(list(a_arr.shape))
        tensor_b = cls_b(list(b_arr.shape))

        # 4. 利用 Buffer Protocol 将 NumPy 内存填入 C++ (零拷贝写入)
        np.array(tensor_a, copy=False)[:] = a_arr
        np.array(tensor_b, copy=False)[:] = b_arr

        # 5. 执行 CPU 计算
        result_tensor = backend_func(subscripts, tensor_a, tensor_b)

        # 6. 转回 NumPy 并解绑内存所有权
        return np.array(result_tensor, copy=True)

    elif device == "cuda":
        if ee_cuda is None:
            raise RuntimeError(
                "CUDA backend (easyeinsum_cuda) is not installed or failed to import."
            )
        if cp is None:
            raise RuntimeError("CuPy is required for GPU backend but failed to import.")

        # 1. 强制转换为连续的 CuPy 数组 (如果传入了 numpy 数组，cp.asarray 会自动做 H2D 拷贝)
        a_arr = cp.ascontiguousarray(cp.asarray(a), dtype=cp.float32)
        b_arr = cp.ascontiguousarray(cp.asarray(b), dtype=cp.float32)
        ra, rb = a_arr.ndim, b_arr.ndim

        # 2. 寻找后端实例
        func_name = f"einsum_f32_r{ra}{rb}{rc}"
        if not hasattr(ee_cuda, func_name):
            raise NotImplementedError(
                f"未找到 CUDA 后端实例: {func_name} (RA={ra}, RB={rb}, RC={rc})"
            )
        backend_func = getattr(ee_cuda, func_name)

        # 3. 构造 C++ CudaTensor (注意你 C++ 里绑定的前缀是 CudaTensor)
        cls_a = getattr(ee_cuda, f"CudaTensor_f32_R{ra}")
        cls_b = getattr(ee_cuda, f"CudaTensor_f32_R{rb}")
        tensor_a = cls_a(list(a_arr.shape))
        tensor_b = cls_b(list(b_arr.shape))

        # 4. 利用 __cuda_array_interface__ 实现显存级拷贝 (Device to Device)
        # cp.asarray 识别到接口后，生成一个视图，然后 copyto 直接在 GPU 内部转移数据
        cp.copyto(cp.asarray(tensor_a), a_arr)
        cp.copyto(cp.asarray(tensor_b), b_arr)

        # 5. 执行 GPU 计算
        result_tensor = backend_func(subscripts, tensor_a, tensor_b)

        # 6. 转回 CuPy 并解绑内存所有权
        return cp.array(cp.asarray(result_tensor), copy=True)

    else:
        raise ValueError(f"Unknown device: {device}. Expected 'cpu' or 'cuda'.")
