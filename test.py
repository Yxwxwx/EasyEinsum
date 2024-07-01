import numpy as np
import time

def large_tensor_einsum():
    # 创建 50x50x50x50 的两个张量
    I = np.fromfunction(lambda i, j, k, l: i + j + k + l, (50, 50, 50, 50), dtype=int)
    D = np.fromfunction(lambda i, j, k, l: i * j * k * l, (50, 50, 50, 50), dtype=int)

    # 计算 einsum 并测量时间
    start_time = time.time()
    J = np.einsum('ijpq,pqrs->ijrs', I, D)
    end_time = time.time()

    # 打印执行时间
    print(f"Einsum takes: {end_time - start_time:.6f} s")

    # 打印或者处理结果
    # print(J)  # 打印结果张量 J

if __name__ == "__main__":
    large_tensor_einsum()
