import time

import torch
import numpy as np


# Проверяем поддержку MPS
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Успех! Используется GPU (MPS).")
# else:
#     device = torch.device("cpu")
#     print("MPS недоступен, используется CPU.")

# 3.1
A = torch.rand(64, 1024, 1024)
B = torch.rand(128, 512, 512)
C = torch.rand(256, 256, 256)

# 3.2


# 3.3
def bench_matmul():
    matmul = torch.matmul(C, C)
    matmul = torch.matmul(B, B)
    matmul = torch.matmul(A, A)


def bench_mult():
    mult = A * A
    mult = B * B
    mult = C * C


def bench_add():
    sum = A + A
    sum = B + B
    sum = C + C


def bench_T():
    t = A.mT
    t = B.mT
    t = C.mT


def bench_sum():
    sum = A.sum() + B.sum() + C.sum()


def count_diff_devices(cpu, mps, f):
    torch.set_default_device(cpu)
    f()
    cpu_start = time.time()
    f()
    cpu_end = time.time()

    torch.set_default_device(mps)
    f()
    torch.mps.synchronize()
    mps_start = time.time()
    f()
    torch.mps.synchronize()
    mps_end = time.time()
    cpu_delta = (cpu_end - cpu_start) * 1000
    mps_delta = (mps_end - mps_start) * 1000
    diff = cpu_delta / mps_delta
    return (round(cpu_delta, 1), round(mps_delta, 1), round(diff, 1))


cpu = "cpu"
mps = "mps"


def compare_all():
    funcs = [bench_matmul, bench_mult, bench_add, bench_T, bench_sum]
    names = [
        "Matmul умножение",
        "Матричное умножение",
        "Матричное сложение",
        "Матричное транспанирование",
        "Сумма элементов матрицы",
    ]
    print("CPU", "   MPS", " Diff", "Name")
    for i in range(len(funcs)):
        res = count_diff_devices(cpu, mps, funcs[i])
        print(res[0], " | ", res[1], " | ", res[2], " | ", names[i])


compare_all()
