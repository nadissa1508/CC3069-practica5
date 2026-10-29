# Universidad del Valle de Guatemala
# Departamento de Computación
# CC3069 - Computación Paralela y Distribuida
# Practica guiada 5
# Autores:
#   - Maria Jose Yee (231193)
#   - Angie Vela (23764)
#
# Multiplicación de matrices densas C = A x B: CPU secuencial vs GPU (CUDA con Numba).
#
# Uso:
#   python matmul_cuda.py --n 64   --cpu pure     # CPU con 3 ciclos for (Python puro)
#   python matmul_cuda.py --n 1024 --cpu numpy    # CPU: producto punto fila x columna con NumPy
#   python matmul_cuda.py --n 2048 --cpu skip     # solo GPU (N grande, la CPU tardaría demasiado)
#
# Los datos A y B vienen de matricesGenerador.py; la CPU es matmul_secuencial.py (mismos datos).
# 
# Nota: matrices de 1,000,000 x 1,000,000 ocuparían ~4 TB en float32; no caben en ninguna GPU.
# Por eso utilizamos matrices virtuales.

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from numba import cuda, float32

from matricesGenerador import generar_matrices
from matmul_secuencial import matmul_numpy, matmul_puro, verificar_resultado

TPB = 16  # hilos por dimensión de bloque -> bloques de 16x16 = 256 hilos


# ----------------------------------------------------------------------------
# Kernels CUDA
# ----------------------------------------------------------------------------
@cuda.jit
def matmul_kernel(A, B, C):
    """Ingenuo: cada hilo calcula UN elemento C[row, col] (producto punto fila x columna)."""
    col, row = cuda.grid(2)  # grid(2) devuelve (x, y)
    n, k = A.shape
    m = B.shape[1]
    if row < n and col < m:
        acc = float32(0.0)
        for p in range(k):
            acc += A[row, p] * B[p, col]
        C[row, col] = acc


@cuda.jit
def matmul_tiled_kernel(A, B, C):
    """Con memoria compartida: el bloque carga tiles TPBxTPB de A y B y los reutiliza."""
    sA = cuda.shared.array((TPB, TPB), dtype=float32)
    sB = cuda.shared.array((TPB, TPB), dtype=float32)
    col, row = cuda.grid(2)  # grid(2) devuelve (x, y)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    n, k = A.shape
    m = B.shape[1]
    acc = float32(0.0)
    for t in range((k + TPB - 1) // TPB):
        a_col = t * TPB + tx
        b_row = t * TPB + ty
        sA[ty, tx] = A[row, a_col] if (row < n and a_col < k) else float32(0.0)
        sB[ty, tx] = B[b_row, col] if (b_row < k and col < m) else float32(0.0)
        cuda.syncthreads()
        for p in range(TPB):
            acc += sA[ty, p] * sB[p, tx]
        cuda.syncthreads()
    if row < n and col < m:
        C[row, col] = acc


# ----------------------------------------------------------------------------
# Ejecución en GPU con tiempos separados
# ----------------------------------------------------------------------------
def run_gpu(kernel, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, dict]:
    n, m = A.shape[0], B.shape[1]
    threads = (TPB, TPB)
    blocks = (math.ceil(m / TPB), math.ceil(n / TPB))  # x -> columnas, y -> filas

    cuda.synchronize()
    t0 = time.perf_counter()
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.device_array((n, m), dtype=np.float32)
    cuda.synchronize()
    t_h2d = time.perf_counter() - t0

    t0 = time.perf_counter()
    kernel[blocks, threads](dA, dB, dC)
    cuda.synchronize()  # el lanzamiento es asíncrono
    t_kernel = time.perf_counter() - t0

    t0 = time.perf_counter()
    C = dC.copy_to_host()
    t_d2h = time.perf_counter() - t0

    return C, {"h2d": t_h2d, "kernel": t_kernel, "d2h": t_d2h,
               "total": t_h2d + t_kernel + t_d2h, "blocks": blocks, "threads": threads}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="matrices cuadradas N x N")
    ap.add_argument("--cpu", choices=["pure", "numpy", "skip"], default="pure")
    args = ap.parse_args()
    n = args.n

    if not cuda.is_available():
        raise SystemExit("CUDA no disponible. Revisa driver/toolkit (cuda.detect()).")

    mem = cuda.current_context().get_memory_info()  # (libre, total) en bytes
    if 3 * n * n * 4 > mem[0]:
        raise SystemExit(f"A+B+C necesitan {3*n*n*4/1024**3:.2f} GB y solo hay {mem[0]/1024**3:.2f} GB libres en la GPU.")

    # --- Datos (mismos para CPU y GPU) ---
    A, B = generar_matrices(n)
    print(f"\nN = {n} | elementos por matriz: {n*n:,} | "
          f"memoria A+B+C: {3*A.nbytes/1024**2:.2f} MB | MACs: {n**3:,}")

    # --- CPU ---
    t_cpu = None
    C_cpu = None
    if args.cpu != "skip":
        fn = matmul_puro if args.cpu == "pure" else matmul_numpy
        t0 = time.perf_counter()
        C_cpu = fn(A, B)
        t_cpu = time.perf_counter() - t0
        print(f"\nCPU secuencial ({args.cpu}): {t_cpu:.4f} s | correcto: {verificar_resultado(C_cpu)}")

    # --- Calentamiento (compila los kernels con JIT; no se mide) ---
    w = np.ones((TPB * 2, TPB * 2), dtype=np.float32)
    run_gpu(matmul_kernel, w, w)
    run_gpu(matmul_tiled_kernel, w, w)

    # --- GPU ---
    for name, kernel in [("ingenuo", matmul_kernel), ("tiled (shared mem)", matmul_tiled_kernel)]:
        C_gpu, t = run_gpu(kernel, A, B)
        ok = verificar_resultado(C_gpu)  # después de medir; no entra en los tiempos
        if C_cpu is not None:
            ok = ok and bool(np.array_equal(C_gpu, C_cpu))
        print(f"\nGPU kernel {name}  | grid={t['blocks']} bloques de {t['threads']} hilos")
        print(f"  Host -> GPU : {t['h2d']*1000:10.3f} ms")
        print(f"  Kernel      : {t['kernel']*1000:10.3f} ms")
        print(f"  GPU -> Host : {t['d2h']*1000:10.3f} ms")
        print(f"  Total CUDA  : {t['total']*1000:10.3f} ms")
        print(f"  Resultado correcto: {ok}")
        if t_cpu is not None:
            print(f"  Speedup = t_CPU / t_CUDA_total = {t_cpu:.4f} / {t['total']:.4f} = {t_cpu/t['total']:.2f}x")


if __name__ == "__main__":
    main()
