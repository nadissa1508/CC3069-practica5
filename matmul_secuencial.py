# Universidad del Valle de Guatemala
# Departamento de Computación
# CC3069 - Computación Paralela y Distribuida
# Practica guiada 5
# Autores:
#   - Maria Jose Yee (231193)
#   - Angie Vela (23764)
#
# Multiplicación de matrices densas C = A x B, versión SECUENCIAL en CPU (referencia para el benchmark).
#
# Algoritmo clásico O(N^3): C[i][j] = sum_k A[i][k] * B[k][j]. Un solo hilo, sin bibliotecas
# paralelas ni np.dot / np.matmul.
#
# Uso:
#    python matmul_secuencial.py --n 64 --verificar                 # 3 ciclos for, Python puro
#    python matmul_secuencial.py --n 1024 --modo numpy --verificar  # alternativa de CPU (NO es la version pura)
#
# Los datos A y B vienen de matricesGenerador.py (formulas deterministas), los mismos que usa matmul_cuda.py.

from __future__ import annotations

import argparse
import time

import numpy as np

from matricesGenerador import generar_matrices


def verificar_resultado(C: np.ndarray, muestras: int = 2000, max_completo: int = 2048, seed: int = 0) -> bool:
    """Compara C contra el producto exacto en enteros (int64). No se cronometra.

    N <= max_completo: compara toda la matriz. N mayor: compara `muestras` elementos al azar,
    cada uno como producto punto exacto fila x columna generado por bloques (sin materializar A ni B).
    """
    n = C.shape[0]
    if n <= max_completo:
        A, B = generar_matrices(n)
        ref = A.astype(np.int64) @ B.astype(np.int64)
        idx = (slice(None), slice(None))
    else:
        rng = np.random.default_rng(seed)
        fi = rng.integers(0, n, muestras)
        ci = rng.integers(0, n, muestras)
        k = np.arange(n, dtype=np.int64)
        ref = np.array([int(((3 * i + 7 * k + 1) % 10) @ ((5 * k + 2 * j + 4) % 10)) for i, j in zip(fi, ci)])
        idx = (fi, ci)
    obtenido = C[idx]
    # Los valores son enteros 0..9: la suma float32 es exacta mientras 81*N < 2^24.
    if 81 * n < 2**24:
        return bool(np.array_equal(obtenido, ref.astype(np.float32)))
    return bool(np.allclose(obtenido, ref, rtol=1e-5))


def matmul_puro(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tres ciclos for anidados (i: fila de A, j: columna de B, k: índice del producto punto)."""
    n, k = A.shape
    m = B.shape[1]
    C = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for p in range(k):
                acc += A[i, p] * B[p, j]
            C[i, j] = acc
    return C


def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Dos ciclos for; el producto punto fila x columna es (A[i,:] * B[:,j]).sum() de NumPy."""
    n, m = A.shape[0], B.shape[1]
    C = np.zeros((n, m), dtype=np.float32)
    BT = np.ascontiguousarray(B.T)  # filas contiguas en memoria
    for i in range(n):
        for j in range(m):
            C[i, j] = np.sum(A[i, :] * BT[j, :])
    return C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="matrices cuadradas N x N")
    ap.add_argument("--modo", choices=["puro", "numpy"], default="puro")
    ap.add_argument("--verificar", action="store_true", help="compara contra el producto exacto (no se cronometra)")
    args = ap.parse_args()

    A, B = generar_matrices(args.n)
    print(f"N = {args.n} | elementos por matriz: {args.n**2:,} | "
          f"A+B+C en memoria: {3 * A.nbytes / 1024**2:.2f} MB | MACs: {args.n**3:,}")

    fn = matmul_puro if args.modo == "puro" else matmul_numpy
    t0 = time.perf_counter()
    C = fn(A, B)
    t = time.perf_counter() - t0
    print(f"CPU secuencial ({args.modo}): {t:.4f} s")

    if args.verificar:
        print("Resultado correcto:", verificar_resultado(C))


if __name__ == "__main__":
    main()
