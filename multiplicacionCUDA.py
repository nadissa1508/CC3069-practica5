
import time
import cupy as cp

# =====================================
# CONFIGURACION
# =====================================

N = 100

# =====================================
# KERNEL CUDA
# =====================================

codigo_cuda = r'''
extern "C" __global__
void multiplicar(
    long long* C,
    int N
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {

        long long suma = 0;

        for (int k = 0; k < N; k++) {

            // Mismas formulas del generador virtual
            long long a = (3LL * i + 7LL * k + 1) % 10;
            long long b = (5LL * k + 2LL * j + 4) % 10;

            suma += a * b;
        }

        C[(long long)i * N + j] = suma;
    }
}
'''

kernel = cp.RawKernel(codigo_cuda, "multiplicar")

# =====================================
# MULTIPLICACION PARALELA
# =====================================

def multiplicar_cuda(n):

    # Reservar espacio para el resultado en GPU
    C_gpu = cp.empty((n, n), dtype=cp.int64)

    # Configuracion de hilos
    bloque = (16, 16)

    grid = (
        (n + bloque[0] - 1) // bloque[0],
        (n + bloque[1] - 1) // bloque[1]
    )

    # Ejecutar kernel
    kernel(grid, bloque, (C_gpu, n))

    # Esperar que termine la GPU
    cp.cuda.Stream.null.synchronize()

    return C_gpu


if __name__ == "__main__":

    print("Multiplicacion paralela CUDA")
    print(f"Dimension: {N} x {N}")

    # Compilacion y calentamiento previo
    multiplicar_cuda(2)

    inicio = time.perf_counter()

    C_gpu = multiplicar_cuda(N)

    # Copiar resultado de GPU a CPU
    C = cp.asnumpy(C_gpu)

    fin = time.perf_counter()

    print("\nResultado C (primeros 5x5):")
    print(C[:5, :5])

    print(f"\nTiempo CUDA: {fin - inicio:.6f} segundos")
