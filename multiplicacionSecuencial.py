
import time
import numpy as np

from matricesGenerador import elemento_A, elemento_B

# =====================================
# CONFIGURACION
# =====================================

N = 100

# =====================================
# MULTIPLICACION SECUENCIAL
# =====================================

def multiplicar_secuencial(n):
    C = np.zeros((n, n), dtype=np.int64)

    for i in range(n):
        for j in range(n):

            suma = 0

            for k in range(n):
                a = elemento_A(i, k)
                b = elemento_B(k, j)

                suma += a * b

            C[i, j] = suma

    return C


if __name__ == "__main__":

    print("Multiplicacion secuencial")
    print(f"Dimension: {N} x {N}")

    inicio = time.perf_counter()

    C = multiplicar_secuencial(N)

    fin = time.perf_counter()

    print("\nResultado C (primeros 5x5):")
    print(C[:5, :5])

    print(f"\nTiempo CPU: {fin - inicio:.6f} segundos")
