
import numpy as np

# =====================================
# CONFIGURACION
# =====================================

N = 1_000_000

# =====================================
# MATRICES VIRTUALES
# =====================================

def elemento_A(i, j):
    return (3 * i + 7 * j + 1) % 10

def elemento_B(i, j):
    return (5 * i + 2 * j + 4) % 10

# =====================================
# GENERAR BLOQUES PEQUENOS
# =====================================

def obtener_bloque_A(fila, columna, tamano):
    filas = np.arange(fila, min(fila + tamano, N))[:, None]
    columnas = np.arange(columna, min(columna + tamano, N))[None, :]

    return (3 * filas + 7 * columnas + 1) % 10

def obtener_bloque_B(fila, columna, tamano):
    filas = np.arange(fila, min(fila + tamano, N))[:, None]
    columnas = np.arange(columna, min(columna + tamano, N))[None, :]

    return (5 * filas + 2 * columnas + 4) % 10

# =====================================
# VISUALIZAR MATRICES
# =====================================

print(f"Dimension virtual: {N} x {N}")

print("\nPrimeros 5x5 elementos de A:")
print(obtener_bloque_A(0, 0, 5))

print("\nPrimeros 5x5 elementos de B:")
print(obtener_bloque_B(0, 0, 5))

print("\nElemento A[500000, 500000]:")
print(elemento_A(500000, 500000))

print("\nElemento B[500000, 500000]:")
print(elemento_B(500000, 500000))