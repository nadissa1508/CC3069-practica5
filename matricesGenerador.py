# Universidad del Valle de Guatemala
# Departamento de Computación
# CC3069 - Computación Paralela y Distribuida
# Practica guiada 5
# Autores:
#   - Maria Jose Yee (231193)
#   - Angie Vela (23764)

import numpy as np

# =====================================
# CONFIGURACION
# =====================================

# Dimension conceptual del problema. Las funciones aceptan un `n` propio para pruebas pequenas.
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

def _rango(inicio, tamano, n):
    if tamano <= 0:
        raise ValueError("tamano debe ser positivo")
    if not 0 <= inicio < n:
        raise ValueError(f"indice {inicio} fuera de la matriz de dimension {n}")
    return np.arange(inicio, min(inicio + tamano, n))

def obtener_bloque_A(fila, columna, tamano, n=N, dtype=np.float32):
    """Bloque de A de hasta tamano x tamano, recortado a la dimension n. float32 por defecto."""
    filas = _rango(fila, tamano, n)[:, None]
    columnas = _rango(columna, tamano, n)[None, :]

    return ((3 * filas + 7 * columnas + 1) % 10).astype(dtype)

def obtener_bloque_B(fila, columna, tamano, n=N, dtype=np.float32):
    """Bloque de B de hasta tamano x tamano, recortado a la dimension n. float32 por defecto."""
    filas = _rango(fila, tamano, n)[:, None]
    columnas = _rango(columna, tamano, n)[None, :]

    return ((5 * filas + 2 * columnas + 4) % 10).astype(dtype)

# =====================================
# MATERIALIZAR MATRICES COMPLETAS (solo dimensiones pequenas)
# =====================================

# float32, 4 bytes por elemento; con 4 GB de VRAM, A+B+C ya no caben a partir de N ~ 18,000.
N_MAX_MATERIALIZAR = 20_000

def generar_matrices(n):
    """Devuelve A y B completas (n x n, float32) con las mismas formulas del generador virtual."""
    if n > N_MAX_MATERIALIZAR:
        raise ValueError(f"n={n} es demasiado grande para materializar; usa obtener_bloque_A/B")
    return obtener_bloque_A(0, 0, n, n), obtener_bloque_B(0, 0, n, n)

# =====================================
# VISUALIZAR MATRICES
# =====================================

if __name__ == "__main__":
    print(f"Dimension virtual: {N} x {N}")

    print("\nPrimeros 5x5 elementos de A:")
    print(obtener_bloque_A(0, 0, 5))

    print("\nPrimeros 5x5 elementos de B:")
    print(obtener_bloque_B(0, 0, 5))

    print("\nElemento A[500000, 500000]:")
    print(elemento_A(500000, 500000))

    print("\nElemento B[500000, 500000]:")
    print(elemento_B(500000, 500000))
