"""Punto 1: convolución, ReLU y max pooling implementados a mano.

Nota: nn.Conv2d (como casi todos los frameworks) calcula en realidad una
correlación cruzada: el filtro NO se voltea. Aquí hacemos lo mismo.
"""
import torch


def convolucion_manual(X, K, stride=1, padding=0):
    """Convolución 2D de una imagen X (H x W) con un filtro K (kh x kw), sin sesgo."""
    if padding > 0:
        X = torch.nn.functional.pad(X, (padding, padding, padding, padding))
    H, W = X.shape
    kh, kw = K.shape
    H_out = (H - kh) // stride + 1
    W_out = (W - kw) // stride + 1
    salida = torch.zeros(H_out, W_out, dtype=X.dtype)
    for i in range(H_out):
        for j in range(W_out):
            ventana = X[i * stride:i * stride + kh, j * stride:j * stride + kw]
            salida[i, j] = (ventana * K).sum()
    return salida


def relu_manual(X):
    """ReLU elemento a elemento: max(0, x)."""
    return torch.clamp(X, min=0)


def maxpool_manual(X, tam=2, stride=2):
    """Max pooling 2D sobre una matriz X (H x W)."""
    H, W = X.shape
    H_out = (H - tam) // stride + 1
    W_out = (W - tam) // stride + 1
    salida = torch.zeros(H_out, W_out, dtype=X.dtype)
    for i in range(H_out):
        for j in range(W_out):
            ventana = X[i * stride:i * stride + tam, j * stride:j * stride + tam]
            salida[i, j] = ventana.max()
    return salida


def explicar_convolucion(X, K, stride=1):
    """Imprime, para cada posición de salida, la ventana usada y la suma de productos.

    Sirve para seguir el cálculo a mano paso a paso.
    """
    kh, kw = K.shape
    H_out = (X.shape[0] - kh) // stride + 1
    W_out = (X.shape[1] - kw) // stride + 1
    for i in range(H_out):
        for j in range(W_out):
            ventana = X[i * stride:i * stride + kh, j * stride:j * stride + kw]
            productos = ventana * K
            terminos = " + ".join(str(int(v)) for v in productos.flatten())
            print(f"salida[{i},{j}] = {terminos} = {int(productos.sum())}")


def explicar_maxpool(X, tam=2, stride=1):
    """Imprime, para cada posición de salida, los valores de la ventana y su máximo."""
    H_out = (X.shape[0] - tam) // stride + 1
    W_out = (X.shape[1] - tam) // stride + 1
    for i in range(H_out):
        for j in range(W_out):
            ventana = X[i * stride:i * stride + tam, j * stride:j * stride + tam]
            valores = ", ".join(str(int(v)) for v in ventana.flatten())
            print(f"salida[{i},{j}] = max({valores}) = {int(ventana.max())}")


def parametros_conv(c_in, c_out, k, sesgo=True):
    """Parámetros de nn.Conv2d(c_in, c_out, kernel_size=k): c_out*c_in*k*k (+ c_out)."""
    return c_out * c_in * k * k + (c_out if sesgo else 0)


def parametros_densa(n_in, n_out, sesgo=True):
    """Parámetros de una capa densa: n_in*n_out (+ n_out)."""
    return n_in * n_out + (n_out if sesgo else 0)
