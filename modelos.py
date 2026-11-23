"""Modelos de la tarea. Cada red es una clase que extiende nn.Module."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Red totalmente conectada para imágenes 28x28 (escala de grises).

    Entrada (B,1,28,28) -> Flatten -> 784 -> 256 -> 128 -> 10 (logits).
    No se aplica softmax a la salida: nn.CrossEntropyLoss ya lo incluye.
    """

    def __init__(self, in_features=28 * 28, ocultas=(256, 128), n_clases=10):
        super().__init__()
        capas = [nn.Flatten()]
        previo = in_features
        for h in ocultas:
            capas += [nn.Linear(previo, h), nn.ReLU()]
            previo = h
        capas.append(nn.Linear(previo, n_clases))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)


class CNN(nn.Module):
    """Red convolucional para imágenes 1x28x28 (FashionMNIST).

    Consta de al menos 2 bloques [Conv2d -> ReLU -> MaxPool2d(2, 2)],
    seguidos de Flatten y capas densas (con ReLU y Dropout opcional).
    No se aplica softmax al final porque nn.CrossEntropyLoss recibe logits.
    """

    def __init__(self, in_channels=1, lado_imagen=28, canales=(16, 32),
                 kernel_size=3, ocultas=(128,), dropout=0.0, n_clases=10):
        super().__init__()
        padding = kernel_size // 2  # mantiene alto y ancho en la convolución
        capas = []
        c_prev = in_channels
        lado = lado_imagen
        for c_out in canales:
            capas += [
                nn.Conv2d(c_prev, c_out, kernel_size=kernel_size, stride=1, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            c_prev = c_out
            lado = lado // 2  # cada MaxPool2d(2, 2) divide el tamaño espacial entre 2

        capas.append(nn.Flatten())
        previo = c_prev * lado * lado
        for h in ocultas:
            capas += [nn.Linear(previo, h), nn.ReLU()]
            if dropout > 0.0:
                capas.append(nn.Dropout(p=dropout))
            previo = h
        capas.append(nn.Linear(previo, n_clases))
        self.red = nn.Sequential(*capas)

    @property
    def primera_conv(self):
        """Devuelve la primera capa nn.Conv2d de la red."""
        for capa in self.red:
            if isinstance(capa, nn.Conv2d):
                return capa
        raise RuntimeError("No se encontró ninguna capa Conv2d.")

    def forward(self, x):
        return self.red(x)


@torch.no_grad()
def resumen_capas(modelo, forma_entrada=(1, 1, 28, 28), dispositivo=None):
    """Ejecuta un forward de prueba capa por capa y verifica formas y parámetros con p.numel()."""
    if dispositivo is None:
        dispositivo = next(modelo.parameters()).device
    x = torch.zeros(forma_entrada, device=dispositivo)
    filas = []
    print(f"{'#':<4}{'Capa':<32}{'Forma de salida':<22}{'Parámetros (p.numel())':>22}")
    print("-" * 80)
    print(f"{'0':<4}{'Entrada':<32}{str(tuple(x.shape)):<22}{0:>22,}")
    total = 0
    for idx, capa in enumerate(modelo.red, start=1):
        x = capa(x)
        n_params = sum(p.numel() for p in capa.parameters())
        total += n_params
        nombre = f"{capa.__class__.__name__}"
        if isinstance(capa, nn.Conv2d):
            nombre += f"({capa.in_channels}, {capa.out_channels}, k={capa.kernel_size[0]}, p={capa.padding[0]})"
        elif isinstance(capa, nn.MaxPool2d):
            nombre += f"(k={capa.kernel_size}, s={capa.stride})"
        elif isinstance(capa, nn.Linear):
            nombre += f"({capa.in_features} -> {capa.out_features})"
        elif isinstance(capa, nn.Dropout):
            nombre += f"(p={capa.p})"
        filas.append((idx, nombre, tuple(x.shape), n_params))
        print(f"{idx:<4}{nombre:<32}{str(tuple(x.shape)):<22}{n_params:>22,}")
    print("-" * 80)
    total_modelo = sum(p.numel() for p in modelo.parameters())
    assert total == total_modelo, f"Suma por capas ({total}) != total modelo ({total_modelo})"
    print(f"{'':<4}{'TOTAL':<32}{'':<22}{total_modelo:>22,}")
    return filas

