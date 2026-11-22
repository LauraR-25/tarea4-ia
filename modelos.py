"""Modelos de la tarea. Cada red es una clase que extiende nn.Module."""
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
