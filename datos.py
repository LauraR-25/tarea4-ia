"""Punto 2: carga del dataset, separación train/val/test y visualización.

Dataset: FashionMNIST (10 clases, imágenes 28x28 en escala de grises).
La separación se genera UNA vez con una semilla fija y se guarda en
splits.pt, de modo que todos los modelos de la tarea usan los mismos datos.
"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEMILLA = 42
MEDIA, STD = 0.2860, 0.3530  # estadísticas de FashionMNIST (entrenamiento)
NOMBRES_CLASES = [
    "Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín",
]


def fijar_semilla(semilla=SEMILLA):
    """Fija las semillas de Python, NumPy y PyTorch para reproducibilidad."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def obtener_dispositivo():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cargar_datasets(root="data", n_entrenamiento=20000, frac_val=0.2,
                    semilla=SEMILLA, archivo_split="splits.pt"):
    """Devuelve (train_ds, val_ds, test_ds).

    - De los 60 000 ejemplos de entrenamiento se toma una muestra aleatoria de
      `n_entrenamiento` (el enunciado lo permite si hay más de 20 000).
    - Esa muestra se separa en entrenamiento y validación (`frac_val`).
    - El conjunto de prueba es el test oficial de FashionMNIST (10 000 imágenes).
    """
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEDIA,), (STD,))])
    completo = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST(root, train=False, download=True, transform=tf)

    if os.path.exists(archivo_split):
        split = torch.load(archivo_split)
    else:
        g = torch.Generator().manual_seed(semilla)
        perm = torch.randperm(len(completo), generator=g)[:n_entrenamiento]
        n_val = int(n_entrenamiento * frac_val)
        split = {"val": perm[:n_val].tolist(), "train": perm[n_val:].tolist()}
        torch.save(split, archivo_split)

    return Subset(completo, split["train"]), Subset(completo, split["val"]), test_ds


def obtener_etiquetas(ds):
    """Etiquetas (tensor) de un Dataset o de un Subset."""
    if isinstance(ds, Subset):
        return ds.dataset.targets[ds.indices]
    return ds.targets


def crear_dataloaders(train_ds, val_ds, test_ds, batch_size=128, semilla=SEMILLA):
    g = torch.Generator().manual_seed(semilla)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def desnormalizar(img):
    return img * STD + MEDIA


def mostrar_ejemplos(ds, n_por_clase=6, nombres=NOMBRES_CLASES):
    """Muestra `n_por_clase` imágenes de cada clase (una fila por clase)."""
    etiquetas = obtener_etiquetas(ds)
    n_clases = len(nombres)
    fig, ejes = plt.subplots(n_clases, n_por_clase, figsize=(1.6 * n_por_clase, 1.6 * n_clases))
    for c in range(n_clases):
        posiciones = (etiquetas == c).nonzero().flatten()[:n_por_clase].tolist()
        for k in range(n_por_clase):
            ax = ejes[c, k]
            img, _ = ds[posiciones[k]]
            ax.imshow(desnormalizar(img).squeeze(), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(nombres[c], rotation=0, ha="right", va="center", fontsize=9)
    plt.suptitle("Ejemplos por clase", y=1.0)
    plt.tight_layout()
    plt.show()


def proporcion_por_clase(train_ds, val_ds, test_ds, nombres=NOMBRES_CLASES):
    """Imprime y grafica la proporción de ejemplos por clase en cada conjunto."""
    conjuntos = {"Entrenamiento": train_ds, "Validación": val_ds, "Prueba": test_ds}
    n_clases = len(nombres)
    proporciones = {}
    for nombre, ds in conjuntos.items():
        conteo = torch.bincount(obtener_etiquetas(ds), minlength=n_clases).float()
        proporciones[nombre] = conteo / conteo.sum()

    print(f"{'Clase':<12}" + "".join(f"{n:>16}" for n in conjuntos))
    for c in range(n_clases):
        print(f"{nombres[c]:<12}" + "".join(f"{100 * proporciones[n][c]:>15.2f}%" for n in conjuntos))

    x = np.arange(n_clases)
    ancho = 0.27
    plt.figure(figsize=(10, 4))
    for i, (nombre, p) in enumerate(proporciones.items()):
        plt.bar(x + (i - 1) * ancho, 100 * p.numpy(), ancho, label=nombre)
    plt.axhline(10, color="k", ls="--", lw=0.8, label="Uniforme (10 %)")
    plt.xticks(x, nombres, rotation=45)
    plt.ylabel("% de ejemplos")
    plt.title("Proporción de ejemplos por clase")
    plt.legend()
    plt.tight_layout()
    plt.show()
