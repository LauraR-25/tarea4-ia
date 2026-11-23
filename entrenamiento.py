"""Ciclo de entrenamiento, evaluación, curvas y análisis (puntos 2, 3 y 4)."""
import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from datos import NOMBRES_CLASES, desnormalizar


def contar_parametros(modelo):
    """Número total de parámetros entrenables, sumando p.numel()."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def entrenar_una_epoca(modelo, loader, criterio, optimizador, dispositivo):
    modelo.train()  # activa el modo entrenamiento (dropout/batchnorm)
    perdida_total, correctas, total = 0.0, 0, 0
    for x, y in loader:  # cada iteración es un mini-batch = un paso
        x, y = x.to(dispositivo), y.to(dispositivo)
        optimizador.zero_grad()          # borra los gradientes del paso anterior
        logits = modelo(x)               # forward
        perdida = criterio(logits, y)    # CrossEntropyLoss recibe logits (sin softmax previo)
        perdida.backward()               # backpropagation: calcula gradientes
        optimizador.step()               # actualiza los pesos
        perdida_total += perdida.item() * x.size(0)
        correctas += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return perdida_total / total, correctas / total


@torch.no_grad()
def evaluar(modelo, loader, criterio, dispositivo):
    modelo.eval()  # modo evaluación (desactiva dropout)
    perdida_total, correctas, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dispositivo), y.to(dispositivo)
        logits = modelo(x)
        perdida_total += criterio(logits, y).item() * x.size(0)
        correctas += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return perdida_total / total, correctas / total


def entrenar(modelo, train_loader, val_loader, epocas=20, lr=1e-3,
             weight_decay=0.0, paciencia=None, dispositivo="cpu", verbose=True):
    """Entrena con Adam y devuelve el historial de pérdida y exactitud por época.

    Si `paciencia` no es None, aplica early stopping sobre la pérdida de validación
    y restaura los pesos de la mejor época al finalizar.
    """
    modelo.to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "mejor_epoca": 1,
    }
    mejor_val_loss = float("inf")
    mejor_estado = None
    sin_mejora = 0

    for ep in range(1, epocas + 1):
        tl, ta = entrenar_una_epoca(modelo, train_loader, criterio, optimizador, dispositivo)
        vl, va = evaluar(modelo, val_loader, criterio, dispositivo)
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)
        if verbose:
            print(f"Época {ep:2d}/{epocas} | pérdida train {tl:.4f} val {vl:.4f} "
                  f"| exactitud train {ta:.4f} val {va:.4f}")

        if vl < mejor_val_loss - 1e-4:
            mejor_val_loss = vl
            hist["mejor_epoca"] = ep
            mejor_estado = copy.deepcopy(modelo.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1
            if paciencia is not None and sin_mejora >= paciencia:
                if verbose:
                    print(f"Early stopping en época {ep} (mejor val_loss en época {hist['mejor_epoca']}: {mejor_val_loss:.4f})")
                break

    if paciencia is not None and mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return hist


def graficar_curvas(hist, titulo="", marcar_mejor=False):
    """Pérdida y exactitud de entrenamiento y validación por época."""
    epocas = range(1, len(hist["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epocas, hist["train_loss"], marker="o", ms=3, label="Entrenamiento")
    ax1.plot(epocas, hist["val_loss"], marker="s", ms=3, label="Validación")
    if marcar_mejor and "mejor_epoca" in hist:
        ax1.axvline(hist["mejor_epoca"], color="gray", ls="--", lw=1,
                    label=f"Mejor época ({hist['mejor_epoca']})")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida")
    ax1.set_title("Pérdida")
    ax1.legend()

    ax2.plot(epocas, hist["train_acc"], marker="o", ms=3, label="Entrenamiento")
    ax2.plot(epocas, hist["val_acc"], marker="s", ms=3, label="Validación")
    if marcar_mejor and "mejor_epoca" in hist:
        ax2.axvline(hist["mejor_epoca"], color="gray", ls="--", lw=1,
                    label=f"Mejor época ({hist['mejor_epoca']})")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Exactitud")
    ax2.set_title("Exactitud")
    ax2.legend()
    if titulo:
        fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()


def comparar_curvas_regularizacion(hist_sin, hist_con, nombre_arq="Mejor CNN"):
    """Compara las curvas antes y después de corregir el sobreajuste."""
    ep_sin = range(1, len(hist_sin["train_loss"]) + 1)
    ep_con = range(1, len(hist_con["train_loss"]) + 1)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.3))

    ejes[0].plot(ep_sin, hist_sin["train_loss"], "--", color="#1f77b4", label="Train (sin reg.)")
    ejes[0].plot(ep_sin, hist_sin["val_loss"], "-", color="#1f77b4", label="Val (sin reg.)")
    ejes[0].plot(ep_con, hist_con["train_loss"], "--", color="#2ca02c", label="Train (con reg.)")
    ejes[0].plot(ep_con, hist_con["val_loss"], "-", color="#2ca02c", label="Val (con reg.)")
    if "mejor_epoca" in hist_con:
        ejes[0].axvline(hist_con["mejor_epoca"], color="#2ca02c", ls=":", lw=1.2,
                        label=f"Early stop mejor época ({hist_con['mejor_epoca']})")
    ejes[0].set_xlabel("Época")
    ejes[0].set_ylabel("Pérdida")
    ejes[0].set_title(f"{nombre_arq}: Pérdida")
    ejes[0].legend(fontsize=8.5)

    ejes[1].plot(ep_sin, hist_sin["train_acc"], "--", color="#1f77b4", label="Train (sin reg.)")
    ejes[1].plot(ep_sin, hist_sin["val_acc"], "-", color="#1f77b4", label="Val (sin reg.)")
    ejes[1].plot(ep_con, hist_con["train_acc"], "--", color="#2ca02c", label="Train (con reg.)")
    ejes[1].plot(ep_con, hist_con["val_acc"], "-", color="#2ca02c", label="Val (con reg.)")
    ejes[1].set_xlabel("Época")
    ejes[1].set_ylabel("Exactitud")
    ejes[1].set_title(f"{nombre_arq}: Exactitud")
    ejes[1].legend(fontsize=8.5)

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predecir_conjunto(modelo, loader, dispositivo):
    """Devuelve (imagenes, y_real, y_pred, probabilidades) sobre todo el loader."""
    modelo.eval()
    imagenes, y_real, y_pred, probs = [], [], [], []
    for x, y in loader:
        logits = modelo(x.to(dispositivo))
        p = torch.softmax(logits, dim=1).cpu()
        imagenes.append(x.cpu())
        y_real.append(y.cpu())
        y_pred.append(p.argmax(dim=1))
        probs.append(p)
    return (
        torch.cat(imagenes, dim=0),
        torch.cat(y_real, dim=0),
        torch.cat(y_pred, dim=0),
        torch.cat(probs, dim=0),
    )


def graficar_matriz_confusion(y_real, y_pred, nombres=NOMBRES_CLASES):
    """Calcula, grafica e imprime las principales confusiones en el conjunto de prueba."""
    n = len(nombres)
    matriz = torch.zeros((n, n), dtype=torch.int64)
    for r, p in zip(y_real.tolist(), y_pred.tolist()):
        matriz[r, p] += 1

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(matriz.numpy(), cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(nombres, rotation=45, ha="right")
    ax.set_yticklabels(nombres)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión en prueba (Mejor CNN)")

    umbral = matriz.max().item() / 2.0
    for i in range(n):
        for j in range(n):
            val = matriz[i, j].item()
            color = "white" if val > umbral else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8.5)
    plt.tight_layout()
    plt.show()

    # Pares más confundidos fuera de la diagonal
    confusiones = []
    for i in range(n):
        for j in range(n):
            if i != j and matriz[i, j].item() > 0:
                confusiones.append((matriz[i, j].item(), nombres[i], nombres[j]))
    confusiones.sort(reverse=True)
    print("Principales pares confundidos (Real -> Predicha: casos):")
    for cant, real, pred in confusiones[:8]:
        print(f"  {real:<10} -> {pred:<10}: {cant:3d} ejemplos")
    return matriz


def mostrar_mal_clasificadas(imagenes, y_real, y_pred, probs,
                             nombres=NOMBRES_CLASES, n_mostrar=10):
    """Muestra ejemplos mal clasificados con la probabilidad asignada por la red."""
    idx_err = (y_real != y_pred).nonzero().flatten().tolist()
    n_mostrar = min(n_mostrar, len(idx_err))
    cols = 5
    filas = math.ceil(n_mostrar / cols)
    fig, ejes = plt.subplots(filas, cols, figsize=(2.6 * cols, 2.9 * filas))
    ejes = np.array(ejes).reshape(-1)

    for k in range(n_mostrar):
        i = idx_err[k]
        real = y_real[i].item()
        pred = y_pred[i].item()
        p_pred = 100.0 * probs[i, pred].item()
        p_real = 100.0 * probs[i, real].item()
        ax = ejes[k]
        ax.imshow(desnormalizar(imagenes[i]).squeeze(), cmap="gray")
        ax.set_title(
            f"Real: {nombres[real]} ({p_real:.1f}%)\n"
            f"Pred: {nombres[pred]} ({p_pred:.1f}%)",
            fontsize=8.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(n_mostrar, len(ejes)):
        ejes[k].axis("off")

    plt.suptitle("Ejemplos mal clasificados en prueba y probabilidad asignada por la CNN", y=1.02)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def visualizar_filtros_conv1(modelo, img_ejemplo=None):
    """Visualiza los filtros 3x3 de la primera capa convolucional (y sus mapas de activación)."""
    conv1 = modelo.primera_conv
    pesos = conv1.weight.detach().cpu()  # (c_out, 1, k, k)
    n_filtros = pesos.shape[0]
    cols = 8
    filas = math.ceil(n_filtros / cols)
    vmax = float(pesos.abs().max().item())

    fig, ejes = plt.subplots(filas, cols, figsize=(1.75 * cols, 1.85 * filas))
    ejes = np.array(ejes).reshape(-1)
    for idx in range(n_filtros):
        ax = ejes[idx]
        w = pesos[idx, 0].numpy()
        im = ax.imshow(w, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        for r in range(w.shape[0]):
            for c in range(w.shape[1]):
                ax.text(c, r, f"{w[r, c]:.2f}", ha="center", va="center", fontsize=6.5, color="black")
        ax.set_title(f"Filtro {idx}", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(n_filtros, len(ejes)):
        ejes[idx].axis("off")
    fig.colorbar(im, ax=ejes[:n_filtros].tolist(), shrink=0.75, pad=0.02, label="Peso")
    plt.suptitle(f"Filtros aprendidos en la primera capa convolucional ({n_filtros} filtros de {pesos.shape[2]}x{pesos.shape[3]})", y=1.02)
    plt.show()

    if img_ejemplo is not None:
        dispositivo = next(modelo.parameters()).device
        act = torch.relu(conv1(img_ejemplo.unsqueeze(0).to(dispositivo))).squeeze(0).cpu()
        n_mostrar = min(8, n_filtros)
        fig, ejes = plt.subplots(1, n_mostrar + 1, figsize=(1.7 * (n_mostrar + 1), 2.2))
        ejes[0].imshow(desnormalizar(img_ejemplo).squeeze(), cmap="gray")
        ejes[0].set_title("Entrada", fontsize=8.5)
        ejes[0].axis("off")
        for idx in range(n_mostrar):
            ejes[idx + 1].imshow(act[idx].numpy(), cmap="viridis")
            ejes[idx + 1].set_title(f"Act. filtro {idx}", fontsize=8.5)
            ejes[idx + 1].axis("off")
        plt.suptitle("Mapas de características tras ReLU de los primeros 8 filtros", y=1.02)
        plt.tight_layout()
        plt.show()

