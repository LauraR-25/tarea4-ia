"""Ciclo de entrenamiento, evaluación y curvas."""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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
        perdida = criterio(logits, y)    # CrossEntropyLoss recibe logits
        perdida.backward()               # backpropagation: calcula gradientes
        optimizador.step()               # actualiza los pesos
        perdida_total += perdida.item() * x.size(0)
        correctas += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return perdida_total / total, correctas / total


@torch.no_grad()
def evaluar(modelo, loader, criterio, dispositivo):
    modelo.eval()  # modo evaluación
    perdida_total, correctas, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dispositivo), y.to(dispositivo)
        logits = modelo(x)
        perdida_total += criterio(logits, y).item() * x.size(0)
        correctas += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return perdida_total / total, correctas / total


def entrenar(modelo, train_loader, val_loader, epocas=20, lr=1e-3,
             weight_decay=0.0, dispositivo="cpu", verbose=True):
    """Entrena con Adam y devuelve el historial de pérdida y exactitud por época."""
    modelo.to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
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
    return hist


def graficar_curvas(hist, titulo=""):
    """Pérdida y exactitud de entrenamiento y validación por época."""
    epocas = range(1, len(hist["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epocas, hist["train_loss"], label="Entrenamiento")
    ax1.plot(epocas, hist["val_loss"], label="Validación")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida")
    ax1.set_title("Pérdida")
    ax1.legend()
    ax2.plot(epocas, hist["train_acc"], label="Entrenamiento")
    ax2.plot(epocas, hist["val_acc"], label="Validación")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Exactitud")
    ax2.set_title("Exactitud")
    ax2.legend()
    if titulo:
        fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()
