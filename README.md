# Tarea 4 – Inteligencia Artificial (puntos 1 y 2)

## Estructura
| Archivo | Contenido |
|---|---|
| `tarea4.ipynb` | Notebook con el punto 1 (convolución a mano) y el punto 2 (datos y MLP) |
| `convolucion.py` | Convolución, ReLU y max pooling manuales; fórmulas de parámetros |
| `datos.py` | Semilla, carga de FashionMNIST, separación train/val/test, gráficas |
| `modelos.py` | Clase `MLP(nn.Module)` |
| `entrenamiento.py` | Ciclo de entrenamiento, evaluación, curvas, conteo de parámetros |
| `requirements.txt` | Dependencias |

## Uso
```bash
pip install -r requirements.txt
jupyter notebook tarea4.ipynb
```
El dataset se descarga solo en `data/` la primera vez. La separación se guarda en `splits.pt`
(semilla 42) y los pesos del MLP en `mlp.pt`.
