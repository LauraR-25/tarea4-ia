# Tarea 4 – Inteligencia Artificial (Puntos 1, 2, 3 y 4)

## Estructura
| Archivo | Contenido |
|---|---|
| `tarea4.ipynb` | Notebook completo con los puntos 1 (convolución a mano), 2 (datos y MLP), 3 (CNN y arquitectura) y 4 (análisis) |
| `convolucion.py` | Convolución, ReLU y max pooling manuales; fórmulas de parámetros |
| `datos.py` | Semilla, carga de FashionMNIST, separación train/val/test, gráficas |
| `modelos.py` | Clases `MLP(nn.Module)` y `CNN(nn.Module)`, y verificación de formas y parámetros por capa (`resumen_capas`) |
| `entrenamiento.py` | Ciclo de entrenamiento (con soporte de `weight_decay` y early stopping), evaluación, curvas, matriz de confusión, errores y visualización de filtros |
| `requirements.txt` | Dependencias |

## Uso
```bash
pip install -r requirements.txt
jupyter notebook tarea4.ipynb
```
El dataset se descarga solo en `data/` la primera vez. La separación se guarda en `splits.pt`
(semilla 42), los pesos del MLP en `mlp.pt` y los pesos de la mejor CNN en `mejor_cnn.pt`.
