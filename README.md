# README - SCVideogames

## 📝 Descripción

**SCVideogames** es un sistema de recomendación de videojuegos de Steam que permite a los usuarios descubrir títulos similares a sus juegos favoritos. Utiliza técnicas de **Deep Embedded Clustering (DEC)** para aprender representaciones latentes y agrupar juegos, mejorando la pertinencia de las sugerencias.

## ✨ Características Principales

- **Recomendación basada en DEC**: Pipeline que combina autoencoder y K-Means dentro de un bucle de optimización con `KLD loss` para generar clusters de juegos.
- **Interfaz Web con Streamlit**: Aplicación intuitiva y responsiva para la interacción con el usuario.
- **Base de Datos de Steam**: Más de 40,000 juegos con metadatos completos (etiquetas, géneros, categorías, ratings, tiempo de juego y precio).
- **Estrategia de Múltiples Etapas**:
  1. Coincidencia estricta (etiquetas y géneros comunes mínimos).
  2. Respaldo por clústeres DEC con criterios relajados.
  3. Recomendación global con criterios mínimos.
- **Métricas de Evaluación**: Cálculo de similitud de Jaccard y evaluación de desempeño del sistema.

## 🤖 Deep Embedded Clustering (DEC)

1. **Autoencoder**:
   - Encoder reduce la dimensionalidad de los vectores de características (etiquetas y géneros) a un espacio latente `Z`.
   - Decoder reconstruye las entradas, asegurando una representación compacta.

2. **Inicialización de Clústeres**:
   - Se aplica K-Means sobre `Z` para obtener centros iniciales de clúster.

3. **Optimización Iterativa**:
   - Se calcula la **distribución objetivo** \(p_{ij}\) a partir de las asignaciones suaves \(q_{ij}\).
   - Se minimiza la **pérdida de divergencia KL** entre \(p_{ij}\) y \(q_{ij}\) ajustando encoder y centros de K-Means.

4. **Asignación Final**:
   - Cada juego recibe un cluster definitivo en `Z`.
   - Las recomendaciones se basan en la cercanía en el espacio latente y la pertenencia a cluster.

> Con DEC, la representación latente captura patrones complejos de similitud, mejorando la calidad de las recomendaciones para todo tipo de juegos.

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: UI Web.
- **TensorFlow**: Autoencoder y entrenamiento DEC.
- **Scikit-learn**: K-Means, MultiLabelBinarizer.
- **Pandas & NumPy**: Manipulación y cálculo de datos.
- **Joblib**: Serialización de modelos y artefactos.

## 🚀 Instalación

```bash
git clone https://github.com/antorm08/SCVideogames.git
cd SCVideogames
pip install -r requirements.txt
````

> Asegúrate de tener:
>
> * `steam.csv` y `steam_media_small.csv`
> * Carpeta `modelo/` con los pesos y artefactos DEC.

## 📖 Uso

1. Inicia la app:

   ```bash
   streamlit run app.py
   ```
2. Abre `http://localhost:8501` en tu navegador.
3. Elige un juego y pulsa **"🔍 Recomendar juegos similares"**.

## 📁 Estructura del Proyecto

```
SCVideogames/
├── app.py                    # App de Streamlit
├── steam.csv                 # Datos de Steam
├── steam_media_small.csv     # Metadatos adicionales
├── modelo/                   # Autoencoder + DEC
│   ├── encoder.h5
│   ├── decoder.h5
│   ├── cluster_centers.npy
│   └── artifacts.pkl
└── README.md                 # Este archivo
```

## 🎯 Funciones Clave

* `train_dec()`              : Entrena el autoencoder y refina clusters con DEC.
* `compute_target_distribution(q)` : Calcula distribución objetivo para `KLD loss`.
* `recommend_via_dec(game_id)`   : Obtiene recomendaciones usando cluster latente.
* `evaluate_jaccard()`       : Mide similitud de conjuntos para evaluar precisión.

## 🚀 Despliegue

Diseñado para desplegar en plataformas como Render o Heroku:

* Carga modelo DEC en memoria.
* Responde a consultas en tiempo real.


## 📄 Licencia

MIT License. Consulta `LICENSE` para detalles.

```
```
