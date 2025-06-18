import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar datasets
steam_df = pd.read_csv("steam.csv")
media_df = pd.read_csv("steam_media_small.csv")

# Unir imagen de portada
steam_df = pd.merge(
    steam_df,
    media_df[['appid', 'header_image']],
    left_on='appid',
    right_on='appid',
    how='left'
)
steam_df.drop(columns=['appid'], inplace=True)

# Cargar archivos para recomendación
df = joblib.load('modelo/df.pkl')
X_encoded = np.load('modelo/X_encoded.npy')
clusters_dec = np.load('modelo/clusters_dec.npy')
tags_bin = joblib.load('modelo/tags_bin.pkl')
genres_bin = joblib.load('modelo/genres_bin.pkl')

# Función recomendadora (igual que antes)
def recomendar_juegos_por_nombre(nombre_juego, n_recomendaciones=5, min_tags_comunes=2, min_genres_comunes=2):
    if nombre_juego not in df['name'].values:
        similares = df['name'][df['name'].str.contains(nombre_juego, case=False)].head(3).tolist()
        raise ValueError(f"Juego no encontrado. Sugerencias: {similares}")
    
    juego_idx = df.index[df['name'] == nombre_juego][0]
    tags_objetivo = set(df.loc[juego_idx, 'tags_list'])
    genres_objetivo = set(df.loc[juego_idx, 'genres_list'])
    cluster_objetivo = clusters_dec[juego_idx]
    vector_objetivo = X_encoded[juego_idx]

    TAG_WEIGHT = 5.0
    GENRE_WEIGHT = 1.0
    resultados = []
    indices_incluidos = {juego_idx}

    def procesar_indices(indices, min_tags, min_genres, resultados, indices_incluidos):
        for idx in indices:
            if idx in indices_incluidos:
                continue
            tags_actual = set(df.loc[idx, 'tags_list'])
            genres_actual = set(df.loc[idx, 'genres_list'])
            tags_comunes = tags_objetivo.intersection(tags_actual)
            genres_comunes = genres_objetivo.intersection(genres_actual)
            if len(tags_comunes) >= min_tags and len(genres_comunes) >= min_genres:
                tag_freqs = tags_bin[list(tags_comunes)].mean(axis=0)
                genre_freqs = genres_bin[list(genres_comunes)].mean(axis=0)
                tag_rarity_score = sum(1 / (freq + 1e-6) for freq in tag_freqs) if tag_freqs.size > 0 else 1.0
                genre_rarity_score = sum(1 / (freq + 1e-6) for freq in genre_freqs) if genre_freqs.size > 0 else 1.0
                similitud = 1 / (1 + 0.5 * np.linalg.norm(vector_objetivo - X_encoded[idx]))
                weighted_similitud = similitud * (TAG_WEIGHT * tag_rarity_score + GENRE_WEIGHT * genre_rarity_score)

                resultados.append({
                    'indice': idx,
                    'tags_comunes': list(tags_comunes),
                    'n_tags_comunes': len(tags_comunes),
                    'genres_comunes': list(genres_comunes),
                    'n_genres_comunes': len(genres_comunes),
                    'similitud': weighted_similitud,
                    'rating': df.loc[idx, 'positive_ratio']
                })
                indices_incluidos.add(idx)

    procesar_indices(np.arange(len(df)), min_tags_comunes, min_genres_comunes, resultados, indices_incluidos)

    if len(resultados) < n_recomendaciones:
        indices_cluster = np.where(clusters_dec == cluster_objetivo)[0]
        procesar_indices(indices_cluster, 1, 0, resultados, indices_incluidos)

    if len(resultados) < n_recomendaciones:
        indices_restantes = np.where(clusters_dec != cluster_objetivo)[0]
        procesar_indices(indices_restantes, 1, 0, resultados, indices_incluidos)

    resultados_ordenados = sorted(resultados, key=lambda x: (-x['n_genres_comunes'], -x['n_tags_comunes'], -x['similitud'], -x['rating']))
    top_indices = [r['indice'] for r in resultados_ordenados[:n_recomendaciones]]

    recomendaciones = []
    for idx in top_indices:
        juego = df.loc[idx]
        recomendaciones.append({
            'nombre': juego['name'],
            'developer': juego['developer'],
            'tags_comunes': ', '.join(t.replace('tag_', '') for t in set(df.loc[juego_idx, 'tags_list']).intersection(juego['tags_list'])),
            'genres_comunes': ', '.join(g.replace('gen_', '') for g in set(df.loc[juego_idx, 'genres_list']).intersection(juego['genres_list'])),
            'rating': f"{juego['positive_ratio']:.0%}",
            'precio': f"${juego['price']:.2f}" if juego['price'] > 0 else "Gratis",
            'cluster': clusters_dec[idx]
        })
    return recomendaciones

# ================= INTERFAZ STREAMLIT ===================

st.set_page_config(layout="wide")
st.title("🎮 Sistema de Recomendación de Videojuegos")

nombre_juego = st.selectbox("Selecciona un juego:", sorted(steam_df['name'].dropna().unique()))
juego_seleccionado = steam_df[steam_df['name'] == nombre_juego].iloc[0]

# Mostrar info del juego seleccionado
st.markdown("### 🎯 Juego seleccionado")
col1, col2 = st.columns([1, 3])
with col1:
    if pd.notna(juego_seleccionado['header_image']):
        st.image(juego_seleccionado['header_image'], use_column_width=True)
with col2:
    st.write(f"**Desarrollador:** {juego_seleccionado['developer']}")
    st.write(f"**Fecha de lanzamiento:** {juego_seleccionado['release_date']}")
    st.write(f"**Plataformas:** {juego_seleccionado['platforms']}")
    st.write(f"**Géneros:** {juego_seleccionado['genres']}")
    st.write(f"**Categorías:** {juego_seleccionado['categories']}")

# Recomendaciones
if st.button("🔍 Recomendar juegos similares"):
    try:
        recomendaciones = recomendar_juegos_por_nombre(nombre_juego)
        st.markdown("## 🧠 Juegos recomendados")

        cols = st.columns(3)
        for i, rec in enumerate(recomendaciones):
            with cols[i % 3]:
                st.markdown(f"**🎮 {rec['nombre']}**")
                img_url = steam_df[steam_df['name'] == rec['nombre']]['header_image'].values
                if len(img_url) > 0 and pd.notna(img_url[0]):
                    st.image(img_url[0], use_column_width=True)
                st.write(f"- 👨‍💻 Developer: {rec['developer']}")
                st.write(f"- 🏷️ Tags comunes: {rec['tags_comunes']}")
                st.write(f"- 🎵 Géneros comunes: {rec['genres_comunes']}")
                st.write(f"- ⭐ Rating: {rec['rating']}")
                st.write(f"- 💰 Precio: {rec['precio']}")
                st.write(f"- 🧩 Cluster: {rec['cluster']}")
                st.markdown("---")
    except ValueError as e:
        st.error(str(e))
