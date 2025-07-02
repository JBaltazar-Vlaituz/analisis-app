import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, norm


st.title("🔬 Análisis Estadístico de Control de Calidad")

archivo = st.file_uploader("Carga tu archivo Excel (.xlsx)", type=["xlsx"])

if archivo:
    df_datos = pd.read_excel(archivo, sheet_name="Datos")
    df_limites = pd.read_excel(archivo, sheet_name="Limites")

    st.success("Archivo cargado correctamente.")



def analisis_estadistico(df):
    resultados = {}

    for columna in df.select_dtypes(include=[np.number]).columns:
        serie = df[columna].dropna()
        if len(serie) == 0:
            continue

        media = np.mean(serie)
        std_dev = np.std(serie, ddof=1)
        coef_var = (std_dev / media)*100 if media != 0 else np.nan
        asimetria = skew(serie, bias=False)
        curt = kurtosis(serie, bias=False, fisher=True)

        resultados[columna] = {
            'Media': round(media, 2),
            'Coef. Variación (%)': round(coef_var, 2),
            'Desviación estándar': round(std_dev, 2),
            'Asimetría': round(asimetria, 2),
            'Curtosis': round(curt, 2),
            'Mínimo': round(serie.min(), 2),
            'Máximo': round(serie.max(), 2)
        }

    return pd.DataFrame(resultados).T

if archivo:
    st.subheader("📋 Resumen estadístico")
    st.dataframe(analisis_estadistico(df_datos))

def validar_limites_con_leyenda(df_datos, df_limites):
    resultados = {}
    leyendas = []

    for columna in df_datos.columns:
        if columna not in df_limites.columns:
            continue

        sup_raw = df_limites.loc[df_limites.iloc[:, 0].str.lower() == 'superior', columna].values[0]
        inf_raw = df_limites.loc[df_limites.iloc[:, 0].str.lower() == 'inferior', columna].values[0]

        def convertir(valor, tipo):
            if isinstance(valor, str) and valor.strip().lower() in ['n.a.', 'na', '']:
                return float('inf') if tipo == 'sup' else float('-inf')
            try:
                return float(valor)
            except:
                return float('inf') if tipo == 'sup' else float('-inf')

        sup = convertir(sup_raw, 'sup')
        inf = convertir(inf_raw, 'inf')

        cumple = df_datos[columna].between(inf, sup)
        resultados[columna] = cumple

        if not cumple.all():
            leyendas.append(f"⚠️ {columna} contiene valores fuera de especificación.")

    if not leyendas:
        leyendas.append("✅ Todos los parámetros están dentro de los límites de especificación.")

    return pd.DataFrame(resultados), leyendas

if archivo:
    st.subheader("📐 Verificación de límites")
    _, mensajes = validar_limites_con_leyenda(df_datos, df_limites)
    for mensaje in mensajes:
        st.info(mensaje)


def histograma_con_curva(df, columna, df_limites):
    serie = df[columna].dropna()
    if len(serie) < 2:
        st.warning(f"No hay suficientes datos para graficar {columna}")
        return

    unidad = ''
    try:
        unidad = df_limites.loc[df_limites.iloc[:, 0].str.lower() == 'unidades', columna].values[0]
        if isinstance(unidad, str) and unidad.lower().strip() in ['n.a.', 'na']: unidad = ''
    except: pass

    media = serie.mean()
    std = serie.std()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(serie, bins=10, stat='count', kde=False, ax=ax, color='skyblue', edgecolor='black')

    x = np.linspace(media - 3*std, media + 3*std, 300)
    ancho_bin = (max(serie) - min(serie)) / 10
    gauss = norm.pdf(x, media, std) * len(serie) * ancho_bin
    ax.plot(x, gauss, color='red')

    ax.axvline(media, color='blue', linestyle='--')
    ax.set_title(f"{columna}")
    ax.set_xlabel(f"{columna} ({unidad})" if unidad else columna)
    ax.set_ylabel("Frecuencia")

    leyenda = f"N: {len(serie)} | Media: {media:.2f} | σ: {std:.2f}"
    ax.legend([leyenda, "Distribución Normal"])
    ax.grid(True)
    st.pyplot(fig)









def grafica_control(df, nombre_columna, nombre_eje_x=None, df_limites=None):
    serie = df[nombre_columna].dropna().reset_index(drop=True)
    eje_x = df[nombre_eje_x].dropna().reset_index(drop=True) if nombre_eje_x else serie.index + 1

    media = serie.mean()
    std_dev = serie.std(ddof=1)
    limites = {
        'Media': media,
        '+3σ': media + 3*std_dev,
        '-3σ': media - 3*std_dev
    }

    if df_limites is not None:
        try:
            lim_sup = pd.to_numeric(df_limites.loc[df_limites['Limites'] == 'superior', nombre_columna].values[0], errors='coerce')
            lim_inf = pd.to_numeric(df_limites.loc[df_limites['Limites'] == 'inferior', nombre_columna].values[0], errors='coerce')
            if pd.notna(lim_sup): limites['Límite Superior'] = lim_sup
            if pd.notna(lim_inf): limites['Límite Inferior'] = lim_inf
        except: pass

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eje_x, serie, marker='o', label='Datos')

    for etiqueta, valor in limites.items():
        estilo = {'--': 'gray', '-.': 'red'}
        linea = '-' if 'Media' in etiqueta else '--'
        ax.axhline(valor, linestyle=linea, linewidth=1.2, label=f"{etiqueta}: {valor:.2f}")

    unidad = ''
    try:
        unidad = df_limites.loc[df_limites['Limites'] == 'unidades', nombre_columna].values[0]
    except: pass

    ax.set_title(f'Gráfico de Control: {nombre_columna}')
    ax.set_xlabel(nombre_eje_x if nombre_eje_x else "Índice")
    ax.set_ylabel(f"{nombre_columna} ({unidad})" if unidad else nombre_columna)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if archivo:
    st.subheader("📌 Selecciona una variable para análisis detallado")
    variable = st.selectbox("Variable numérica", df_datos.select_dtypes(include='number').columns)

    st.markdown("### 📊 Histograma")
    histograma_con_curva(df_datos, variable, df_limites)

    st.markdown("### 📉 Gráfico de Control")
    grafica_control(df_datos, variable, nombre_eje_x='LOTE', df_limites=df_limites)