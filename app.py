import streamlit as st
import os
import tempfile
import urllib.request
import requests
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from PIL import Image
import io
import base64
import pandas as pd
import geopandas as gpd
from folium.plugins import MarkerCluster

st.set_page_config(page_title="Accesibilidad Centros de Salud", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .legend-item { display:flex; align-items:center; margin:6px 0; }
    .legend-color { width:24px; height:18px; border-radius:3px; margin-right:10px;
        border:1px solid rgba(255,255,255,0.3); display:inline-block; flex-shrink:0; }
    .legend-label { color:white; font-size:13px; }
    [data-testid="stMetricValue"] { font-size:28px; }
    .separator { border-top:1px solid rgba(255,255,255,0.1); margin:15px 0; }
    .info-box { background-color:rgba(255,255,255,0.05); border-radius:8px;
        padding:15px; margin:10px 0; border-left:4px solid #2980b9; }
    .info-box p { margin:4px 0; font-size:13px; }
    .pna-dot { width:12px; height:12px; border-radius:50%; display:inline-block;
        margin-right:8px; border:1px solid rgba(255,255,255,0.4); }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONSTANTES
# ============================================
COLORES_CAT = {
    "1-2 NUCLEOS":         {"color": "#3498db", "radio": 4,  "label": "1-2 Nucleos"},
    "3-5 NUCLEOS":         {"color": "#2ecc71", "radio": 6,  "label": "3-5 Nucleos"},
    "6-12 NUCLEOS":        {"color": "#f39c12", "radio": 8,  "label": "6-12 Nucleos"},
    "SERVICIOS AMPLIADOS": {"color": "#e74c3c", "radio": 10, "label": "Servicios Ampliados"},
}

ESQUEMAS_ISOCRONA = {
    "Azules": {'colores':['#d4e6f1','#a2cce3','#5faed1','#2980b9','#1a5276'],
               'labels':['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']},
    "Rojos":  {'colores':['#f9e4e4','#f1a9a9','#e06666','#cc0000','#800000'],
               'labels':['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']},
    "Verdes": {'colores':['#d5f5e3','#a9dfbf','#52be80','#27ae60','#1e8449'],
               'labels':['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']},
}

RANGOS_ISOCRONA = [(0.01,30),(30,60),(60,120),(120,450),(450,50000)]

# Rangos hospitales: 0-30, 30-60, >60 min
RANGOS_HOSPITALES = [(0.01, 30), (30, 60), (60, 99999)]

ESQUEMA_HOSP_TODOS = {
    'colores': ['#2ecc71', '#f39c12', '#e74c3c'],
    'labels':  ['< 30 min', '30 - 60 min', '> 60 min']
}
ESQUEMA_HOSP_CAMAS = {
    'colores': ['#5dade2', '#2e86c1', '#1a5276'],
    'labels':  ['< 30 min', '30 - 60 min', '> 60 min']
}

# Colores para grado de marginacion (gm)
COLORES_GM = {
    "Muy alto":  "#7b0c0c",
    "Alto":      "#d62728",
    "Medio":     "#ff7f0e",
    "Bajo":      "#bcbd22",
    "Muy bajo":  "#2ca02c",
    "ND":        "#aaaaaa",
}


# Rangos y colores del HeatMap SSS — según QGIS
# Esquema A (amarillo-rojo): poblacion total SSS alta
# Esquema B (verde-amarillo): poblacion SSS baja-media
RANGOS_SSS = [(1, 100),(100, 500),(500, 3000),(3000, 15000),(15000, 50000),(50000, 300001)]

ESQUEMAS_SSS = {
    "Calor (amarillo-rojo)": {
        'colores': ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026','#800026'],
        'labels':  ['1-100','100-500','500-3k','3k-15k','15k-50k','50k-300k']
    },
    "Verde-amarillo": {
        'colores': ['#006837','#31a354','#78c679','#c2e699','#ffffcc','#ffeda0'],
        'labels':  ['1-100','100-500','500-3k','3k-15k','15k-50k','50k-300k']
    },
}

# ============================================
# FUNCIONES DE CARGA
# ============================================
# ============================================
# URLs DE ARCHIVOS PESADOS EN ONEDRIVE
# Reemplaza cada URL con tu link de descarga directa
# ============================================
HF_BASE = "https://huggingface.co/datasets/UnicornCodes/dashboard-isocronas/resolve/main"

ARCHIVOS_REMOTOS = {
    "AC_PN_NUMM_4326.tif": f"{HF_BASE}/AC_PN_NUMM_4326.tif",
    "HeatMap_sss.tif":     f"{HF_BASE}/HeatMap_sss.tif",
    "Distanc_SNA.tif":     f"{HF_BASE}/Distanc_SNA.tif",
    "DA_CAMAS.tif":        f"{HF_BASE}/DA_CAMAS.tif",
}

def obtener_ruta_archivo(nombre):
    """
    Si el archivo existe localmente (modo local/dev) lo usa directo.
    Si no, lo descarga desde Hugging Face con streaming.
    """
    # Modo local — archivo en la misma carpeta que app.py
    if os.path.exists(nombre):
        return nombre

    # Modo cloud — descargar desde Hugging Face
    url = ARCHIVOS_REMOTOS.get(nombre)
    if not url:
        raise FileNotFoundError(
            f"'{nombre}' no encontrado localmente y sin URL configurada.")

    tmp_dir  = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, nombre)

    if not os.path.exists(tmp_path):
        # Streaming con requests — más robusto para archivos grandes
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        tmp_path_partial = tmp_path + ".part"
        with open(tmp_path_partial, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.rename(tmp_path_partial, tmp_path)

    return tmp_path


@st.cache_data
def cargar_raster_isocrona():
    ruta = obtener_ruta_archivo("AC_PN_NUMM_4326.tif")
    with rasterio.open(ruta) as src:
        data = src.read(1, out_shape=(src.height // 4, src.width // 4),
                        resampling=rasterio.enums.Resampling.average).astype(float)
        bounds = (src.bounds.bottom, src.bounds.left, src.bounds.top, src.bounds.right)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    data[data == -9999] = np.nan
    data[data >= 1e+30] = np.nan
    data[data == 0] = np.nan
    return data, bounds


@st.cache_data
def cargar_raster_sss():
    """Carga HeatMap SSS — reproyecta si está en EPSG:6372"""
    ruta = obtener_ruta_archivo("HeatMap_sss.tif")
    with rasterio.open(ruta) as src:
        crs_actual = src.crs
        if crs_actual and crs_actual.to_epsg() != 4326:
            # Reproyectar al vuelo
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:4326', src.width // 4, src.height // 4, *src.bounds)
            data = np.empty((height, width), dtype=float)
            reproject(source=rasterio.band(src, 1),
                      destination=data,
                      src_transform=src.transform,
                      src_crs=src.crs,
                      dst_transform=transform,
                      dst_crs='EPSG:4326',
                      resampling=Resampling.average)
            # Recalcular bounds en 4326
            # array_bounds devuelve (left, bottom, right, top)
            from rasterio.transform import array_bounds
            b = array_bounds(height, width, transform)
            left_b, bottom_b, right_b, top_b = b
            bounds = (bottom_b, left_b, top_b, right_b)  # (bottom, left, top, right)
        else:
            scale = 4
            data = src.read(1, out_shape=(src.height // scale, src.width // scale),
                            resampling=rasterio.enums.Resampling.average).astype(float)
            bounds = (src.bounds.bottom, src.bounds.left, src.bounds.top, src.bounds.right)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan

    data = data.astype(float)
    data[data == -9999] = np.nan
    data[data >= 1e+30] = np.nan
    data[data <= 0] = np.nan
    return data, bounds


@st.cache_data
def cargar_pna():
    gdf = gpd.read_file("PNA-IMSSB.gpkg")
    return gdf[['clues_imb','nombre_de_la_unidad','categoria_gerencial',
                'nombre_de_tipologia','entidad','municipio','latitud','longitud']]


@st.cache_data
def cargar_capas_poligonos():
    TOL = 0.01
    entidad   = gpd.read_file("Entidad.gpkg").to_crs(4326)
    entidad['geometry'] = entidad['geometry'].simplify(TOL)
    edos_nofed = gpd.read_file("EDOS_NOFED.gpkg").to_crs(4326)
    edos_nofed['geometry'] = edos_nofed['geometry'].simplify(TOL)
    ro = gpd.read_file("Regiones_Operativas.gpkg").to_crs(4326)
    ro['geometry'] = ro['geometry'].simplify(TOL)
    return entidad, edos_nofed, ro


@st.cache_data
def cargar_raster_hospital(nombre_archivo):
    """Carga raster de distancia hospitalaria — reproyecta si es necesario"""
    ruta = obtener_ruta_archivo(nombre_archivo)
    with rasterio.open(ruta) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:4326', src.width // 4, src.height // 4, *src.bounds)
            data = np.empty((height, width), dtype=float)
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs='EPSG:4326',
                      resampling=Resampling.average)
            from rasterio.transform import array_bounds
            left_b, bottom_b, right_b, top_b = array_bounds(height, width, transform)
            bounds = (bottom_b, left_b, top_b, right_b)
        else:
            scale = 4
            data = src.read(1, out_shape=(src.height // scale, src.width // scale),
                            resampling=Resampling.average).astype(float)
            bounds = (src.bounds.bottom, src.bounds.left, src.bounds.top, src.bounds.right)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
    data[data == -9999] = np.nan
    data[data >= 1e+30] = np.nan
    data[data <= 0] = np.nan
    # Convertir segundos a minutos si los valores son muy grandes
    if np.nanmax(data[~np.isnan(data)]) > 1440:  # más de 24 hrs en minutos = probablemente segundos
        data = data / 60.0
    return data, bounds


@st.cache_data
def cargar_municipios():
    """Carga MUNICIPAL_CARACTERISTICAS — geometria + atributos sociodemograficos"""
    gdf = gpd.read_file("MUNICIPAL_CARACTERISTICAS.gpkg").to_crs(4326)
    gdf['geometry'] = gdf['geometry'].simplify(0.005)
    # Normalizar columna clave de entidad para filtro
    if 'CVEGEO' in gdf.columns:
        gdf['CVE_ENT'] = gdf['CVEGEO'].astype(str).str[:2].str.zfill(2)
    elif 'clmun' in gdf.columns:
        gdf['CVE_ENT'] = gdf['clmun'].astype(str).str.zfill(4).str[:2]
    return gdf


@st.cache_data
def crear_imagen_raster(data, rangos, colores):
    h, w = data.shape
    colored = np.zeros((h, w, 4), dtype=np.uint8)
    for i, (vmin, vmax) in enumerate(rangos):
        mask = (data >= vmin) & (data < vmax)
        hx = colores[i].lstrip('#')
        r, g, b = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
        colored[mask] = [r, g, b, 210]
    colored[np.isnan(data)] = [0, 0, 0, 0]
    img = Image.fromarray(colored, mode='RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def calcular_estadisticas(data):
    valid = data[~np.isnan(data)]
    rangos_stats = [('< 30 min',0,30,'🟢'),('30 min - 1 hr',30,60,'🟡'),
                    ('1 - 2 hrs',60,120,'🟠'),('2 - 7.5 hrs',120,450,'🔴'),('> 7.5 hrs',450,99999,'⚫')]
    dist = []
    for label, vmin, vmax, emoji in rangos_stats:
        count = int(np.sum((valid >= vmin) & (valid < vmax)))
        dist.append({'Rango':label,'Emoji':emoji,'Pixeles':count,'Porcentaje':count/len(valid)*100})
    return {
        'media': np.mean(valid), 'mediana': np.median(valid),
        'pct_30min': np.sum(valid < 30)/len(valid)*100,
        'pct_1hr':   np.sum(valid < 60)/len(valid)*100,
        'pct_2hr':   np.sum(valid < 120)/len(valid)*100,
        'distribucion': dist
    }


# ============================================
# FUNCIONES DE MAPA
# ============================================
def agregar_puntos_pna(mapa, gdf_fil, usar_cluster):
    if usar_cluster:
        capa = MarkerCluster(name="📍 Unidades PNA (agrupadas)")
        capa.add_to(mapa)
    else:
        capa = folium.FeatureGroup(name="📍 Unidades PNA", show=True)
        capa.add_to(mapa)
    for _, row in gdf_fil.iterrows():
        cfg = COLORES_CAT.get(row['categoria_gerencial'], {"color":"#95a5a6","radio":4})
        popup_html = (f'<div style="font-family:Arial;min-width:220px;">'
                      f'<b>{row["nombre_de_la_unidad"]}</b><br><hr style="margin:4px 0;">'
                      f'<span style="color:#555;">📍 {row["municipio"]}, {row["entidad"]}</span><br>'
                      f'<span style="color:#555;">🏷️ {row["nombre_de_tipologia"]}</span><br>'
                      f'<span style="color:#555;">🔑 CLUES: {row["clues_imb"]}</span></div>')
        folium.CircleMarker(
            location=[row['latitud'], row['longitud']],
            radius=cfg['radio'], color=cfg['color'],
            fill=True, fill_color=cfg['color'], fill_opacity=0.75, weight=1,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=row['nombre_de_la_unidad'],
        ).add_to(capa)


def agregar_capa_estados(mapa, gdf, edos_nofed_nombres):
    capa = folium.FeatureGroup(name="🗺️ Estados", show=True)
    nofed_set = set(edos_nofed_nombres)
    for _, row in gdf.iterrows():
        es_nofed   = row['NOMGEO'] in nofed_set
        color_fill = '#c0392b' if es_nofed else '#2980b9'
        opacity    = 0.25     if es_nofed else 0.10
        folium.GeoJson(
            row['geometry'].__geo_interface__,
            style_function=lambda f, cf=color_fill, op=opacity: {
                'fillColor': cf, 'color': '#555555', 'weight': 1, 'fillOpacity': op},
            tooltip=f"{row['NOMGEO']} {'(no IMSS-B)' if es_nofed else ''}",
        ).add_to(capa)
    capa.add_to(mapa)


def agregar_capa_ro(mapa, gdf):
    capa = folium.FeatureGroup(name="🔶 Regiones Operativas", show=False)
    for _, row in gdf.iterrows():
        popup_html = (f'<div style="font-family:Arial;">'
                      f'<b>{row["nueva_regionalizacion"]}</b><br>'
                      f'<span style="color:#555;">CVE_RO: {row["CVE_RO"]}</span></div>')
        folium.GeoJson(
            row['geometry'].__geo_interface__,
            style_function=lambda f: {
                'fillColor': '#8e44ad', 'color': '#6c3483', 'weight': 1.5, 'fillOpacity': 0.08},
            tooltip=row['nueva_regionalizacion'],
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(capa)
    capa.add_to(mapa)


def agregar_capa_municipios(mapa, gdf_mun, estado_filtro, cve_ent, variable_color):
    """Municipios del estado seleccionado, coloreados por la variable elegida"""
    gdf_fil = gdf_mun[gdf_mun['CVE_ENT'] == str(cve_ent).zfill(2)].copy()
    if gdf_fil.empty:
        return

    nombre_capa = f"🏘️ Municipios — {estado_filtro} ({variable_color})"
    capa = folium.FeatureGroup(name=nombre_capa, show=True)

    # Determinar color por fila segun variable elegida
    def get_color(row):
        if variable_color == "gm":
            return COLORES_GM.get(str(row.get('gm', 'ND')).strip(), '#aaaaaa')
        elif variable_color == "PJS":
            # NPJS = no PJS, PJS = si tiene
            return '#e74c3c' if str(row.get('PJS','')).strip() == 'PJS' else '#3498db'
        else:
            return '#8e44ad'

    for _, row in gdf_fil.iterrows():
        color = get_color(row)

        # Popup con todos los atributos clave
        gm_val     = row.get('gm', 'N/D')
        pjs_val    = row.get('PJS', 'N/D')
        pc_val     = row.get('pc_pb3_', 0)
        pob_val    = row.get('p_hl3ms', 0)
        reg_val    = row.get('Reg_Len', 'N/D')
        mun_val    = row.get('municip', row.get('NOMGEO', 'N/D'))

        popup_html = (
            f'<div style="font-family:Arial;min-width:230px;">'
            f'<b style="font-size:13px;">{mun_val}</b><br>'
            f'<hr style="margin:4px 0;">'
            f'<span style="color:#555;">🗺️ {estado_filtro}</span><br>'
            f'<span style="color:#555;">📊 Marginacion: <b>{gm_val}</b></span><br>'
            f'<span style="color:#555;">🏛️ Jurisdiccion: <b>{pjs_val}</b></span><br>'
            f'<span style="color:#555;">👥 Pob. SSS: <b>{int(pob_val):,}</b></span><br>'
            f'<span style="color:#555;">📈 % Pob. SSS: <b>{pc_val:.1%}</b></span><br>'
            f'<span style="color:#555;">📝 {reg_val}</span>'
            f'</div>'
        )

        folium.GeoJson(
            row['geometry'].__geo_interface__,
            style_function=lambda f, c=color: {
                'fillColor': c, 'color': '#333333', 'weight': 0.6, 'fillOpacity': 0.55},
            tooltip=f"{mun_val} | {gm_val}",
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(capa)

    capa.add_to(mapa)


# ============================================
# CARGAR DATOS — solo capas ligeras al inicio
# Los rasters se cargan solo si el usuario los activa
# ============================================
gdf_pna = cargar_pna()
gdf_entidad, gdf_edos_nofed, gdf_ro = cargar_capas_poligonos()
edos_nofed_nombres = gdf_edos_nofed['NOMGEO'].tolist()

# Valores por defecto para métricas (se actualizan si se carga isocrona)
_stats_default = {
    'media': 0, 'mediana': 0, 'pct_30min': 0, 'pct_1hr': 0, 'pct_2hr': 0,
    'distribucion': [{'Rango': r, 'Emoji': e, 'Pixeles': 0, 'Porcentaje': 0}
                     for r, e in [('< 30 min','🟢'),('30-60 min','🟡'),
                                  ('1-2 hrs','🟠'),('2-7.5 hrs','🔴'),('> 7.5 hrs','⚫')]]
}

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## 🏥 Panel de Control")
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # --- RASTER ISOCRONAS ---
    st.markdown("### ⏱️ Isocronas de accesibilidad")
    mostrar_iso = st.toggle("Mostrar isocronas", value=True)
    if mostrar_iso:
        opacity_iso = st.slider("Opacidad isocronas", 0.1, 1.0, 0.8, 0.1)
        esquema_iso = st.selectbox("🎨 Esquema", list(ESQUEMAS_ISOCRONA.keys()))
        colores_iso = ESQUEMAS_ISOCRONA[esquema_iso]['colores']
        labels_iso  = ESQUEMAS_ISOCRONA[esquema_iso]['labels']
    else:
        opacity_iso = 0.0
        colores_iso = ESQUEMAS_ISOCRONA["Azules"]['colores']
        labels_iso  = ESQUEMAS_ISOCRONA["Azules"]['labels']

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # --- RASTER HEATMAP SSS ---
    st.markdown("### 👥 Población SSS (HeatMap)")
    mostrar_sss = st.toggle("Mostrar HeatMap SSS", value=False)
    if mostrar_sss:
        opacity_sss = st.slider("Opacidad HeatMap SSS", 0.1, 1.0, 0.75, 0.1)
        esquema_sss = st.selectbox("🎨 Esquema SSS", list(ESQUEMAS_SSS.keys()))
        colores_sss = ESQUEMAS_SSS[esquema_sss]['colores']
        labels_sss  = ESQUEMAS_SSS[esquema_sss]['labels']

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # --- RASTER HOSPITALES TODOS (Distanc_SNA) ---
    st.markdown("### 🏥 Todos los hospitales (Distanc_SNA)")
    mostrar_sna = st.toggle("Mostrar distancia hospitales", value=False)
    if mostrar_sna:
        opacity_sna  = st.slider("Opacidad hospitales", 0.1, 1.0, 0.75, 0.1, key="op_sna")
        esquema_sna  = st.selectbox("🎨 Esquema", list({"Semaforo": ESQUEMA_HOSP_TODOS,
                                                         "Azules": ESQUEMA_HOSP_CAMAS}.keys()),
                                     key="esq_sna")
        colores_sna  = ESQUEMA_HOSP_TODOS['colores'] if esquema_sna == "Semaforo" else ESQUEMA_HOSP_CAMAS['colores']
        labels_sna   = ESQUEMA_HOSP_TODOS['labels']

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # --- RASTER HOSPITALES NO ESPECIALIZADOS (DA_CAMAS) ---
    st.markdown("### 🛏️ Hospitales no especializados (DA_CAMAS)")
    mostrar_camas = st.toggle("Mostrar distancia H. no espec.", value=False)
    if mostrar_camas:
        opacity_camas = st.slider("Opacidad H. no espec.", 0.1, 1.0, 0.75, 0.1, key="op_camas")
        esquema_camas = st.selectbox("🎨 Esquema", list({"Semaforo": ESQUEMA_HOSP_TODOS,
                                                          "Azules": ESQUEMA_HOSP_CAMAS}.keys()),
                                      key="esq_camas")
        colores_camas = ESQUEMA_HOSP_TODOS['colores'] if esquema_camas == "Semaforo" else ESQUEMA_HOSP_CAMAS['colores']
        labels_camas  = ESQUEMA_HOSP_CAMAS['labels']

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # --- CAPAS POLIGONALES ---
    st.markdown("### 🗂️ Capas poligonales")
    mostrar_estados = st.toggle("🗺️ Limites de estados", value=True)
    mostrar_ro      = st.toggle("🔶 Regiones Operativas", value=False)

    st.markdown("**🏘️ Municipios**")
    dict_estados = dict(zip(gdf_entidad['NOMGEO'], gdf_entidad['CVE_ENT'].astype(str).str.zfill(2)))
    estado_mun   = st.selectbox("Mostrar municipios de...",
                                 options=["(ninguno)"] + sorted(dict_estados.keys()), index=0)
    mostrar_municipios = estado_mun != "(ninguno)"
    cve_ent_sel = dict_estados.get(estado_mun, None)
    if mostrar_municipios:
        variable_mun = st.selectbox(
            "🎨 Colorear por",
            options=["gm", "PJS"],
            format_func=lambda x: {
                "gm":  "Grado de marginacion",
                "PJS": "Jurisdiccion (PJS / NPJS)",
            }[x]
        )
    else:
        variable_mun = "gm"

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # --- PUNTOS PNA ---
    st.markdown("### 📍 Unidades PNA")
    mostrar_pna = st.toggle("Mostrar unidades en mapa", value=True)
    if mostrar_pna:
        usar_cluster  = st.toggle("Agrupar puntos (cluster)", value=True)
        entidades_sel = st.multiselect("🔍 Filtrar por estado",
                                       options=sorted(gdf_pna['entidad'].unique()),
                                       default=[], placeholder="Todos los estados...")
        cats_sel = st.multiselect("🏷️ Categoria gerencial",
                                   options=sorted(gdf_pna['categoria_gerencial'].unique()),
                                   default=sorted(gdf_pna['categoria_gerencial'].unique()))
        gdf_filtrado = gdf_pna.copy()
        if entidades_sel:
            gdf_filtrado = gdf_filtrado[gdf_filtrado['entidad'].isin(entidades_sel)]
        if cats_sel:
            gdf_filtrado = gdf_filtrado[gdf_filtrado['categoria_gerencial'].isin(cats_sel)]
        n_unidades = len(gdf_filtrado)
    else:
        gdf_filtrado = pd.DataFrame()
        n_unidades   = 0
        entidades_sel = []

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # --- MAPA BASE ---
    st.markdown("### 🗺️ Mapa base")
    basemap = st.selectbox("Selecciona mapa base", ["CartoDB positron","OpenStreetMap","CartoDB dark_matter"], label_visibility="collapsed")

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # --- LEYENDAS ---
    if mostrar_iso:
        st.markdown("### 🎨 Leyenda — Isocronas")
        for color, label in zip(colores_iso, labels_iso):
            st.markdown(f'<div class="legend-item"><span class="legend-color" style="background-color:{color};"></span>'
                        f'<span class="legend-label">{label}</span></div>', unsafe_allow_html=True)

    if mostrar_sss:
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        st.markdown("### 🎨 Leyenda — Población SSS")
        for color, label in zip(colores_sss, labels_sss):
            st.markdown(f'<div class="legend-item"><span class="legend-color" style="background-color:{color};"></span>'
                        f'<span class="legend-label">{label}</span></div>', unsafe_allow_html=True)

    if mostrar_sna:
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        st.markdown("### 🎨 Leyenda — Todos los hospitales")
        for color, label in zip(colores_sna, labels_sna):
            st.markdown(f'<div class="legend-item"><span class="legend-color" style="background-color:{color};"></span>'
                        f'<span class="legend-label">{label}</span></div>', unsafe_allow_html=True)

    if mostrar_camas:
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        st.markdown("### 🎨 Leyenda — H. no especializados")
        for color, label in zip(colores_camas, labels_camas):
            st.markdown(f'<div class="legend-item"><span class="legend-color" style="background-color:{color};"></span>'
                        f'<span class="legend-label">{label}</span></div>', unsafe_allow_html=True)

    if mostrar_estados:
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        st.markdown("### 🗺️ Leyenda — Estados")
        st.markdown('<div class="legend-item"><span class="legend-color" style="background-color:#2980b9;opacity:0.5;"></span>'
                    '<span class="legend-label">Estados IMSS-Bienestar</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="legend-item"><span class="legend-color" style="background-color:#c0392b;opacity:0.5;"></span>'
                    '<span class="legend-label">Estados no IMSS-B</span></div>', unsafe_allow_html=True)

    if mostrar_municipios:
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        st.markdown("### 🏘️ Leyenda — Municipios")
        if variable_mun == "gm":
            for grado, color in COLORES_GM.items():
                st.markdown(
                    f'<div class="legend-item"><span class="legend-color" style="background-color:{color};"></span>'
                    f'<span class="legend-label">{grado}</span></div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="legend-item"><span class="legend-color" style="background-color:#e74c3c;"></span>'
                        '<span class="legend-label">Con PJS</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="legend-item"><span class="legend-color" style="background-color:#3498db;"></span>'
                        '<span class="legend-label">Sin PJS (NPJS)</span></div>', unsafe_allow_html=True)

    if mostrar_pna:
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        st.markdown("### 📍 Leyenda — Unidades PNA")
        for cat, cfg in COLORES_CAT.items():
            st.markdown(f'<div class="legend-item"><span class="pna-dot" style="background-color:{cfg["color"]};"></span>'
                        f'<span class="legend-label">{cfg["label"]}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">'
                '<p><b>⏱️ Isocrona:</b> AC_PN_NUMM (250m)</p>'
                '<p><b>👥 HeatMap:</b> Poblacion SSS</p>'
                '<p><b>🏥 Puntos:</b> PNA IMSS-Bienestar</p>'
                '<p><b>🗺️ Poligonos:</b> EPSG:6372 → 4326</p>'
                '</div>', unsafe_allow_html=True)


# ============================================
# CONTENIDO PRINCIPAL
# ============================================
st.markdown("# 🏥 Isocronas de Accesibilidad a Centros de Salud")
st.markdown("**Tiempo de viaje al centro de salud mas cercano en Mexico — Unidades IMSS-Bienestar**")

# Cargar isocrona solo si está activa
if mostrar_iso:
    with st.spinner("⏳ Descargando isocrona..."):
        data_iso, bounds_iso = cargar_raster_isocrona()
    lat_b, lon_l, lat_t, lon_r = bounds_iso
    stats = calcular_estadisticas(data_iso)
else:
    data_iso, bounds_iso = None, None
    lat_b, lon_l, lat_t, lon_r = 14.5, -119.0, 33.0, -86.0
    stats = _stats_default

# Cargar SSS solo si está activo
if mostrar_sss:
    with st.spinner("⏳ Descargando HeatMap SSS..."):
        data_sss, bounds_sss = cargar_raster_sss()
else:
    data_sss, bounds_sss = None, None

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("⏱️ Tiempo promedio", f"{stats['media']:.0f} min",   f"~{stats['media']/60:.1f} hrs",   delta_color="off")
c2.metric("⏱️ Tiempo mediano",  f"{stats['mediana']:.0f} min", f"~{stats['mediana']/60:.1f} hrs", delta_color="off")
c3.metric("🟢 Acceso < 30 min", f"{stats['pct_30min']:.1f}%",
          "Buena cobertura" if stats['pct_30min']>50 else "Cobertura limitada",
          delta_color="normal" if stats['pct_30min']>50 else "inverse")
c4.metric("🟡 Acceso < 1 hora", f"{stats['pct_1hr']:.1f}%",
          "Buena cobertura" if stats['pct_1hr']>70 else "Cobertura limitada",
          delta_color="normal" if stats['pct_1hr']>70 else "inverse")
c5.metric("📍 Unidades PNA", f"{n_unidades:,}",
          "en mapa" if mostrar_pna else "ocultas", delta_color="off")

st.markdown("---")

# ============================================
# MAPA
# ============================================
m = folium.Map(location=[23.6,-102.5], zoom_start=5, tiles=basemap)

# Raster isocronas
if mostrar_iso and data_iso is not None:
    img_iso = crear_imagen_raster(data_iso, tuple(RANGOS_ISOCRONA), tuple(colores_iso))
    folium.raster_layers.ImageOverlay(
        image=img_iso,
        bounds=[[lat_b-0.2, lon_l],[lat_t-0.2, lon_r]],
        opacity=opacity_iso, name="⏱️ Isocronas de accesibilidad"
    ).add_to(m)

# Raster HeatMap SSS
if mostrar_sss and data_sss is not None:
    b_sss = bounds_sss
    img_sss = crear_imagen_raster(data_sss, tuple(RANGOS_SSS), tuple(colores_sss))
    folium.raster_layers.ImageOverlay(
        image=img_sss,
        bounds=[[b_sss[0], b_sss[1]],[b_sss[2], b_sss[3]]],
        opacity=opacity_sss, name="👥 HeatMap Poblacion SSS"
    ).add_to(m)

# Capas base extra
folium.TileLayer('OpenStreetMap', name='OSM').add_to(m)
folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                 attr='Esri', name='Satelite').add_to(m)

# Poligonos
if mostrar_estados:
    agregar_capa_estados(m, gdf_entidad, edos_nofed_nombres)
if mostrar_ro:
    agregar_capa_ro(m, gdf_ro)
if mostrar_municipios:
    try:
        gdf_mun = cargar_municipios()
        agregar_capa_municipios(m, gdf_mun, estado_mun, cve_ent_sel, variable_mun)
    except FileNotFoundError:
        st.warning("⚠️ MUNICIPAL_CARACTERISTICAS.gpkg no encontrado.")

# Puntos PNA (siempre encima)
if mostrar_pna and len(gdf_filtrado) > 0:
    agregar_puntos_pna(m, gdf_filtrado, usar_cluster)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width="stretch", height=600, key="mapa_principal")

st.markdown("---")

# ============================================
# GRAFICAS Y TABLA
# ============================================
cl, cr = st.columns(2)
with cl:
    st.subheader("📊 Distribucion de accesibilidad")
    df_dist = pd.DataFrame(stats['distribucion'])
    st.bar_chart(df_dist.set_index('Rango')['Porcentaje'], horizontal=True, color=colores_iso[2])
with cr:
    st.subheader("📋 Detalle por rango")
    df_t = df_dist.copy()
    df_t['Porcentaje'] = df_t['Porcentaje'].apply(lambda x: f"{x:.1f}%")
    df_t['Pixeles'] = df_t['Pixeles'].apply(lambda x: f"{x:,}")
    st.dataframe(df_t[['Emoji','Rango','Pixeles','Porcentaje']], hide_index=True, width="stretch")

if mostrar_pna and len(gdf_filtrado) > 0 and entidades_sel:
    st.markdown("---")
    st.subheader(f"📋 Unidades PNA — {', '.join(entidades_sel)}")
    df_p = gdf_filtrado[['nombre_de_la_unidad','municipio','entidad',
                          'categoria_gerencial','nombre_de_tipologia','clues_imb']].copy()
    df_p.columns = ['Unidad','Municipio','Estado','Categoria','Tipologia','CLUES']
    st.dataframe(df_p, hide_index=True, width="stretch")

st.markdown("---")

# ============================================
# INSIGHTS
# ============================================
st.subheader("💡 Hallazgos principales")
i1,i2,i3 = st.columns(3)
i1.markdown(f'<div style="background:rgba(39,174,96,0.1);border-left:4px solid #27ae60;border-radius:4px;padding:15px;">'
            f'<h4 style="color:#27ae60;margin:0;">🟢 Buena accesibilidad</h4>'
            f'<p style="font-size:24px;font-weight:bold;margin:8px 0;">{stats["pct_1hr"]:.1f}%</p>'
            f'<p>del territorio llega en <b>menos de 1 hora</b></p></div>', unsafe_allow_html=True)
i2.markdown(f'<div style="background:rgba(243,156,18,0.1);border-left:4px solid #f39c12;border-radius:4px;padding:15px;">'
            f'<h4 style="color:#f39c12;margin:0;">🟡 Accesibilidad media</h4>'
            f'<p style="font-size:24px;font-weight:bold;margin:8px 0;">{stats["pct_2hr"]-stats["pct_1hr"]:.1f}%</p>'
            f'<p>requiere entre <b>1 y 2 horas</b></p></div>', unsafe_allow_html=True)
i3.markdown(f'<div style="background:rgba(231,76,60,0.1);border-left:4px solid #e74c3c;border-radius:4px;padding:15px;">'
            f'<h4 style="color:#e74c3c;margin:0;">🔴 Baja accesibilidad</h4>'
            f'<p style="font-size:24px;font-weight:bold;margin:8px 0;">{100-stats["pct_2hr"]:.1f}%</p>'
            f'<p>necesita <b>mas de 2 horas</b></p></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:gray;font-size:12px;">'
            '🏥 Dashboard de Accesibilidad | Isocronas AC_PN_NUMM · HeatMap SSS · PNA IMSS-Bienestar'
            '</div>', unsafe_allow_html=True)
