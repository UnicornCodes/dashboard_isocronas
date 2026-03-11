import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
import numpy as np
from PIL import Image
import io
import base64
import pandas as pd
import os
import tempfile
import requests

# ============================================
# CONSTANTES
# ============================================
COLORES_CAT = {
    "1-2 NUCLEOS":         {"color": "#3498db", "radio": 4,  "label": "1-2 Núcleos"},
    "3-5 NUCLEOS":         {"color": "#2ecc71", "radio": 6,  "label": "3-5 Núcleos"},
    "6-12 NUCLEOS":        {"color": "#f39c12", "radio": 8,  "label": "6-12 Núcleos"},
    "SERVICIOS AMPLIADOS": {"color": "#e74c3c", "radio": 10, "label": "Servicios Ampliados"},
}

ESQUEMAS_ISOCRONA = {
    "Azules": {
        'colores': ['#d4e6f1','#a2cce3','#5faed1','#2980b9','#1a5276'],
        'labels':  ['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']
    },
    "Rojos": {
        'colores': ['#f9e4e4','#f1a9a9','#e06666','#cc0000','#800000'],
        'labels':  ['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']
    },
    "Verdes": {
        'colores': ['#d5f5e3','#a9dfbf','#52be80','#27ae60','#1e8449'],
        'labels':  ['< 30 min','30-60 min','1-2 hrs','2-7.5 hrs','> 7.5 hrs']
    },
}

RANGOS_ISOCRONA  = [(0.01,30),(30,60),(60,120),(120,450),(450,50000)]
RANGOS_HOSP      = [(0.01,30),(30,60),(60,99999)]

ESQUEMA_SEMAFORO = {'colores': ['#2ecc71','#f39c12','#e74c3c'], 'labels': ['< 30 min','30-60 min','> 60 min']}
ESQUEMA_AZULES   = {'colores': ['#5dade2','#2e86c1','#1a5276'], 'labels': ['< 30 min','30-60 min','> 60 min']}

RANGOS_SSS  = [(1,100),(100,500),(500,3000),(3000,15000),(15000,50000),(50000,300001)]
ESQUEMAS_SSS = {
    "Calor":  {'colores': ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026','#800026'], 'labels': ['1-100','100-500','500-3k','3k-15k','15k-50k','50k-300k']},
    "Verdes": {'colores': ['#006837','#31a354','#78c679','#c2e699','#ffffcc','#ffeda0'], 'labels': ['1-100','100-500','500-3k','3k-15k','15k-50k','50k-300k']},
}

COLORES_GM = {"Muy alto":"#7b0c0c","Alto":"#d62728","Medio":"#ff7f0e","Bajo":"#bcbd22","Muy bajo":"#2ca02c","ND":"#aaaaaa"}

HF_BASE = "https://huggingface.co/datasets/UnicornCodes/dashboard-isocronas/resolve/main"
ARCHIVOS_REMOTOS = {
    "AC_PN_NUMM_4326.tif": f"{HF_BASE}/AC_PN_NUMM_4326.tif",
    "HeatMap_sss.tif":     f"{HF_BASE}/HeatMap_sss.tif",
    "Distanc_SNA.tif":     f"{HF_BASE}/Distanc_SNA.tif",
    "DA_CAMAS.tif":        f"{HF_BASE}/DA_CAMAS.tif",
}

# ============================================
# FUNCIONES DE CARGA
# ============================================
_cache = {}

def obtener_ruta_archivo(nombre):
    if os.path.exists(nombre):
        return nombre
    url = ARCHIVOS_REMOTOS.get(nombre)
    if not url:
        raise FileNotFoundError(f"'{nombre}' no encontrado.")
    tmp_path = os.path.join(tempfile.gettempdir(), nombre)
    if not os.path.exists(tmp_path):
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        part = tmp_path + ".part"
        with open(part, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8*1024*1024):
                if chunk: f.write(chunk)
        os.rename(part, tmp_path)
    return tmp_path


def cargar_raster_isocrona():
    if 'iso' in _cache: return _cache['iso']
    ruta = obtener_ruta_archivo("AC_PN_NUMM_4326.tif")
    with rasterio.open(ruta) as src:
        data = src.read(1, out_shape=(src.height//4, src.width//4),
                        resampling=rasterio.enums.Resampling.average).astype(float)
        bounds = (src.bounds.bottom, src.bounds.left, src.bounds.top, src.bounds.right)
        if src.nodata: data[data == src.nodata] = np.nan
    data[data==-9999]=np.nan; data[data>=1e+30]=np.nan; data[data==0]=np.nan
    _cache['iso'] = (data, bounds)
    return data, bounds


def cargar_raster_sss():
    if 'sss' in _cache: return _cache['sss']
    ruta = obtener_ruta_archivo("HeatMap_sss.tif")
    with rasterio.open(ruta) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:4326', src.width//4, src.height//4, *src.bounds)
            data = np.empty((height, width), dtype=float)
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs='EPSG:4326',
                      resampling=Resampling.average)
            left_b, bottom_b, right_b, top_b = array_bounds(height, width, transform)
            bounds = (bottom_b, left_b, top_b, right_b)
        else:
            data = src.read(1, out_shape=(src.height//4, src.width//4),
                            resampling=rasterio.enums.Resampling.average).astype(float)
            bounds = (src.bounds.bottom, src.bounds.left, src.bounds.top, src.bounds.right)
    data[data==-9999]=np.nan; data[data>=1e+30]=np.nan; data[data<=0]=np.nan
    _cache['sss'] = (data, bounds)
    return data, bounds


def cargar_raster_hospital(nombre):
    if nombre in _cache: return _cache[nombre]
    ruta = obtener_ruta_archivo(nombre)
    with rasterio.open(ruta) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:4326', src.width//4, src.height//4, *src.bounds)
            data = np.empty((height, width), dtype=float)
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs='EPSG:4326',
                      resampling=Resampling.average)
            left_b, bottom_b, right_b, top_b = array_bounds(height, width, transform)
            bounds = (bottom_b, left_b, top_b, right_b)
        else:
            data = src.read(1, out_shape=(src.height//4, src.width//4),
                            resampling=Resampling.average).astype(float)
            bounds = (src.bounds.bottom, src.bounds.left, src.bounds.top, src.bounds.right)
            if src.nodata: data[data == src.nodata] = np.nan
    data[data==-9999]=np.nan; data[data>=1e+30]=np.nan; data[data<=0]=np.nan
    if np.nanmax(data[~np.isnan(data)]) > 1440:
        data = data / 60.0
    _cache[nombre] = (data, bounds)
    return data, bounds


def cargar_pna():
    if 'pna' in _cache: return _cache['pna']
    gdf = gpd.read_file("PNA-IMSSB.gpkg")
    gdf = gdf[['clues_imb','nombre_de_la_unidad','categoria_gerencial',
               'nombre_de_tipologia','entidad','municipio','latitud','longitud']]
    _cache['pna'] = gdf
    return gdf


def cargar_capas_poligonos():
    if 'polys' in _cache: return _cache['polys']
    TOL = 0.01
    entidad    = gpd.read_file("Entidad.gpkg").to_crs(4326)
    entidad['geometry'] = entidad['geometry'].simplify(TOL)
    edos_nofed = gpd.read_file("EDOS_NOFED.gpkg").to_crs(4326)
    edos_nofed['geometry'] = edos_nofed['geometry'].simplify(TOL)
    ro         = gpd.read_file("Regiones_Operativas.gpkg").to_crs(4326)
    ro['geometry'] = ro['geometry'].simplify(TOL)
    _cache['polys'] = (entidad, edos_nofed, ro)
    return entidad, edos_nofed, ro


def cargar_municipios():
    if 'mun' in _cache: return _cache['mun']
    gdf = gpd.read_file("MUNICIPAL_CARACTERISTICAS.gpkg").to_crs(4326)
    gdf['geometry'] = gdf['geometry'].simplify(0.005)
    if 'CVEGEO' in gdf.columns:
        gdf['CVE_ENT'] = gdf['CVEGEO'].astype(str).str[:2].str.zfill(2)
    elif 'clmun' in gdf.columns:
        gdf['CVE_ENT'] = gdf['clmun'].astype(str).str.zfill(4).str[:2]
    _cache['mun'] = gdf
    return gdf


def cargar_agebs():
    """Carga anexo1 con tiempos a los 3 rasters y población por AGEB"""
    if 'agebs' in _cache: return _cache['agebs']
    gdf = gpd.read_file("anexo1_desde_excel_tiempos.gpkg")
    _cache['agebs'] = gdf
    return gdf


def tabla_pob_por_categoria(gdf, col_cat, col_pob, titulo, color_accent):
    """Genera tabla HTML de población total por categoría de tiempo"""
    COLORES_CAT_TIEMPO = {
        '0 a 30 min':   '#2ecc71',
        '30.1 a 60':    '#f39c12',
        '60.1 a 120':   '#e67e22',
        '>120 min':     '#e74c3c',
    }
    if col_cat not in gdf.columns or col_pob not in gdf.columns:
        return html.P(f"Columna {col_cat} o {col_pob} no encontrada.",
                      style={"color":"#555","fontSize":"11px","fontFamily":"DM Mono"})

    resumen = (gdf.groupby(col_cat)[col_pob]
               .sum().reset_index()
               .rename(columns={col_cat:'Categoría', col_pob:'Población'}))
    total   = resumen['Población'].sum()
    resumen['%'] = resumen['Población'] / total * 100 if total > 0 else 0

    # Orden fijo de categorías
    orden = ['0 a 30 min','30.1 a 60','60.1 a 120','>120 min']
    resumen['_ord'] = resumen['Categoría'].map({c:i for i,c in enumerate(orden)}).fillna(99)
    resumen = resumen.sort_values('_ord').drop(columns='_ord')

    filas = []
    for _, row in resumen.iterrows():
        c = COLORES_CAT_TIEMPO.get(str(row['Categoría']).strip(), '#aaa')
        filas.append(html.Tr([
            html.Td([
                html.Span(style={"display":"inline-block","width":"8px","height":"8px",
                                 "borderRadius":"50%","backgroundColor":c,
                                 "marginRight":"6px","flexShrink":"0"}),
                str(row['Categoría'])
            ], style={"fontSize":"11px","color":"#c8d0e7","padding":"5px 8px",
                      "fontFamily":"DM Mono","display":"flex","alignItems":"center"}),
            html.Td(f"{int(row['Población']):,}",
                    style={"fontSize":"11px","color":"#9ba8c0","padding":"5px 8px",
                           "textAlign":"right","fontFamily":"DM Mono"}),
            html.Td(f"{row['%']:.1f}%",
                    style={"fontSize":"11px","color":c,"padding":"5px 8px",
                           "textAlign":"right","fontFamily":"DM Mono","fontWeight":"600"}),
        ], style={"borderBottom":"1px solid #1a2035"}))

    return html.Div([
        html.P(titulo, style={"color": color_accent,"fontSize":"12px","fontWeight":"600",
                              "fontFamily":"Syne","marginBottom":"6px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Categoría",  style={"color":"#3d4f6e","fontSize":"10px","padding":"4px 8px","fontFamily":"DM Mono","fontWeight":"500","borderBottom":"1px solid #1e2438"}),
                html.Th("Población",  style={"color":"#3d4f6e","fontSize":"10px","padding":"4px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"500","borderBottom":"1px solid #1e2438"}),
                html.Th("%",          style={"color":"#3d4f6e","fontSize":"10px","padding":"4px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"500","borderBottom":"1px solid #1e2438"}),
            ])),
            html.Tbody(filas),
            html.Tfoot(html.Tr([
                html.Td("Total", style={"fontSize":"11px","color":"#e8eaf6","padding":"5px 8px","fontFamily":"DM Mono","fontWeight":"600","borderTop":"1px solid #2d3555"}),
                html.Td(f"{int(total):,}", style={"fontSize":"11px","color":"#e8eaf6","padding":"5px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"600","borderTop":"1px solid #2d3555"}),
                html.Td("100%", style={"fontSize":"11px","color":"#e8eaf6","padding":"5px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"600","borderTop":"1px solid #2d3555"}),
            ])),
        ], style={"width":"100%","borderCollapse":"collapse",
                  "backgroundColor":"#0f1420","borderRadius":"6px","overflow":"hidden"}),
    ], style={"backgroundColor":"#0f1420","borderRadius":"8px","padding":"12px",
              "border":"1px solid #1e2438"})


def crear_imagen_raster(data, rangos, colores):
    h, w = data.shape
    colored = np.zeros((h, w, 4), dtype=np.uint8)
    for i, (vmin, vmax) in enumerate(rangos):
        mask = (data >= vmin) & (data < vmax)
        hx = colores[i].lstrip('#')
        r, g, b = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
        colored[mask] = [r, g, b, 210]
    colored[np.isnan(data)] = [0,0,0,0]
    img = Image.fromarray(colored, mode='RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def calcular_estadisticas(data):
    valid = data[~np.isnan(data)]
    rangos_labels = [
        ('< 30 min',    0,   30,   '🟢'),
        ('30 min-1 hr', 30,  60,   '🟡'),
        ('1 - 2 hrs',   60,  120,  '🟠'),
        ('2 - 7.5 hrs', 120, 450,  '🔴'),
        ('> 7.5 hrs',   450, 99999,'⚫'),
    ]
    dist = []
    for label, vmin, vmax, emoji in rangos_labels:
        count = int(np.sum((valid >= vmin) & (valid < vmax)))
        dist.append({'Rango': label, 'Emoji': emoji,
                     'Pixeles': count, 'Porcentaje': count/len(valid)*100})
    return {
        'media':       float(np.mean(valid)),
        'mediana':     float(np.median(valid)),
        'pct_30':      float(np.sum(valid<30)/len(valid)*100),
        'pct_60':      float(np.sum(valid<60)/len(valid)*100),
        'pct_120':     float(np.sum(valid<120)/len(valid)*100),
        'distribucion': dist,
    }


def construir_mapa(capas):
    basemap  = capas.get('basemap', 'CartoDB positron')
    m = folium.Map(location=[23.6, -102.5], zoom_start=5, tiles=basemap)

    # Isocronas
    if capas.get('iso_on') and capas.get('iso_data') is not None:
        data, bounds = capas['iso_data']
        esquema = ESQUEMAS_ISOCRONA.get(capas.get('iso_esquema','Azules'))
        img = crear_imagen_raster(data, RANGOS_ISOCRONA, esquema['colores'])
        lat_b, lon_l, lat_t, lon_r = bounds
        folium.raster_layers.ImageOverlay(
            image=img, bounds=[[lat_b-0.2,lon_l],[lat_t-0.2,lon_r]],
            opacity=capas.get('iso_opacity',0.8), name="⏱️ Isocronas"
        ).add_to(m)

    # HeatMap SSS
    if capas.get('sss_on') and capas.get('sss_data') is not None:
        data, bounds = capas['sss_data']
        esquema = ESQUEMAS_SSS.get(capas.get('sss_esquema','Calor'))
        img = crear_imagen_raster(data, RANGOS_SSS, esquema['colores'])
        folium.raster_layers.ImageOverlay(
            image=img, bounds=[[bounds[0],bounds[1]],[bounds[2],bounds[3]]],
            opacity=capas.get('sss_opacity',0.75), name="👥 HeatMap SSS"
        ).add_to(m)

    # Distanc_SNA
    if capas.get('sna_on') and capas.get('sna_data') is not None:
        data, bounds = capas['sna_data']
        esq_key = capas.get('sna_esquema','Semaforo')
        esquema = ESQUEMA_SEMAFORO if esq_key == 'Semaforo' else ESQUEMA_AZULES
        img = crear_imagen_raster(data, RANGOS_HOSP, esquema['colores'])
        folium.raster_layers.ImageOverlay(
            image=img, bounds=[[bounds[0],bounds[1]],[bounds[2],bounds[3]]],
            opacity=capas.get('sna_opacity',0.75), name="🏥 Todos los hospitales"
        ).add_to(m)

    # DA_CAMAS
    if capas.get('camas_on') and capas.get('camas_data') is not None:
        data, bounds = capas['camas_data']
        esq_key = capas.get('camas_esquema','Semaforo')
        esquema = ESQUEMA_SEMAFORO if esq_key == 'Semaforo' else ESQUEMA_AZULES
        img = crear_imagen_raster(data, RANGOS_HOSP, esquema['colores'])
        folium.raster_layers.ImageOverlay(
            image=img, bounds=[[bounds[0],bounds[1]],[bounds[2],bounds[3]]],
            opacity=capas.get('camas_opacity',0.75), name="🛏️ H. no especializados"
        ).add_to(m)

    # Estados
    if capas.get('estados_on'):
        entidad, edos_nofed, _ = cargar_capas_poligonos()
        nofed_set = set(edos_nofed['NOMGEO'].tolist())
        capa_est = folium.FeatureGroup(name="🗺️ Estados", show=True)
        for _, row in entidad.iterrows():
            es_nofed = row['NOMGEO'] in nofed_set
            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda f, c='#c0392b' if es_nofed else '#2980b9',
                    op=0.25 if es_nofed else 0.10: {
                        'fillColor':c,'color':'#555','weight':1,'fillOpacity':op},
                tooltip=row['NOMGEO']
            ).add_to(capa_est)
        capa_est.add_to(m)

    # Regiones Operativas
    if capas.get('ro_on'):
        _, _, ro = cargar_capas_poligonos()
        capa_ro = folium.FeatureGroup(name="🔶 Regiones Operativas", show=True)
        for _, row in ro.iterrows():
            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda f: {'fillColor':'#8e44ad','color':'#6c3483','weight':1.5,'fillOpacity':0.08},
                tooltip=row.get('nueva_regionalizacion','')
            ).add_to(capa_ro)
        capa_ro.add_to(m)

    # Municipios
    estado_mun = capas.get('estado_mun')
    if estado_mun and estado_mun != '(ninguno)':
        try:
            gdf_mun = cargar_municipios()
            entidad_gdf, _, _ = cargar_capas_poligonos()
            dict_estados = dict(zip(entidad_gdf['NOMGEO'], entidad_gdf['CVE_ENT'].astype(str).str.zfill(2)))
            cve = dict_estados.get(estado_mun)
            if cve:
                var = capas.get('var_mun','gm')
                gdf_fil = gdf_mun[gdf_mun['CVE_ENT']==cve]
                capa_mun = folium.FeatureGroup(name=f"🏘️ Municipios — {estado_mun}", show=True)
                for _, row in gdf_fil.iterrows():
                    if var == 'gm':
                        color = COLORES_GM.get(str(row.get('gm','ND')).strip(),'#aaaaaa')
                    else:
                        color = '#e74c3c' if str(row.get('PJS','')).strip()=='PJS' else '#3498db'
                    folium.GeoJson(
                        row['geometry'].__geo_interface__,
                        style_function=lambda f, c=color: {'fillColor':c,'color':'#333','weight':0.6,'fillOpacity':0.55},
                        tooltip=f"{row.get('municip', row.get('NOMGEO',''))} | {row.get('gm','')}"
                    ).add_to(capa_mun)
                capa_mun.add_to(m)
        except Exception as e:
            pass

    # PNA
    if capas.get('pna_on'):
        gdf_pna = cargar_pna()
        filtro_ent = capas.get('filtro_ent', [])
        filtro_cat = capas.get('filtro_cat', [])
        gdf_fil = gdf_pna.copy()
        if filtro_ent: gdf_fil = gdf_fil[gdf_fil['entidad'].isin(filtro_ent)]
        if filtro_cat: gdf_fil = gdf_fil[gdf_fil['categoria_gerencial'].isin(filtro_cat)]
        usar_cluster = capas.get('cluster', True)
        if usar_cluster:
            capa = MarkerCluster(name="📍 Unidades PNA")
        else:
            capa = folium.FeatureGroup(name="📍 Unidades PNA", show=True)
        capa.add_to(m)
        for _, row in gdf_fil.iterrows():
            cfg = COLORES_CAT.get(row['categoria_gerencial'], {"color":"#95a5a6","radio":4})
            popup_html = (f'<div style="font-family:Arial;min-width:200px;">'
                          f'<b>{row["nombre_de_la_unidad"]}</b><br>'
                          f'<span style="color:#555;">📍 {row["municipio"]}, {row["entidad"]}</span><br>'
                          f'<span style="color:#555;">🔑 {row["clues_imb"]}</span></div>')
            folium.CircleMarker(
                location=[row['latitud'], row['longitud']],
                radius=cfg['radio'], color=cfg['color'],
                fill=True, fill_color=cfg['color'], fill_opacity=0.75, weight=1,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=row['nombre_de_la_unidad'],
            ).add_to(capa)

    folium.TileLayer('OpenStreetMap', name='OSM').add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


# ============================================
# CARGAR DATOS LIGEROS AL INICIO
# ============================================
print("Cargando capas vectoriales...")
gdf_pna_global        = cargar_pna()
gdf_entidad_g, _, _   = cargar_capas_poligonos()
estados_lista         = ['(ninguno)'] + sorted(gdf_entidad_g['NOMGEO'].tolist())
cats_lista            = sorted(gdf_pna_global['categoria_gerencial'].unique().tolist())
print("Listo.")

# ============================================
# APP DASH
# ============================================
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.CYBORG,
    "https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap"
], suppress_callback_exceptions=True)
app.title = "Accesibilidad Centros de Salud"

# CSS global — fuerza tema oscuro en todos los dropdowns de Dash
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }

        body {
            background-color: #0a0e1a !important;
            font-family: 'DM Mono', monospace;
        }

        h1, h2, h3, h4, h5 {
            font-family: 'Syne', sans-serif !important;
        }

        /* Scrollbar sidebar */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0f1117; }
        ::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 4px; }

        /* ---- Dropdowns Dash oscuros ---- */
        .Select-control,
        .Select--single > .Select-control,
        .Select--multi > .Select-control {
            background-color: #161b2e !important;
            border: 1px solid #2d3555 !important;
            border-radius: 6px !important;
            color: #c8d0e7 !important;
            min-height: 32px !important;
        }

        .Select-value-label,
        .Select-placeholder,
        .Select-input input {
            color: #c8d0e7 !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 11px !important;
        }

        .Select-menu-outer {
            background-color: #161b2e !important;
            border: 1px solid #2d3555 !important;
            border-radius: 6px !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
            z-index: 9999 !important;
        }

        .Select-option {
            background-color: #161b2e !important;
            color: #c8d0e7 !important;
            font-size: 11px !important;
            padding: 8px 12px !important;
            font-family: 'DM Mono', monospace !important;
        }

        .Select-option:hover,
        .Select-option.is-focused {
            background-color: #1e2640 !important;
            color: #7eb8f7 !important;
        }

        .Select-option.is-selected {
            background-color: #1a3a6e !important;
            color: #7eb8f7 !important;
        }

        .Select-arrow { border-color: #555 transparent transparent !important; }
        .Select-clear { color: #555 !important; }

        /* Multi-select tags */
        .Select-value {
            background-color: #1a3a6e !important;
            border: 1px solid #2d5299 !important;
            border-radius: 4px !important;
            color: #7eb8f7 !important;
        }
        .Select-value-icon { border-right: 1px solid #2d5299 !important; color: #7eb8f7 !important; }
        .Select-value-icon:hover { background-color: #e74c3c !important; color: #fff !important; }

        /* Cards métricas */
        .metric-card {
            background: linear-gradient(135deg, #161b2e 0%, #1a2035 100%);
            border: 1px solid #2d3555;
            border-radius: 10px;
            padding: 14px 16px;
            transition: border-color 0.2s;
        }
        .metric-card:hover { border-color: #4a6fa5; }

        /* Insight cards */
        .insight-verde  { background: rgba(39,174,96,0.08);  border-left: 3px solid #27ae60; border-radius: 6px; padding: 14px; }
        .insight-ambar  { background: rgba(243,156,18,0.08); border-left: 3px solid #f39c12; border-radius: 6px; padding: 14px; }
        .insight-rojo   { background: rgba(231,76,60,0.08);  border-left: 3px solid #e74c3c; border-radius: 6px; padding: 14px; }

        /* Switch labels */
        .form-check-label { color: #9ba8c0 !important; font-size: 12px !important; }

        /* Slider rail */
        .rc-slider-rail { background-color: #2a2a3a !important; }
        .rc-slider-track { background-color: #2980b9 !important; }
        .rc-slider-handle { border-color: #2980b9 !important; background-color: #2980b9 !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "280px", "padding": "20px 16px", "overflowY": "auto",
    "backgroundColor": "#0f1117", "borderRight": "1px solid #2a2a3a", "zIndex": 1000,
}
CONTENT_STYLE = {"marginLeft": "290px", "padding": "20px"}

def make_legend_dots(items):
    return [html.Div([
        html.Span(style={"display":"inline-block","width":"14px","height":"14px",
                         "borderRadius":"3px","backgroundColor":c,"marginRight":"8px",
                         "border":"1px solid rgba(255,255,255,0.2)","flexShrink":"0"}),
        html.Span(label, style={"fontSize":"12px","color":"#ccc"})
    ], style={"display":"flex","alignItems":"center","margin":"4px 0"}) for c, label in items]

sidebar = html.Div([
    html.H5("🏥 Panel de Control", style={"color":"#fff","marginBottom":"20px","fontWeight":"700"}),

    # Isocronas
    html.Div([
        html.P("⏱️ Isocronas de accesibilidad", style={"color":"#aaa","fontWeight":"600","fontSize":"13px","marginBottom":"6px"}),
        dbc.Switch(id="iso-toggle", label="Mostrar", value=True, style={"color":"#ccc"}),
        html.Div([
            html.Label("Opacidad", style={"color":"#aaa","fontSize":"11px"}),
            dcc.Slider(id="iso-opacity", min=0.1, max=1.0, step=0.1, value=0.8,
                       marks={0.1:"0.1",1.0:"1"}, tooltip={"always_visible":False}),
            html.Label("Esquema", style={"color":"#aaa","fontSize":"11px","marginTop":"6px"}),
            dcc.Dropdown(id="iso-esquema", options=list(ESQUEMAS_ISOCRONA.keys()),
                         value="Azules", clearable=False,
                         style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
        ], id="iso-controls"),
    ], style={"borderBottom":"1px solid #2a2a3a","paddingBottom":"14px","marginBottom":"14px"}),

    # SSS HeatMap
    html.Div([
        html.P("👥 Población SSS (HeatMap)", style={"color":"#aaa","fontWeight":"600","fontSize":"13px","marginBottom":"6px"}),
        dbc.Switch(id="sss-toggle", label="Mostrar", value=False, style={"color":"#ccc"}),
        html.Div([
            html.Label("Opacidad", style={"color":"#aaa","fontSize":"11px"}),
            dcc.Slider(id="sss-opacity", min=0.1, max=1.0, step=0.1, value=0.75,
                       marks={0.1:"0.1",1.0:"1"}, tooltip={"always_visible":False}),
            html.Label("Esquema", style={"color":"#aaa","fontSize":"11px","marginTop":"6px"}),
            dcc.Dropdown(id="sss-esquema", options=list(ESQUEMAS_SSS.keys()),
                         value="Calor", clearable=False,
                         style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
        ], id="sss-controls"),
    ], style={"borderBottom":"1px solid #2a2a3a","paddingBottom":"14px","marginBottom":"14px"}),

    # Distanc SNA
    html.Div([
        html.P("🏥 Todos los hospitales (SNA)", style={"color":"#aaa","fontWeight":"600","fontSize":"13px","marginBottom":"6px"}),
        dbc.Switch(id="sna-toggle", label="Mostrar", value=False, style={"color":"#ccc"}),
        html.Div([
            html.Label("Opacidad", style={"color":"#aaa","fontSize":"11px"}),
            dcc.Slider(id="sna-opacity", min=0.1, max=1.0, step=0.1, value=0.75,
                       marks={0.1:"0.1",1.0:"1"}, tooltip={"always_visible":False}),
            dcc.Dropdown(id="sna-esquema", options=["Semaforo","Azules"],
                         value="Semaforo", clearable=False,
                         style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
        ], id="sna-controls"),
    ], style={"borderBottom":"1px solid #2a2a3a","paddingBottom":"14px","marginBottom":"14px"}),

    # DA CAMAS
    html.Div([
        html.P("🛏️ H. no especializados (Camas)", style={"color":"#aaa","fontWeight":"600","fontSize":"13px","marginBottom":"6px"}),
        dbc.Switch(id="camas-toggle", label="Mostrar", value=False, style={"color":"#ccc"}),
        html.Div([
            html.Label("Opacidad", style={"color":"#aaa","fontSize":"11px"}),
            dcc.Slider(id="camas-opacity", min=0.1, max=1.0, step=0.1, value=0.75,
                       marks={0.1:"0.1",1.0:"1"}, tooltip={"always_visible":False}),
            dcc.Dropdown(id="camas-esquema", options=["Semaforo","Azules"],
                         value="Semaforo", clearable=False,
                         style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
        ], id="camas-controls"),
    ], style={"borderBottom":"1px solid #2a2a3a","paddingBottom":"14px","marginBottom":"14px"}),

    # Polígonos
    html.Div([
        html.P("🗂️ Capas poligonales", style={"color":"#aaa","fontWeight":"600","fontSize":"13px","marginBottom":"6px"}),
        dbc.Switch(id="estados-toggle", label="Límites de estados", value=True, style={"color":"#ccc"}),
        dbc.Switch(id="ro-toggle", label="Regiones Operativas", value=False, style={"color":"#ccc"}),
        html.Label("Municipios de...", style={"color":"#aaa","fontSize":"11px","marginTop":"8px"}),
        dcc.Dropdown(id="estado-mun", options=estados_lista, value="(ninguno)",
                     clearable=False,
                     style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
        html.Div([
            html.Label("Colorear por", style={"color":"#aaa","fontSize":"11px","marginTop":"6px"}),
            dcc.Dropdown(id="var-mun",
                         options=[{"label":"Grado de marginación","value":"gm"},
                                  {"label":"Jurisdicción PJS/NPJS","value":"PJS"}],
                         value="gm", clearable=False,
                         style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
        ], id="var-mun-div"),
    ], style={"borderBottom":"1px solid #2a2a3a","paddingBottom":"14px","marginBottom":"14px"}),

    # PNA
    html.Div([
        html.P("📍 Unidades PNA", style={"color":"#aaa","fontWeight":"600","fontSize":"13px","marginBottom":"6px"}),
        dbc.Switch(id="pna-toggle", label="Mostrar unidades", value=True, style={"color":"#ccc"}),
        dbc.Switch(id="cluster-toggle", label="Agrupar (cluster)", value=True, style={"color":"#ccc"}),
        html.Label("Filtrar por estado", style={"color":"#aaa","fontSize":"11px","marginTop":"6px"}),
        dcc.Dropdown(id="filtro-ent", options=sorted(gdf_pna_global['entidad'].unique()),
                     multi=True, placeholder="Todos...",
                     style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
        html.Label("Categoría", style={"color":"#aaa","fontSize":"11px","marginTop":"6px"}),
        dcc.Dropdown(id="filtro-cat", options=cats_lista, value=cats_lista,
                     multi=True,
                     style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
    ], style={"borderBottom":"1px solid #2a2a3a","paddingBottom":"14px","marginBottom":"14px"}),

    # Mapa base
    html.Div([
        html.P("🗺️ Mapa base", style={"color":"#aaa","fontWeight":"600","fontSize":"13px","marginBottom":"6px"}),
        dcc.Dropdown(id="basemap",
                     options=["CartoDB positron","OpenStreetMap","CartoDB dark_matter"],
                     value="CartoDB positron", clearable=False,
                     style={"backgroundColor":"#1e1e2e","color":"#fff","fontSize":"12px"}),
    ]),

], style=SIDEBAR_STYLE)

content = html.Div([

    # Header
    html.Div([
        html.H2("Isocronas de Accesibilidad a Centros de Salud",
                style={"color":"#e8eaf6","fontFamily":"Syne, sans-serif","fontWeight":"800",
                       "fontSize":"24px","margin":"0","letterSpacing":"-0.5px"}),
        html.P("Tiempo de viaje al centro de salud más cercano · Unidades IMSS-Bienestar",
               style={"color":"#6b7a99","fontSize":"13px","margin":"4px 0 0 0","fontFamily":"DM Mono, monospace"}),
    ], style={"marginBottom":"20px","paddingBottom":"16px","borderBottom":"1px solid #1e2438"}),

    # Métricas
    dbc.Row([
        dbc.Col(html.Div([
            html.P("⏱️ Tiempo promedio", style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-media", style={"color":"#e8eaf6","margin":"4px 0 0 0","fontFamily":"Syne"}),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.P("⏱️ Tiempo mediano", style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-mediana", style={"color":"#e8eaf6","margin":"4px 0 0 0","fontFamily":"Syne"}),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.P("🟢 Acceso < 30 min", style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-30", style={"color":"#2ecc71","margin":"4px 0 0 0","fontFamily":"Syne"}),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.P("🟡 Acceso < 1 hora", style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-60", style={"color":"#f39c12","margin":"4px 0 0 0","fontFamily":"Syne"}),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.P("📍 Unidades PNA", style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-pna", style={"color":"#7eb8f7","margin":"4px 0 0 0","fontFamily":"Syne"}),
        ], className="metric-card"), width=2),
        dbc.Col(html.Div([
            html.P("🔴 Acceso > 2 hrs", style={"color":"#6b7a99","fontSize":"11px","margin":"0","fontFamily":"DM Mono"}),
            html.H4(id="metric-120", style={"color":"#e74c3c","margin":"4px 0 0 0","fontFamily":"Syne"}),
        ], className="metric-card"), width=2),
    ], className="mb-3 g-2"),

    # Mapa
    dbc.Spinner(
        html.Iframe(id="mapa", style={
            "width":"100%","height":"600px","border":"none",
            "borderRadius":"10px","boxShadow":"0 4px 24px rgba(0,0,0,0.5)"
        }),
        color="primary", spinner_style={"width":"3rem","height":"3rem"}
    ),

    html.Div(id="status-msg", style={"color":"#f39c12","fontSize":"12px","marginTop":"6px","fontFamily":"DM Mono"}),

    # ---- ANÁLISIS ----
    html.Div([
        html.Hr(style={"borderColor":"#1e2438","margin":"28px 0 20px 0"}),
        html.H5("📊 Análisis de distribución", style={
            "color":"#e8eaf6","fontFamily":"Syne, sans-serif","fontWeight":"700",
            "fontSize":"16px","marginBottom":"16px"
        }),

        dbc.Row([
            # Gráfica de barras
            dbc.Col([
                html.P("Distribución por rango de tiempo", style={"color":"#6b7a99","fontSize":"12px","marginBottom":"8px","fontFamily":"DM Mono"}),
                html.Div(id="tabla-dist"),
            ], width=7),

            # Tabla detalle
            dbc.Col([
                html.P("Detalle numérico", style={"color":"#6b7a99","fontSize":"12px","marginBottom":"8px","fontFamily":"DM Mono"}),
                html.Div(id="tabla-detalle"),
            ], width=5),
        ], className="mb-4"),

        # Tablitas de población por categoría
        html.Hr(style={"borderColor":"#1e2438","margin":"8px 0 20px 0"}),
        html.H5("👥 Población por categoría de acceso", style={
            "color":"#e8eaf6","fontFamily":"Syne, sans-serif","fontWeight":"700",
            "fontSize":"16px","marginBottom":"4px"
        }),
        html.P("Totales de población según tiempo de viaje al centro de salud más cercano — fuente: AGEBs",
               style={"color":"#6b7a99","fontSize":"12px","fontFamily":"DM Mono","marginBottom":"16px"}),
        dbc.Row([
            dbc.Col(html.Div(id="tabla-pob-pna"),   width=4),
            dbc.Col(html.Div(id="tabla-pob-camas"),  width=4),
            dbc.Col(html.Div(id="tabla-pob-sna"),    width=4),
        ], className="mb-4 g-3"),

        # Insight cards
        html.Hr(style={"borderColor":"#1e2438","margin":"8px 0 20px 0"}),
        html.H5("💡 Hallazgos principales", style={
            "color":"#e8eaf6","fontFamily":"Syne, sans-serif","fontWeight":"700",
            "fontSize":"16px","marginBottom":"16px"
        }),
        dbc.Row([
            dbc.Col(html.Div(id="insight-verde"), width=4),
            dbc.Col(html.Div(id="insight-ambar"), width=4),
            dbc.Col(html.Div(id="insight-rojo"),  width=4),
        ], className="mb-4"),

    ], id="seccion-analisis"),

    html.Hr(style={"borderColor":"#1e2438","margin":"16px 0"}),
    html.P("🏥 Dashboard de Accesibilidad · IMSS-Bienestar · AC_PN_NUMM · HeatMap SSS · PNA",
           style={"color":"#2d3555","fontSize":"11px","textAlign":"center","fontFamily":"DM Mono"}),

], style=CONTENT_STYLE)

app.layout = html.Div([sidebar, content], style={"backgroundColor":"#0f1117","minHeight":"100vh"})


# ============================================
# CALLBACK PRINCIPAL
# ============================================
@app.callback(
    Output("mapa",           "srcDoc"),
    Output("metric-media",   "children"),
    Output("metric-mediana", "children"),
    Output("metric-30",      "children"),
    Output("metric-60",      "children"),
    Output("metric-pna",     "children"),
    Output("metric-120",     "children"),
    Output("status-msg",     "children"),
    Output("tabla-dist",     "children"),
    Output("tabla-detalle",  "children"),
    Output("insight-verde",    "children"),
    Output("insight-ambar",    "children"),
    Output("insight-rojo",     "children"),
    Output("tabla-pob-pna",    "children"),
    Output("tabla-pob-camas",  "children"),
    Output("tabla-pob-sna",    "children"),
    Input("iso-toggle",     "value"),
    Input("iso-opacity",    "value"),
    Input("iso-esquema",    "value"),
    Input("sss-toggle",     "value"),
    Input("sss-opacity",    "value"),
    Input("sss-esquema",    "value"),
    Input("sna-toggle",     "value"),
    Input("sna-opacity",    "value"),
    Input("sna-esquema",    "value"),
    Input("camas-toggle",   "value"),
    Input("camas-opacity",  "value"),
    Input("camas-esquema",  "value"),
    Input("estados-toggle", "value"),
    Input("ro-toggle",      "value"),
    Input("estado-mun",     "value"),
    Input("var-mun",        "value"),
    Input("pna-toggle",     "value"),
    Input("cluster-toggle", "value"),
    Input("filtro-ent",     "value"),
    Input("filtro-cat",     "value"),
    Input("basemap",        "value"),
)
def actualizar_mapa(
    iso_on, iso_op, iso_esq,
    sss_on, sss_op, sss_esq,
    sna_on, sna_op, sna_esq,
    camas_on, camas_op, camas_esq,
    estados_on, ro_on, estado_mun, var_mun,
    pna_on, cluster, filtro_ent, filtro_cat,
    basemap
):
    status = ""
    capas = {
        'basemap': basemap,
        'iso_on': iso_on, 'iso_opacity': iso_op, 'iso_esquema': iso_esq, 'iso_data': None,
        'sss_on': sss_on, 'sss_opacity': sss_op, 'sss_esquema': sss_esq, 'sss_data': None,
        'sna_on': sna_on, 'sna_opacity': sna_op, 'sna_esquema': sna_esq, 'sna_data': None,
        'camas_on': camas_on, 'camas_opacity': camas_op, 'camas_esquema': camas_esq, 'camas_data': None,
        'estados_on': estados_on, 'ro_on': ro_on,
        'estado_mun': estado_mun, 'var_mun': var_mun,
        'pna_on': pna_on, 'cluster': cluster,
        'filtro_ent': filtro_ent or [], 'filtro_cat': filtro_cat or [],
    }

    stats = {'media':0,'mediana':0,'pct_30':0,'pct_60':0,'pct_120':0,'distribucion':[]}

    try:
        if iso_on:
            data, bounds = cargar_raster_isocrona()
            capas['iso_data'] = (data, bounds)
            stats = calcular_estadisticas(data)
    except Exception as e:
        status += f"⚠️ Isocrona: {e} "

    try:
        if sss_on:
            capas['sss_data'] = cargar_raster_sss()
    except Exception as e:
        status += f"⚠️ SSS: {e} "

    try:
        if sna_on:
            capas['sna_data'] = cargar_raster_hospital("Distanc_SNA.tif")
    except Exception as e:
        status += f"⚠️ SNA: {e} "

    try:
        if camas_on:
            capas['camas_data'] = cargar_raster_hospital("DA_CAMAS.tif")
    except Exception as e:
        status += f"⚠️ Camas: {e} "

    mapa_html = construir_mapa(capas)

    # Contar PNA filtrado
    gdf_p = cargar_pna()
    if filtro_ent: gdf_p = gdf_p[gdf_p['entidad'].isin(filtro_ent)]
    if filtro_cat: gdf_p = gdf_p[gdf_p['categoria_gerencial'].isin(filtro_cat)]
    n_pna = len(gdf_p) if pna_on else 0

    # ---- Componentes de análisis ----
    COLORES_RANGOS = ['#2ecc71','#f39c12','#e67e22','#e74c3c','#7b0c0c']
    EMOJIS         = ['🟢','🟡','🟠','🔴','⚫']

    dist = stats.get('distribucion', [])

    # Barras de distribución
    if dist:
        max_pct = max(d['Porcentaje'] for d in dist) or 1
        barras = html.Div([
            html.Div([
                html.Div(d['Emoji'] + " " + d['Rango'],
                         style={"fontSize":"11px","color":"#9ba8c0","width":"120px",
                                "flexShrink":"0","fontFamily":"DM Mono"}),
                html.Div(style={
                    "height":"18px","borderRadius":"3px","flexGrow":"1",
                    "backgroundColor": COLORES_RANGOS[i],
                    "width": f"{d['Porcentaje']/max_pct*100:.0f}%",
                    "minWidth":"2px","transition":"width 0.4s ease"
                }),
                html.Div(f"{d['Porcentaje']:.1f}%",
                         style={"fontSize":"11px","color":"#9ba8c0","width":"44px",
                                "textAlign":"right","flexShrink":"0","fontFamily":"DM Mono"}),
            ], style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"6px"})
            for i, d in enumerate(dist)
        ])
    else:
        barras = html.P("Activa las isocronas para ver la distribución.",
                        style={"color":"#555","fontSize":"12px","fontFamily":"DM Mono"})

    # Tabla detalle
    if dist:
        tabla = html.Table([
            html.Thead(html.Tr([
                html.Th("Rango",     style={"color":"#6b7a99","fontSize":"11px","padding":"4px 8px","fontFamily":"DM Mono","fontWeight":"500"}),
                html.Th("Píxeles",   style={"color":"#6b7a99","fontSize":"11px","padding":"4px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"500"}),
                html.Th("%",         style={"color":"#6b7a99","fontSize":"11px","padding":"4px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"500"}),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(d['Emoji']+" "+d['Rango'],
                            style={"fontSize":"11px","color":"#c8d0e7","padding":"4px 8px","fontFamily":"DM Mono"}),
                    html.Td(f"{d['Pixeles']:,}",
                            style={"fontSize":"11px","color":"#9ba8c0","padding":"4px 8px","textAlign":"right","fontFamily":"DM Mono"}),
                    html.Td(f"{d['Porcentaje']:.1f}%",
                            style={"fontSize":"11px","color": COLORES_RANGOS[i],"padding":"4px 8px","textAlign":"right","fontFamily":"DM Mono","fontWeight":"600"}),
                ], style={"borderBottom":"1px solid #1e2438"})
                for i, d in enumerate(dist)
            ])
        ], style={"width":"100%","borderCollapse":"collapse"})
    else:
        tabla = html.P("—", style={"color":"#555","fontSize":"12px"})

    # Insight cards
    pct_buena  = stats['pct_60']
    pct_media  = stats.get('pct_120', 0) - stats['pct_60']
    pct_baja   = 100 - stats.get('pct_120', 0)

    ins_verde = html.Div([
        html.H6("🟢 Buena accesibilidad", style={"color":"#27ae60","margin":"0 0 6px 0","fontFamily":"Syne","fontSize":"13px"}),
        html.H3(f"{pct_buena:.1f}%", style={"color":"#2ecc71","margin":"0 0 4px 0","fontFamily":"Syne"}),
        html.P("llega en menos de 1 hora", style={"color":"#9ba8c0","fontSize":"12px","margin":"0","fontFamily":"DM Mono"}),
    ], className="insight-verde")

    ins_ambar = html.Div([
        html.H6("🟡 Accesibilidad media", style={"color":"#d68910","margin":"0 0 6px 0","fontFamily":"Syne","fontSize":"13px"}),
        html.H3(f"{pct_media:.1f}%", style={"color":"#f39c12","margin":"0 0 4px 0","fontFamily":"Syne"}),
        html.P("requiere entre 1 y 2 horas", style={"color":"#9ba8c0","fontSize":"12px","margin":"0","fontFamily":"DM Mono"}),
    ], className="insight-ambar")

    ins_rojo = html.Div([
        html.H6("🔴 Baja accesibilidad", style={"color":"#c0392b","margin":"0 0 6px 0","fontFamily":"Syne","fontSize":"13px"}),
        html.H3(f"{pct_baja:.1f}%", style={"color":"#e74c3c","margin":"0 0 4px 0","fontFamily":"Syne"}),
        html.P("necesita más de 2 horas", style={"color":"#9ba8c0","fontSize":"12px","margin":"0","fontFamily":"DM Mono"}),
    ], className="insight-rojo")

    # ---- Tablitas de población por categoría (AGEBs) ----
    try:
        gdf_ageb = cargar_agebs()
        # Detectar columna de población automáticamente
        col_pob_candidates = [c for c in gdf_ageb.columns
                              if any(k in c.upper() for k in ['POB','TOTAL','HAB'])]
        col_pob = col_pob_candidates[0] if col_pob_candidates else None

        if col_pob:
            tab_pna   = tabla_pob_por_categoria(gdf_ageb, 'cat_PNA',   col_pob, "📍 PNA · Centros de salud", "#7eb8f7")
            tab_camas = tabla_pob_por_categoria(gdf_ageb, 'cat_camas', col_pob, "🛏️ H. no especializados",   "#a29bfe")
            tab_sna   = tabla_pob_por_categoria(gdf_ageb, 'cat_sna',   col_pob, "🏥 Todos los hospitales",   "#55efc4")
        else:
            msg = html.P("Columna de población no encontrada.", style={"color":"#555","fontSize":"11px"})
            tab_pna = tab_camas = tab_sna = msg
    except FileNotFoundError:
        msg = html.P("⚠️ anexo1_desde_excel_tiempos.gpkg no encontrado en la carpeta del proyecto.",
                     style={"color":"#f39c12","fontSize":"11px","fontFamily":"DM Mono"})
        tab_pna = tab_camas = tab_sna = msg
    except Exception as e:
        msg = html.P(f"⚠️ Error AGEBs: {e}", style={"color":"#e74c3c","fontSize":"11px","fontFamily":"DM Mono"})
        tab_pna = tab_camas = tab_sna = msg

    return (
        mapa_html,
        f"{stats['media']:.0f} min",
        f"{stats['mediana']:.0f} min",
        f"{stats['pct_30']:.1f}%",
        f"{stats['pct_60']:.1f}%",
        f"{n_pna:,}",
        f"{pct_baja:.1f}%",
        status,
        barras,
        tabla,
        ins_verde,
        ins_ambar,
        ins_rojo,
        tab_pna,
        tab_camas,
        tab_sna,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
