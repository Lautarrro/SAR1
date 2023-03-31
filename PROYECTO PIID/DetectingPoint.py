import streamlit as st
import folium
import pandas as pd
from streamlit_folium import folium_static
import json

st.set_page_config(layout="wide")

def render_map():
    # Carga los datos del mapa desde un archivo CSV
    data = pd.read_csv("data.csv")
    
    # Crea un mapa centrado en las coordenadas (0, 0)
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Agrega un marcador para cada punto en los datos del mapa
    for index, row in data.iterrows():
        folium.Marker([row['lat'], row['lon']], popup=row['name']).add_to(m)
        
    return m
def save_points(points):
    # Crea un diccionario de Python a partir de los puntos seleccionados
    data = {"points": []}
    for point in points:
        data["points"].append({
            "lat": point[0],
            "lon": point[1]
        })
        # Guarda los datos en un archivo JSON
    with open("points.json", "w") as f:
        json.dump(data, f)
# Título de la aplicación
st.title("Seleccionar puntos en el mapa")

# Renderiza el mapa con Folium
map = render_map()
folium_static(map)

# Botón para guardar los puntos seleccionados
if st.button("Guardar puntos"):
    # Obtiene los puntos seleccionados por el usuario
    selected_points = map.location_picker.points
    
    # Guarda los puntos seleccionados en un archivo JSON automáticamente
    save_points(selected_points)
    
    # Muestra un mensaje de éxito
    st.success(f"Se guardaron {len(selected_points)} puntos en 'points.json'.")
