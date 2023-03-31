import streamlit as st
import folium
import json
from streamlit_folium import folium_static
# Título de la aplicación
st.title("Mapa interactivo")

# Crear un mapa centrado en una ubicación determinada
mapa = folium.Map(location=[40.416775, -3.703790], zoom_start=13)

# Añadir un marcador en el mapa
marker = folium.Marker(location=[40.416775, -3.703790])
marker.add_to(mapa)

localizacion = st.button("Selecciona un punto en el mapa")
if localizacion:
    # Obtener la ubicación del punto seleccionado por el usuario
    location = mapa.latlng

    # Añadir un marcador en el punto seleccionado por el usuario
    marker = folium.Marker(location=location)
    marker.add_to(mapa)

# Mostrar el mapa en Streamlit
folium_static(mapa)

# Botón para guardar la ubicación del marcador en un archivo JSON
if st.button("Guardar marcador"):
    # Obtener la ubicación del marcador en el mapa
    location = [marker.location]

    # Guardar la ubicación del marcador en un archivo JSON
    with open("marcadores.json", "w") as f:
        json.dump(location, f)

    # Mostrar un mensaje de confirmación al usuario
    st.success("Ubicación del marcador guardada correctamente")
