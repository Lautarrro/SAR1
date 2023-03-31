import streamlit as st
import folium
import json
from streamlit_folium import folium_static, st_folium
# Título de la aplicación
st.title("Mapa interactivo")

# Crear un mapa centrado en una ubicación determinada
m = folium.Map(location=[-34.92, -57.95], zoom_start=12)
m.add_child(folium.LatLngPopup())

# Añadir un marcador en el mapa
mapa = st_folium(m, height=350, width=700, key='mapat')
localizacion = st.button("Guardar punto en el mapa")
if localizacion:
    # Obtener la ubicación del punto seleccionado por el usuario
    location = [mapa['last_clicked']['lng'], mapa['last_clicked']['lat']]

    with open("marcadores.json", "w") as f:
        json.dump(location, f)

    # Mostrar un mensaje de confirmación al usuario
    st.success("Ubicación del marcador guardada correctamente")









"""
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
"""
