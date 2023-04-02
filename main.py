import os

import streamlit as st
import folium
import streamlit.components.v1 as components
import json

from PIL import Image
from streamlit_folium import folium_static, st_folium
import MapaTool



def get_files(type, type2):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "mapas")
    lista = list(filter(lambda file: type or type2 in file, next(os.walk(path), (None, None, []))[2]))
    lista.sort()
    return lista

def inicio():
    st.markdown(
        """ 
        # Visualización de mapas
        ------
        ## Barra lateral
        > En la barra lateral se puede seleccionar entre Mapa y Resultado. Abría que agregar una opción para 
        setear el catálogo que se quiere consultar, la fecha de inicio y fin, 
        la banda (ahora está hardcodeado VV) y la orientacion.
        
        ### Opcion: Mapas
        ------
        > Carga un mapa y permite que clickeando en cualquier parte se seleccione un punto.
        Al apretar el botón guardar punto se establece en *session.point* el valor del punto, si se selecciona
        otro lugar del mapa y se vuelve a apretar el botón guardar se cambia el valor se *session.point* y el
        punto anterior no se va a guardar
        
        ### Opción: Resultado
        ------
        > Muestra una lista de los archivos posibles a cargar, pueden ser html para los mapas o png para las
        graficas de matplot. 
        
        > Para cargar el punto que se seleccionó antes hay que usar el botón nuevo mapa.
        Esto invoca a las funciones de __MapaTool__. 
        
        > Si ya hay archivos en la carpeta van a figurar en la lista desplegable.
        
        > Cada nuevo mapa va a tener el nombre mapa en combinación 
        con el número siguiente al último nombre de mapa que se encuentre en la carpeta "__mapas__". 
        
        > Por cada nuevo mapa se genera también su gráfica correspondiente en formato "__.png__"
        con el mismo número. 
        
        > Falta seguir integrando las funcionalidades del colab a la clase de __MapaTools__.
        """)


def select_point():
    st.title("Mapa interactivo")

    # Crear un mapa centrado en una ubicación determinada
    m = folium.Map(location=[-34.92, -57.95], zoom_start=12)
    m.add_child(folium.LatLngPopup())

    # Añadir un marcador en el mapa
    mapa = st_folium(m, height=350, width=700, key='mapat')
    localizacion = st.button("Guardar punto en el mapa")
    if localizacion:
        st.success("Ubicación del marcador guardada correctamente")
        # Obtener la ubicación del punto seleccionado por el usuario

        session.point = [mapa['last_clicked']['lng'], mapa['last_clicked']['lat']]


def load_map():
    files = get_files(".html", ".png")
    new = st.button("Nuevo Mapa")
    if files:
        file = st.selectbox("Mapas", files)
        path = os.path.join("mapas",file)
        if file[-4:] == "html":
            with open(path, 'r') as ma:
                components.html(ma.read(), height=500)
        else:
            print(file)
            st.image(Image.open(path))
    if session.point and new:
        tl = MapaTool.Tools(session.point)
        tl.first_map()


session = st.session_state
paginas = { 'Inicio': inicio, 'Mapa': select_point, 'Resultado': load_map}
lista = st.sidebar.selectbox("Opciones", paginas.keys())
paginas[lista]()