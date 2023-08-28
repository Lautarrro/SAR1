import traceback
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import files as fl


st.set_page_config(page_title='Resultados')


def open_map(file, k):
    mapas = [open(fl.file('mapas', f"{i}-{file[12:-5]}"), 'r')
             for i in range(1, k + 1)]
    i = st.slider("Capa", 1, len(mapas))
    container = st.container()
    with container:
        components.html(mapas[i - 1].read(), height=500)
    map(lambda mapa: mapa.close(), mapas)


def open_plot(file):
    st.image(Image.open(file))


def open_views(file, k):
    try:
        with st.expander("Mapas"):
            open_map(fl.file("mapas", file), k)
        with st.expander('Grafico'):
            open_plot(fl.file("plots", f"1-{file}"))
    except Exception:
        st.error(f"{traceback.format_exc()}")


def load_map():
    mapas = fl.get_files("mapas")
    plots = fl.get_files("plots")
    file = st.selectbox("Seleccione el archivo", options=set(mapas + plots))
    try:
        open_views(file, Counter(mapas)[mapas[0]])
    except IndexError:
        st.error("Sin archivos")


try:
    if st.session_state.log:
        load_map()
except AttributeError:
    st.error('Log in')
