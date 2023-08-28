import traceback

import streamlit as st
from streamlit_folium import st_folium
import folium
import MapaTool
import files as fl

st.set_page_config(page_title='Nuevo Mapa')


def trigger(point, name):
    tl = MapaTool.Tools(point, st.session_state.config)
    try:
        tl.first_layer(name)
        return f"guardado, nombre '{name}' "
    except Exception as ex:
        raise ex


def config_parametros():
    form = st.form("Ingrese parámetros de configuración")
    with form:
        selects = {'Catalogo': ['COPERNICUS'], 'Collection': ['S1_GRD_FLOAT'],
                   'Banda': ['VV', 'HH', 'VH', 'HV'], 'Órbita': ['DESCENDING', 'ASCENDING']
                   }

        nombre = st.text_input("Nombre de la configuracion")
        catalogo, collection, band, orbit = [st.selectbox(key, options=value)
                                             for key, value in selects.items()]
        start, end = [st.date_input(f"Fecha de {val}").strftime('%Y-%m-%d')
                      for val in ['inicio', 'fin']]
        bounds = st.number_input("Buffer", max_value=100, min_value=10)
        inputs = [f"{catalogo}/{collection}", start, end, band, orbit, bounds]
        accept = st.form_submit_button('Confirmar')
    return nombre, inputs if accept else None


def new_map():
    config = st.selectbox("Parametros del catálogo",
                          options=[item for item in fl.read_json()['config'].keys()]+['new'])
    if config == 'new':
        key, values = config_parametros()
        if key and values:
            st.session_state.config = fl.get_from_record('config', key, values)

    else:
        st.session_state.config = fl.get_from_record('config', config)

    st.success(f"Configuracion: {st.session_state.config}")
    m = folium.Map(location=[-34.92, -57.95], zoom_start=10)
    m.add_child(folium.LatLngPopup())
    mapa = st_folium(m, height=350, width=700, key='mapat')
    name: str = st.text_input(label="Nombre del punto")
    location = st.button("Confirmar Punto")
    # Crear un mapa centrado en una ubicación determinada
    if (name.isalpha() or name.isalnum()) and location:
        try:
            key = [round(mapa['last_clicked']['lng'], 4),
                   round(mapa['last_clicked']['lat'], 4)]
            estado = f"{trigger(key, name)}" \
                if not fl.get_from_record('sar',
                                          f"{key[0]}, {key[1]}", name) else "Ya revisado"
            st.write(f"Punto actual {key} " + estado)
        except TypeError:
            st.error("Seleccione un punto")
        except Exception as ex:
            print(ex)
    else:
        st.error("Ingrese un nombre")


try:
    if st.session_state.log:
        new_map()
except AttributeError:
    print(traceback.format_exc())
    st.error('Log in')
