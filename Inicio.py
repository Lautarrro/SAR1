import traceback

import streamlit as st
import MapaTool as tl
import files as fl

st.set_page_config(page_title='Inicio')

session = st.session_state

# _key = "key.json"
# _service_account = 'proyecto-0002@ee-tucho.iam.gserviceaccount.com'


def login():
    session.log = False
    login = st.empty()
    if form(login):
        session.log = True
        login.empty()
        inicio()



def save_account(account, key_file):
    fl.get_from_record('users', account, key_file)


def form(login):
    ## meter un expander que tome los usuarios ya ingresados. Tener en cuenta que si el archivo json ya no existe,
    with login.form('-f-'):
        account = st.text_input("Service Account")
        key_path = st.text_input("Path to json key")
        ok = st.form_submit_button('Login')
        if ok:
            try:
                session.log = tl.initialize(account, key_path)
                save_account(account, key_path)
                return True
            except TypeError:
                st.error('Datos Invalidos')
                return False
            except Exception:
                raise traceback.format_exc()

def inicio():

    st.markdown(
        """ 
        # Visualización de mapas
        ------
        ## Barra lateral
        > En la barra lateral se puede seleccionar entre Nuevo Mapa y Resultado. Abría que agregar una opción para 
        setear el catálogo que se quiere consultar, la fecha de inicio y fin, 
        la banda (ahora está hardcodeado VV) y la orientacion.
        
        ### Opcion: Nuevo Mapa
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
        
        
        """)


try:
    if st.session_state.log:
        inicio()
    else:
        raise AttributeError
except AttributeError:
    login()

