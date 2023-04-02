## Funcionamiento

El script "main.py" se corre usando  

    streamlit run main.py

 Esto abrirá en el navegador una pestaña con un menú inicial 
 que explica el funcionamiento.  
 En lineas generales el "_main.py_" opera sobre el
 navegador y le pasa los datos a "_MapaTool.py_" para 
 procesar el mapa. La clase Tools se ocupa de procesar 
y exportar los mapas como __html__ y los graficos __png__.  
El "__main__" toma los archivos de la carpeta mapas y los
muestra en el navegador.
---
## Librerias

* streamlit
* streamli-folium
* folium
* numpy
* matplot
* earthengine-api
* scipy
* PIL
---
## Autenticación para earth engine  


Para poder usar el script "MapaTool" se requiere la autenticación
del usuario. Los pasos están explicados en 
[este archivo](Autenticacion-EE.pdf)

---
