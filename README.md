# FunMeca_ta1
Este Github ha sido creado para poder desarrolar el software de deteccion de centro de masa de un persona
En este caso las clases han sido separadas en archivos independientes
1. La clase classCOM la cual se encarga del procesamiento del video
2. La clase classFrontend la cual se encarga del fronted en streamlit

Ademas de eso se tiene el archivo main donde unicamente se llaman las clases
NO obstante se usa el archivo master.py para correr directamentet todo en un solo lugar. En los demas archivos se evito el uso de open CV para poder correrlo de manera local y ver el funcionamiento frame por frame. En cambio, el archivo master si utiliza opnCV para mejorar la experiencia
