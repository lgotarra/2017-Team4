# -*- coding: utf-8 -*-

import os
import sys 

sys.path.insert(0,'/home/oscarlinux/Escritorio/UPC/Q5/GDSA/Projecte/') # Creem el path a on hi han totes les subcarpetes

path_imagenes_train = "TerrassaBuildings900/train/images" #Directori on hi ha les imatges d'entrenament
path_imagenes_val = "TerrassaBuildings900/val/images" #Directori on hi ha les imatges de validació
dir_archivos_txt = "txt/" # Directori on guardarem els fitxers txt amb les ID de les imatges

def build_database(directori_imag, directori_txt, etiqueta):
    imag_dir = os.listdir(directori_imag) #Assigna el nom de cada imatge al vector imag_dir
    print("Archivos leidos del directorio(path absoluto): " + os.path.abspath(directori_imag))
    if not os.path.exists(dir_archivos_txt):
        os.makedirs(dir_archivos_txt)

    fichero_txt = open(directori_txt + 'ID_images_'+etiqueta+'.txt', 'w') # Obim el fitxer txt on guardarem les ID
    for imagenes  in imag_dir: # Llegim el nom de les imatges del vector imag_dir
        fichero_txt.write(imagenes[0:-4] + "\n") # Escribim el nom de l'imatge sense jpg i el final de linea un ENTER

# os.getcwd() # Aconseguim el directori actual 

build_database(path_imagenes_train, dir_archivos_txt, 'train') # Crida a la funció creada
build_database(path_imagenes_val, dir_archivos_txt, 'val') # Crida a la funció creada 