import os
import torch.multiprocessing as mp
from Applications.JsonGenerator import JsonWriter
from Applications.Utilities import Utilities

class ParallelProcessor:
    def __init__(self):
        self.utilities = Utilities()
        mp.set_start_method('spawn', force=True)
    
    def procesar_en_paralelo(self, ruta_carpeta_entrada, ruta_carpeta_salida, process_func, batch_size = 4):
        try:
            archivos = os.listdir(ruta_carpeta_entrada)
            lista_imagenes = [
                archivo for archivo in archivos if archivo.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
            ]
            cantidad_archivos = len(lista_imagenes)
            # Dividir las imágenes en lotes para procesamiento paralelo
            lotes_imagenes =  [lista_imagenes[i:i+batch_size] for i in range(0, cantidad_archivos, batch_size)]
            num_processes = len(lotes_imagenes)
            lotes_imagenes = self.utilities.convertir_a_diccionarios(lotes_imagenes)
            ruta_limpieza_salida = os.path.join(ruta_carpeta_salida, "Limpieza")
            ruta_traduccion_salida = os.path.join(ruta_carpeta_salida, "Traduccion")

            # Inicializar las colas y el proceso para escribir JSON
            transcripcion_queue = mp.Queue()
            traduccion_queue = mp.Queue()
            transcripcion_process = JsonWriter(transcripcion_queue)
            traduccion_process = JsonWriter(traduccion_queue)
            transcripcion_process.start()
            traduccion_process.start()

            transcripcion_queue.put({
                'agregar_entrada' : {
                    'Título' : os.path.basename(ruta_carpeta_entrada),
                    'Páginas': cantidad_archivos,
                }
            })
            traduccion_queue.put({
                'agregar_entrada' : {
                    'Título' : os.path.basename(ruta_carpeta_entrada),
                    'Páginas': cantidad_archivos,
                }
            })
            
            processes = []
            for i in range(num_processes):
                # Seleccionar un lote de imágenes para cada proceso
                lote = lotes_imagenes[i]
                p = mp.Process(
                    target=process_func,
                    args=(
                        ruta_carpeta_entrada,
                        ruta_limpieza_salida,
                        ruta_traduccion_salida,
                        lote,
                        transcripcion_queue,
                        traduccion_queue,
                    ),
                    daemon=True)
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()

            transcripcion_queue.put({
                'ordenar_por_paginas' : {
                    'tipo' : 'Transcripción',
                }
            })
            traduccion_queue.put({
                'ordenar_por_paginas' : {
                    'tipo' : 'Traducción',
                }
            })
            # Señalizar que la escritura de JSON está completa
            transcripcion_queue.put({
                'guardar_en_archivo' : os.path.join(ruta_limpieza_salida, "Transcripción.json")
            })
            traduccion_queue.put({
                'guardar_en_archivo' : os.path.join(ruta_traduccion_salida, "Traducción.json")
            })

            # Esperar a que los procesos de escritura de JSON terminen
            transcripcion_process.join()
            traduccion_process.join()

            return True
        except Exception as e:
            # Captura cualquier excepción y la imprime
            print(f"Error al procesar imágenes en paralelo: {e}")
            return False
        
        
import time
import cv2
import os
import numpy as np
import torch
from Applications.Utilities import Utilities
from Applications.CleanManga import CleanManga
from Applications.TranslateManga import TranslateManga
from Applications.FileManager import FileManager
import logging

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('debug.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

class ImageProcessor:
    def __init__(self, idioma_entrada, idioma_salida, modelo_inpaint):
        self.file_manager = FileManager()
        self.utilities =Utilities()
        self.clean_manga = CleanManga(modelo_inpaint)
        self.translate_manga = TranslateManga(idioma_entrada, idioma_salida)

    def procesar(self, ruta_carpeta_entrada, ruta_limpieza_salida, ruta_traduccion_salida, lote, transcripcion_queue, traduccion_queue):
        for indice_imagen, archivo in lote.items():
            logger.debug(f"Procesando: {archivo}")
            img_array = np.fromfile(os.path.join(ruta_carpeta_entrada, archivo), np.uint8)
            imagen = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            memoria_suficiente = False
            intentos = 0
            while not memoria_suficiente and intentos < 3:
                try:
                    # Limpieza
                    transcripcion_queue.put({
                        'agregar_elemento_a_lista': {
                            'Transcripción': {
                                'Página': indice_imagen + 1,
                                'Formato': self.obtener_formato_manga(imagen),
                                'Globos de texto': []
                            }
                        }
                    })
                    mascara_capa, imagen_limpia = self.clean_manga.limpiar_manga(imagen)
                    archivo_limpieza_salida = os.path.join(ruta_limpieza_salida, archivo)
                    cv2.imwrite(archivo_limpieza_salida, imagen_limpia)
                    # Traducción
                    traduccion_queue.put({
                        'agregar_elemento_a_lista': {
                            'Traducción': {
                                'Página': indice_imagen + 1,
                                'Formato': self.obtener_formato_manga(imagen),
                                'Globos de texto': []
                            }
                        }
                    })
                    self.translate_manga.insertar_json_queue(
                        indice_imagen=indice_imagen,
                        transcripcion_queue=transcripcion_queue,
                        traduccion_queue=traduccion_queue
                    )
                    imagen_traducida = self.translate_manga.traducir_manga(imagen, imagen_limpia, mascara_capa)
                    archivo_traduccion_salida = os.path.join(ruta_traduccion_salida, archivo)
                    cv2.imwrite(archivo_traduccion_salida, imagen_traducida)
                    memoria_suficiente = True
                except (torch.cuda.CudaError, RuntimeError) as e:
                    logger.error(f"Error al procesar el archivo {archivo}: {e}")
                    imagen = self.reducir_imagen(imagen) 
                except Exception as e:
                    logger.error(f"Error al procesar el archivo {archivo}: {e}")
                    imagen = self.reducir_imagen(imagen)
                finally:
                    intentos += 1
                    
    def obtener_formato_manga(self, imagen):
        # Convierte la imagen a escala de grises
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # Calcula el valor medio de intensidad de los píxeles en la imagen
        valor_medio = cv2.mean(imagen_gris)[0]
        # Determina el formato de la manga basado en el valor medio
        if valor_medio < 50 or valor_medio > 200:
            formato = "Blanco y negro (B/N)"
        else:
            formato = "Color"
            
        return formato
    
    def reducir_imagen(self, imagen):
        try:
            porcentaje_reduccion = 0.75
            nuevo_alto, nuevo_ancho = [int(dim * porcentaje_reduccion) for dim in imagen.shape[:2]]
            return cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
        except Exception as e:
            print(f"Error al reducir la imagen: {e}")
            return imagen