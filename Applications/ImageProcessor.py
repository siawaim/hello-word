import time
import cv2
import os
import numpy as np
import torch
from Applications.Utilities import Utilities
from Applications.CleanManga import CleanManga
from Applications.TranslateManga import TranslateManga
from Applications.FileManager import FileManager

class ImageProcessor:
    def __init__(self, idioma_entrada, idioma_salida, modelo_inpaint):
        self.file_manager = FileManager()
        self.utilities =Utilities()
        self.clean_manga = CleanManga(modelo_inpaint)
        self.translate_manga = TranslateManga(idioma_entrada, idioma_salida)

    def procesar(self, ruta_carpeta_entrada, ruta_limpieza_salida, ruta_traduccion_salida, lote, transcripcion_queue, traduccion_queue):
        for indice_imagen, archivo in lote.items():
            print(f"Procesando: {archivo}")
            intentos = 0
            img_array = np.fromfile(os.path.join(ruta_carpeta_entrada, archivo), np.uint8)
            imagen = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            while intentos < 3:
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
                    break
                except (torch.cuda.CudaError, RuntimeError) as e:
                    print(f"Error: {e}")
                    imagen = self.reducir_imagen(imagen)
                    torch.cuda.empty_cache()
                    time.sleep(1)
                    intentos += 1
                except Exception as e:
                    print(f"Error: {e}")
                    imagen = self.reducir_imagen(imagen)
                    torch.cuda.empty_cache()
                    time.sleep(1)
                    intentos += 1
            else:
                # Si llegamos a este punto, significa que se superaron los tres intentos
                # Entonces, pasamos a la siguiente iteración del bucle for
                continue
            
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