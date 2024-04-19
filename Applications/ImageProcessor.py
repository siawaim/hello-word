import cv2
import os
from Applications.CleanManga import CleanManga
from Applications.TranslateManga import TranslateManga
from Applications.FileManager import FileManager

class ImageProcessor:
    def __init__(self, idioma_entrada, idioma_salida, modelo_inpaint):
        self.file_manager = FileManager()
        self.clean_manga = CleanManga(modelo_inpaint)
        self.translate_manga = TranslateManga(idioma_entrada, idioma_salida)

    def procesar(self, ruta_carpeta_entrada, ruta_limpieza_salida, ruta_traduccion_salida, lote, transcripcion_queue, traduccion_queue):
        for indice_imagen, archivo in lote.items():
            imagen = cv2.imread(os.path.join(ruta_carpeta_entrada, archivo))
            # Limpieza
            transcripcion_queue.put({
                'agregar_elemento_a_lista' : {
                    'Transcripción' : {
                        'Página': indice_imagen + 1,
                        'Formato': self.obtener_formato_manga(imagen),
                        'Globos de texto': []
                    }
                }
            })
            mascara_capa, imagen_limpia = self.clean_manga.limpiar_manga(imagen)
            cv2.imwrite(os.path.join(ruta_limpieza_salida, archivo), imagen_limpia)
            # Traducción
            traduccion_queue.put({
                'agregar_elemento_a_lista' : {
                    'Traducción' : {
                        'Página': indice_imagen + 1,
                        'Formato': self.obtener_formato_manga(imagen),
                        'Globos de texto': []
                    }
                }
            })
            self.translate_manga.insertar_json_queue(
                indice_imagen = indice_imagen,
                transcripcion_queue=transcripcion_queue,
                traduccion_queue=traduccion_queue
            )
            imagen_traducida = self.translate_manga.traducir_manga(imagen, imagen_limpia, mascara_capa)
            cv2.imwrite(os.path.join(ruta_traduccion_salida, archivo), imagen_traducida)
            
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