import sys
import os
import time
import subprocess
import platform
from zipfile import ZipFile
import threading
from PIL import Image

sys.path.append(os.path.dirname(os.getcwd()))
from Utils.Constantes import RUTA_ACTUAL, RUTA_FUENTE, RUTA_LOCAL_FUENTES, RUTA_LOCAL_MODELO_INPAINTING, RUTA_LOCAL_TEMPORAL, RUTA_MODELO_AOT, RUTA_MODELO_LAMA, RUTA_MODELO_LAMA_LARGE, URL_FUENTE, URL_MODELO_AOT, URL_MODELO_LAMA, URL_MODELO_LAMA_LARGE
from .RemoteFileDownloader import RemoteFileDownloader

class Utilities:
    def __init__(self):
        self.canvas_pdf = None
        self.remote_file_downloader = RemoteFileDownloader()

    def descargar_pdf(self, ruta_pdf_resultante):
        if ruta_pdf_resultante is not None:
            hilo_descarga = threading.Thread(target=self.realizar_descarga_pdf, args=(ruta_pdf_resultante,))
            hilo_descarga.start()
            
    def abrir_pdf(self, ruta_pdf_resultante):
        try:
            if platform.system() == "Windows":
                subprocess.Popen([ruta_pdf_resultante], shell=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", ruta_pdf_resultante])
            else:  # Linux
                subprocess.Popen(["xdg-open", ruta_pdf_resultante])
        except Exception as e:
            print(f"Error al abrir el archivo PDF: {e}")

    def realizar_descarga_pdf(self, ruta_pdf_resultante):
        try:
            from google.colab import files
            files.download(ruta_pdf_resultante)
        except Exception as e:
            print(f"Error al descargar el archivo: {e}")

    def capitalizar_oraciones(self, texto):
        oraciones = [s.strip().capitalize() for s in texto.replace('\n', ' ').split('. ') if s]
        return ' '.join(oraciones)

    def generar_pdf(self, imagen_actual, canvas_pdf, ruta_imagen_resultante):
        self.canvas_pdf = canvas_pdf
        img = Image.open(ruta_imagen_resultante)

        img_aspect_ratio = img.width / img.height
        page_width, page_height = self.canvas_pdf._pagesize

        if img_aspect_ratio > 1:
            new_img_width = page_width
            new_img_height = int(page_width / img_aspect_ratio)
        else:
            new_img_height = page_height
            new_img_width = int(page_height * img_aspect_ratio)

        img = img.resize((int(new_img_width), int(new_img_height)), Image.LANCZOS)
        
        timestamp = int(time.time())
        temp_img_path = os.path.join(RUTA_LOCAL_TEMPORAL, f"temp_image_{imagen_actual}_{timestamp}.jpg")
        img.save(temp_img_path, format='JPEG', quality=80)

        self.canvas_pdf.drawImage(temp_img_path, 0, 0, width=int(new_img_width), height=int(new_img_height))
        self.canvas_pdf.showPage()

        os.remove(temp_img_path)
        
    def guardar_pdf(self):
        # Guardar el PDF
        self.canvas_pdf.save()
    
    def download_fonts(self):
        if not os.path.isfile(RUTA_FUENTE):
            self.remote_file_downloader.download(
                download_url=URL_FUENTE,
                output_path=RUTA_LOCAL_FUENTES,
            )
    
    def download_models(self):
        if not os.path.isfile(RUTA_MODELO_LAMA):
            self.remote_file_downloader.download(
                download_url=URL_MODELO_LAMA,
                output_path=RUTA_LOCAL_MODELO_INPAINTING,
                output_filename="lama_mpe.ckpt"
            )
        if not os.path.isfile(RUTA_MODELO_LAMA_LARGE):
            self.remote_file_downloader.download(
                download_url=URL_MODELO_LAMA_LARGE,
                output_path=RUTA_LOCAL_MODELO_INPAINTING,
                output_filename="lama_large_512px.ckpt"
            )
        if not os.path.isfile(RUTA_MODELO_AOT):
            self.remote_file_downloader.download(
                download_url=URL_MODELO_AOT,
                output_path=RUTA_LOCAL_MODELO_INPAINTING,
                output_filename="lama_mpe.ckpt"
            )

    def descargar_y_extraer_zip(self, url_drive):
        try:
            ruta_archivo_descargado = self.remote_file_downloader.download(
                download_url=url_drive,
                output_path=RUTA_ACTUAL,
            )
            ruta_extraccion = os.path.join(RUTA_ACTUAL, os.path.splitext(os.path.basename(ruta_archivo_descargado))[0])
            os.makedirs(ruta_extraccion, exist_ok=True)
            with ZipFile(ruta_archivo_descargado, "r") as zip_ref:
                zip_ref.extractall(ruta_extraccion)
            os.remove(ruta_archivo_descargado)
            return ruta_extraccion
        except Exception as e:
            print(f"Error al descargar y extraer zip: {e}")
            return None
        
    def convertir_a_diccionarios(self, lista_de_listas):
        lotes_diccionarios = []
        indice_acumulado = 0
        for sublist in lista_de_listas:
            lote_diccionario = {}
            for archivo in sublist:
                lote_diccionario[indice_acumulado] = archivo
                indice_acumulado += 1
            lotes_diccionarios.append(lote_diccionario)
        return lotes_diccionarios
    
    def is_colab(self):
        try:
            import google.colab
            return True
        except ImportError:
            return False
