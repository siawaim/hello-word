import easyocr
import torch
import numpy as np
import cv2
import asyncio
from PIL import Image

from Applications.inpaint import OpenCVInpainter, LamaInpainterMPE, LamaLarge, AOTInpainter, BNInpainter

INSTANCIAS_INPAINT = {
    'opencv-tela': OpenCVInpainter(),
    'lama_mpe': LamaInpainterMPE(),
    'lama_large_512px' : LamaLarge(),
    'aot': AOTInpainter(),
    'B/N': BNInpainter()
}

class CleanManga:
    def __init__(self, modelo_inpaint):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.inpaint_model = modelo_inpaint
        self.inpainter = INSTANCIAS_INPAINT[modelo_inpaint]

    def limpiar_manga(self, imagen):
        resultados = self.obtener_cuadros_delimitadores(imagen)
        mascara_capa = self.fusionar_cuadros_delimitadores(imagen, resultados)
        if self.inpaint_model == 'B/N':
           res_impainting = self.inpainter.inpaint(imagen, resultados)
        elif self.inpaint_model == 'opencv-tela':
            res_impainting = self.inpainter.inpaint(imagen, mascara_capa)
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            res_impainting = loop.run_until_complete(self.inpaint_async(imagen, mascara_capa))

        imagen_limpia = self.convertir_a_imagen_limpia(res_impainting, imagen)
        return mascara_capa, imagen_limpia

    async def inpaint_async(self, imagen, mascara_capa):
        await self.inpainter._load()
        inpainted_image = await self.inpainter._inpaint(imagen, mascara_capa)
        return inpainted_image

    def convertir_a_imagen_limpia(self, res_impainting, imagen):
        pil_image_camuflada_limpieza = Image.fromarray(
            cv2.cvtColor(res_impainting, cv2.COLOR_BGR2RGB))
        pil_image_limpieza = Image.new(
            'RGB', (imagen.shape[1], imagen.shape[0]))
        pil_image_limpieza.paste(pil_image_camuflada_limpieza, (0, 0))
        imagen_limpia = np.asarray(pil_image_limpieza)
        imagen_limpia = cv2.cvtColor(imagen_limpia, cv2.COLOR_RGB2BGR)
        return imagen_limpia

    def obtener_cuadros_delimitadores(self, imagen):
        lector = easyocr.Reader(
            ["ja", "en"], gpu=True if self.device == 'cuda' else False)
        # Realizar reconocimiento de texto en la imagen
        resultados = lector.readtext(imagen, paragraph=False, decoder="beamsearch",
                                     batch_size=3,
                                     beamWidth=3,
                                     width_ths=0.1,
                                     height_ths=0.05,
                                     x_ths=0.1,
                                     y_ths=0.3,
                                     min_size=5,
                                     link_threshold=0.98
                                     )
        return resultados

    def fusionar_cuadros_delimitadores(self, imagen, resultados):
        expansion = 1
        # Obtener dimensiones de la imagen
        height, width = imagen.shape[:2]
        # Crear una máscara de capa vacía del mismo tamaño que la imagen
        mascara = np.zeros((height, width), dtype=np.uint8)
        # Iterar sobre los resultados del reconocimiento de texto
        for detection in resultados:
            caja = detection[0] 
            puntos = np.array(caja, dtype=np.int32).reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(puntos)
            x_margin = max(0, x - expansion)
            y_margin = max(0, y - expansion)
            w_margin = min(w + expansion, width - x_margin)
            h_margin = min(h + expansion, height - y_margin)
            puntos_margin = np.array([[x_margin, y_margin], [x_margin + w_margin, y_margin], 
                                  [x_margin + w_margin, y_margin + h_margin], [x_margin, y_margin + h_margin]], dtype=np.int32)
            # Rellenar la máscara con la caja delimitadora expandida
            cv2.fillPoly(mascara, [puntos_margin], 255)

        return mascara