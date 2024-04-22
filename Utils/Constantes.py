import os
# CONSTANTES
def construir_ruta(base, *paths):
    return os.path.join(base, *paths).replace("/", os.path.sep)

RUTA_ACTUAL = os.getcwd()
RUTA_REMOTA = "Parallel manga translate"

IDIOMAS_ENTRADA_DISPONIBLES = ['Chino', 'Coreano', 'Inglés', 'Japonés']
IDIOMAS_SALIDA_DISPONIBLES = ['Español', 'Inglés', 'Portugués', 'Francés', 'Italiano', 'Francés']
MODELOS_INPAINT = ['opencv-tela', 'lama_mpe', 'lama_large_512px', 'aot']
RUTA_LOCAL_MODELO_INPAINTING = construir_ruta(RUTA_ACTUAL, "Models", "inpainting")
RUTA_MODELO_LAMA = construir_ruta(RUTA_LOCAL_MODELO_INPAINTING, "lama_mpe.ckpt")
RUTA_MODELO_LAMA_LARGE = construir_ruta(RUTA_LOCAL_MODELO_INPAINTING, "lama_large_512px.ckpt")
RUTA_MODELO_AOT = construir_ruta(RUTA_LOCAL_MODELO_INPAINTING, "aot_inpainter.ckpt")
RUTA_LOCAL_FUENTES = construir_ruta(RUTA_ACTUAL, "Fonts")
RUTA_FUENTE = construir_ruta(RUTA_LOCAL_FUENTES, "NewWildWordsRoman.ttf")
RUTA_LOCAL_PDFS = construir_ruta(RUTA_ACTUAL, "pdfs")
RUTA_LOCAL_ZIPS = construir_ruta(RUTA_ACTUAL, "zips")
RUTA_LOCAL_TEMPORAL = construir_ruta(RUTA_ACTUAL, "temp")

TAMANIO_MINIMO_FUENTE = 12
FACTOR_ESPACIO = 0.42
URL_FUENTE = "https://drive.google.com/file/d/1uIAh-nGGi04f-7moWsKvRhTbAj-Oq84O/view?usp=sharing"
URL_MODELO_LAMA = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt"
URL_MODELO_LAMA_LARGE = "https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt"
URL_MODELO_AOT = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt"

COLOR_BLANCO = (255, 255, 255)
COLOR_NEGRO = (0, 0, 0)