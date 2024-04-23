import os
from loguru import logger
import warnings
from Applications.ParallelProcessor import ParallelProcessor
from Applications.ImageProcessor import ImageProcessor
from Applications.Utilities import Utilities
from Utils.Constantes import MODELOS_INPAINT

# logger.remove()
warnings.filterwarnings("ignore", message="The class ViTFeatureExtractor is deprecated", category=FutureWarning)

if __name__ == '__main__':
    utilities = Utilities()
    utilities.download_fonts()
    utilities.download_models()

    modelo_inpaint_seleccionado = MODELOS_INPAINT[1]
    print(f"Modelo inpaint seleccionado: {modelo_inpaint_seleccionado}")
    batch_size = 8 # Tamaño de lote -> Si hay 24 imágenes entonces se crearán 3 procesos de 8 imágenes cada uno
    idioma_entrada = "Japonés"
    idioma_salida = "Español"
    ruta_carpeta_entrada = "Dataset"
    ruta_carpeta_salida = os.path.join(ruta_carpeta_entrada, "Outputs")
    ruta_carpeta_limpieza = os.path.join(ruta_carpeta_salida, "Limpieza")
    ruta_carpeta_traduccion = os.path.join(ruta_carpeta_salida, "Traduccion")
    os.makedirs(ruta_carpeta_salida, exist_ok=True)
    os.makedirs(ruta_carpeta_limpieza, exist_ok=True)
    os.makedirs(ruta_carpeta_traduccion, exist_ok=True)

    image_procesor = ImageProcessor(
        idioma_entrada=idioma_entrada,
        idioma_salida=idioma_salida,
        modelo_inpaint=modelo_inpaint_seleccionado
    )

    parallel_processor = ParallelProcessor()
    parallel_processor.procesar_en_paralelo(
        ruta_carpeta_entrada=ruta_carpeta_entrada,
        ruta_carpeta_salida=ruta_carpeta_salida,
        process_func=image_procesor.procesar,
        batch_size=batch_size,
    )
