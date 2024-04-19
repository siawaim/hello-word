import os
from deep_translator import DeeplTranslator, GoogleTranslator
from deep_translator.exceptions import TranslationNotFound, AuthorizationException
from dotenv import load_dotenv

class TranslatorManager:
    def __init__(self, idioma_entrada, idioma_salida):
        load_dotenv()
        DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')
        self.idioma_entrada = idioma_entrada
        self.idioma_salida = idioma_salida
        abv_idioma_entrada, abv_idioma_salida = self.obtener_abreviacion_idioma(idioma_entrada, idioma_salida)
        if DEEPL_API_KEY is not None:
            self.translator = DeeplTranslator(
                api_key=DEEPL_API_KEY, 
                source=abv_idioma_entrada, 
                target=abv_idioma_salida, 
                use_free_api=True
            )
        else:
            self.translator = GoogleTranslator(source=abv_idioma_entrada, target=abv_idioma_salida)
        
    def obtener_abreviacion_idioma(self, idioma_entrada, idioma_salida):
        mapeo_idiomas_admitidos = {
            "Español": "es",
            "Inglés": "en",
            "Portugués": "pt",
            "Francés": "fr",
            "Italiano": "it",
            "Japonés": "ja",
            "Koreano": "ko",
            "Chino": "zh-CN"
        }
        return mapeo_idiomas_admitidos.get(idioma_entrada, "auto"), mapeo_idiomas_admitidos.get(idioma_salida, "es")

    def traducir_texto(self, texto):
        try:
            texto_traducido = self.translator.translate(texto)
            return texto_traducido
        except TranslationNotFound as e:
            # Maneja el error TranslationNotFound aquí
            print(f"No se pudo encontrar una traducción para el texto: {str(e)}")
            return ""