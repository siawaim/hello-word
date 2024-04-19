import json
import torch.multiprocessing as mp

class JsonGenerator:
    def __init__(self):
        self.datos = {}

    def agregar_entrada(self, clave, valor):
        self.datos[clave] = valor

    def agregar_elemento_a_lista(self, clave, elemento):
        self.datos.setdefault(clave, []).append(elemento)

    def agregar_a_sublista(self, clave_lista, pagina, clave_sublista, elemento_sublista):
        # Asegúrate de que la clave_lista ya es una lista en los datos
        if clave_lista not in self.datos or not isinstance(self.datos[clave_lista], list):
            raise ValueError(f"La clave '{clave_lista}' no existe o no es una lista.")
        
        # Obtener la página correspondiente o agregar una nueva si no existe
        pagina_data = next((item for item in self.datos[clave_lista] if item.get("Página") == pagina), None)
        if pagina_data is None:
            pagina_data = {"Página": pagina}
            self.datos[clave_lista].append(pagina_data)
        
        # Asegúrate de que el elemento específico es un diccionario
        if not isinstance(pagina_data, dict):
            raise ValueError(f"El elemento para la página {pagina} no es un diccionario.")
        
        # Añadir o actualizar la sublista
        pagina_data.setdefault(clave_sublista, []).append(elemento_sublista)

    def ordenar_por_paginas(self, tipo):
        try:
            self.datos[tipo] = sorted(self.datos[tipo], key=lambda x: x["Página"])
            return True
        except Exception as e:
            print(f"Error al ordenar json: {e}")
        
    def guardar_en_archivo(self, nombre_archivo):
        with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
            json.dump(self.datos, archivo, ensure_ascii=False, indent=4)
            
class JsonWriter(mp.Process):
    def __init__(self, result_queue):
        super(JsonWriter, self).__init__()
        self.result_queue = result_queue
        self.json_generator = JsonGenerator()

    def run(self):
        while True:
            data = self.result_queue.get()
            # Asumiendo que data es un objeto json
            metodo = next(iter(data))
            if metodo == 'agregar_entrada':
                for clave, valor in data[metodo].items():
                    self.json_generator.agregar_entrada(
                        clave=clave,
                        valor=valor
                    )
            elif metodo == 'agregar_elemento_a_lista':
                for clave, elmento in data[metodo].items():
                    self.json_generator.agregar_elemento_a_lista(
                        clave=clave,
                        elemento=elmento
                    )
            elif metodo == 'agregar_a_sublista':
                sublista_data = data[metodo]
                self.json_generator.agregar_a_sublista(
                    clave_lista=sublista_data['clave_lista'],
                    pagina=sublista_data['pagina'],
                    clave_sublista=sublista_data['clave_sublista'],
                    elemento_sublista=sublista_data['elemento_sublista']
                )
            elif metodo == 'ordenar_por_paginas':
                tipo = data[metodo]['tipo']
                self.json_generator.ordenar_por_paginas(
                        tipo=tipo,
                )   
            elif metodo == 'guardar_en_archivo':
                self.json_generator.guardar_en_archivo(data[metodo])
                break
            else:
                break