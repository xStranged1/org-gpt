import re
from typing import Union

from fastapi import FastAPI
import chromadb
from PyPDF2 import PdfReader
from pydantic import BaseModel
import cohere
import numpy as np
import json
from dotenv import load_dotenv
import os
load_dotenv()
class QueryRequest(BaseModel):
    query: str

def extraer_parrafos(archivo_pdf):
    lector = PdfReader(archivo_pdf)
    parrafos = []
    articulo_actual = ""

    for pagina in lector.pages:
        texto = pagina.extract_text()
        lineas = texto.split("\n")
        
        for linea in lineas:
            linea = linea.strip()
            
            # Ignorar líneas que empiezan con "Capítulo"
            if linea.startswith("Capítulo"):
                continue

            # Si la línea empieza con "Art.", se considera el inicio de un nuevo artículo
            if re.match(r"^Art\.?\s?\d+", linea):
                # Si hay un artículo acumulado, agregarlo al array
                if articulo_actual:
                    parrafos.append(articulo_actual.strip())
                articulo_actual = linea  # Comenzar un nuevo artículo
            else:
                # Si no empieza con "Art.", es parte del artículo actual
                articulo_actual += " " + linea

    # Agregar el último artículo si existe
    if articulo_actual:
        parrafos.append(articulo_actual.strip())

    return parrafos

# Ruta del archivo PDF
ruta_pdf = "./src/docs/reglamento-oficial-firmado.pdf"

# Extraer párrafos mejorados
parrafos = extraer_parrafos(ruta_pdf)

# Imprimir como un array hardcodeado
print("parrafos = [")
for parrafo in parrafos:
    print(f'    "{parrafo}",')
print("]")

app = FastAPI()
api_key = os.getenv('CO_API_KEY')
co = cohere.ClientV2(api_key=api_key) 
archivo_json = 'embed.json'

try:
    # Verificar si el archivo existe
    if os.path.exists(archivo_json):
        # Si el archivo existe, lo abrimos y leemos su contenido
        print('recupera embed del json')
        with open(archivo_json, 'r') as archivo:
            doc_emb = json.load(archivo)
    else:
        print(f"El archivo '{archivo_json}' no existe.")
        # Embed the documents
        doc_emb = co.embed(
                    model="embed-multilingual-v3.0",
                    input_type="search_document",
                    texts=parrafos,
                    embedding_types=["float"]).embeddings.float
        # Abrir el archivo en modo escritura (si no existe, se creará)
        with open('embed.json', 'w') as archivo:
            # Guardar el diccionario en el archivo en formato JSON
            json.dump(doc_emb, archivo, indent=4)  # 'indent' agrega formato legible
except KeyError as e:
    print(f"An error occurred: {e}")


def return_results(query_emb, doc_emb, documents):
    n = 2 # customize your top N results
    scores = np.dot(query_emb, np.transpose(doc_emb))[0]
    max_idx = np.argsort(-scores)[:n]
    results = []
    for rank, idx in enumerate(max_idx):
        result = dict(rank=rank+1, score=scores[idx], document=documents[idx])
        results.append(result)
    return results

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/query")
async def get_query(request: QueryRequest):
    # Embed the query
    query_emb = co.embed(
                model="embed-multilingual-v3.0",
                input_type="search_query",
                texts=[request.query],
                embedding_types=["float"]).embeddings.float
    results = return_results(query_emb, doc_emb, parrafos)
    return {"result": results}

@app.post("/items")
def create_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}