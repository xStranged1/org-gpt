import re
from typing import Union

from fastapi import FastAPI
import chromadb
from PyPDF2 import PdfReader
from pydantic import BaseModel

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

chroma_client = chromadb.Client()

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")
# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=parrafos,
    ids = [str(i) for i in range(1, len(parrafos) + 1)]
)
results = collection.query(
    query_texts=["animal bueno para la compañia"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/query")
async def get_query(request: QueryRequest):
    results = collection.query(query_texts=request.query, n_results=3)
    print(results)
    return {"result": results}

@app.post("/items")
def create_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}