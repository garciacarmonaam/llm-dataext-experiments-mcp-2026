import os

os.environ["HF_TOKEN"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]

from mcp.server.fastmcp import FastMCP
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from ollama import chat
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
import json
import pandas as pd
from pandas import json_normalize
import logging

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializar FastMCP Server
mcp = FastMCP("MedicalServer")

# Memoria para almacenar la historia clínica cargada
historia_clinica_memoria = {}

# Modelos Pydantic para estructurar respuestas
class HistoriaClinica(BaseModel):
    contenido: str

from typing import Optional

class InformacionPaciente(BaseModel):
    edad: Optional[int] = Field(default=None, description="Edad del paciente")
    sexo: Optional[str] = Field(default=None, description="Sexo del paciente")
    fecha_ingreso_uci: Optional[str] = Field(default=None, description="Fecha de ingreso en la UCI")
    fecha_alta_uci: Optional[str] = Field(default=None, description="Fecha de alta de la UCI")
    motivo_ingreso: Optional[str] = Field(default=None, description="Motivo de ingreso en la UCI")
    diagnostico: Optional[str] = Field(default=None, description="Diagnóstico principal según CIE")
    procedencia: Optional[str] = Field(default=None, description="Procedencia del paciente: quirófano, hospitalización convencional, urgencias u otra área hospitalaria")
    

# Tool: cargar PDF
@mcp.tool()
def preparar_historia_clinica(pdf_path: str) -> str:
    """Prepara un documento médico para su análisis."""
    logging.info(f"Iniciando carga de documento: {pdf_path}")
    
    try:
        if not os.path.exists(pdf_path):
            logging.warning(f"Archivo no encontrado: {pdf_path}")
            return "No pude encontrar el documento médico. ¿Podrías verificar la ubicación?"

        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        logging.info(f"Documento cargado - Páginas: {len(docs)} | Tamaño aprox: {sum(len(d.page_content) for d in docs)//1024}KB")

        full_text = "\n\n".join(doc.page_content for doc in docs)
        
        if not full_text.strip():
            logging.error("El documento no contiene texto legible")
            return "El documento parece estar vacío o no tiene texto extraíble."

        historia_clinica_memoria["historia"] = full_text
        logging.info("Documento preparado exitosamente")
        return "Documento médico listo para su revisión. Contiene información del paciente."

    except Exception as e:
        logging.error(f"Error crítico: {str(e)}", exc_info=True)
        return f"Hubo un problema al preparar el documento: {str(e)}"

@mcp.tool()
def extraer_informacion_paciente() -> InformacionPaciente:
    """Extrae la información clínica estructurada usando RAG + validación estricta y autocorrección de JSON."""
    historia = historia_clinica_memoria.get("historia", "")
    if not historia.strip():
        raise ValueError("No hay historia clínica cargada.")

    # 🔵 Chunking semántico
    docs = [Document(page_content=historia)]
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    chunker = SemanticChunker(embedder)
    documentos_chunked = chunker.split_documents(docs)

    vectorstore = FAISS.from_documents(documentos_chunked, embedder)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    consulta = "Obtener la información clínica completa del paciente"
    documentos_relevantes = retriever.get_relevant_documents(consulta)

    if not documentos_relevantes:
        logging.warning("⚠️ No se encontraron documentos relevantes para la consulta. Se usará la historia completa.")
    else:
        historia_clinica = "\n\n".join(doc.page_content for doc in documentos_relevantes if hasattr(doc, 'page_content') and doc.page_content)

    if not historia_clinica.strip():
        raise ValueError("La historia clínica está vacía después de procesar los documentos relevantes.")

    # 🔵 Generar el formato JSON esperado
    formato_resultado = InformacionPaciente.model_json_schema()

    # 🔵 Llamada directa al modelo usando `chat`
    try:
        response = chat(
        messages=[
            {
                'role': 'user',
                'content': (
                    f"""
                    Eres un extractor clínico automático.

                    Te proporciono una historia clínica real. Extrae únicamente los datos solicitados en formato JSON respetando este esquema:

                    {formato_resultado}

                    Historia clínica:
                    {historia_clinica}

                    ⚠️ Normas obligatorias:
                    - Si un dato no aparece, escribe **null** en el campo correspondiente.
                    - **No** expliques nada, **no** escribas texto libre fuera del JSON.
                    - **No** inventes información médica adicional.
                    - **Solo** responde con un objeto JSON puro y válido.

                    Responde ahora:
                    """
                                ),
                        }
                ],
                model="phi3.5",
                format=formato_resultado
        )

    except Exception as e:
        logging.error(f"❌ Error al invocar el modelo: {e}", exc_info=True)
        raise

    # 🔵 Validar y parsear respuesta
    try:
        file_path = "ResultadosQwen3.xlsx"
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            resultado = InformacionPaciente.model_validate_json(response['message']['content'])
            df_existente = pd.read_excel(file_path, sheet_name='con_pydantic')

            data = resultado.model_dump()
            flattened_data = json_normalize(data)
            df_nuevo = pd.DataFrame(flattened_data)
            pd.set_option('display.max_columns', None)
            df_actualizado = pd.concat([df_existente, df_nuevo], ignore_index=True)
            df_actualizado.to_excel(writer, sheet_name="con_pydantic", index=False)
            logging.info("✅ Información clínica extraída y validada correctamente.")
            return df_actualizado

    except Exception as e:
        logging.error(f"❌ Error validando la respuesta del modelo: {e}", exc_info=True)
        raise



if __name__ == "__main__":
    mcp.run(transport="stdio")
