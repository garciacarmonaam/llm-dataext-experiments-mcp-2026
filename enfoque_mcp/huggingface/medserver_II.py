import os
#os.environ["HF_TOKEN"] = 
#os.environ["MISTRAL_API_KEY"] 
import logging
import pandas as pd
from typing import Optional
from pandas import json_normalize
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from ollama import chat

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

mcp = FastMCP("MedicalServer")
historia_clinica_memoria = {}

class InformacionPaciente(BaseModel):
    edad: Optional[int] = Field(default=None, description="Edad del paciente")
    sexo: Optional[str] = Field(default=None, description="Sexo del paciente")
    fecha_ingreso_uci: Optional[str] = Field(default=None, description="Fecha de ingreso en la UCI")
    fecha_alta_uci: Optional[str] = Field(default=None, description="Fecha de alta de la UCI")
    motivo_ingreso: Optional[str] = Field(default=None, description="Motivo de ingreso en la UCI")
    diagnostico: Optional[str] = Field(default=None, description="Diagnóstico principal según CIE")
    procedencia: Optional[str] = Field(default=None, description="Procedencia del paciente: quirófano, hospitalización convencional, urgencias u otra área hospitalaria")
    

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
        historia_clinica_memoria["contexto"] = historia_clinica

    if not historia_clinica.strip():
        raise ValueError("La historia clínica está vacía después de procesar los documentos relevantes.")

    datos = {}
    for campo in InformacionPaciente.model_fields:
        datos[campo] = extraer_campo(campo)
    historia_clinica_memoria["datos_paciente"] = datos

    # 🔵 Guardar datos en hoja específica sin Pydantic
    df_datos = pd.DataFrame([datos])  # Convertimos a DataFrame

    try:
        file_path = "ResultadosNousHermes2Mixtral.xlsx"
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df_existente = pd.read_excel(file_path, sheet_name='sin_pydantic')

            df_nuevo = df_datos
            pd.set_option('display.max_columns', None)
            df_actualizado = pd.concat([df_existente, df_nuevo], ignore_index=True)
            df_actualizado.to_excel(writer, sheet_name="sin_pydantic", index=False)
            logging.info("✅ Información clínica extraída y validada correctamente.")
            return df_actualizado

    except Exception as e:
        logging.error(f"❌ Error validando la respuesta del modelo: {e}", exc_info=True)
        raise


def extraer_campo(campo: str) -> Optional[str]:
    contexto = historia_clinica_memoria.get("contexto", "")
    if not contexto.strip():
        raise ValueError("No hay contexto disponible.")

    descripcion = InformacionPaciente.model_fields[campo].description
    tipo_campo = InformacionPaciente.model_fields[campo].annotation

    # Construcción del prompt dinámico para cada campo
    prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            f"""
            Eres un extractor de datos médicos.

            Campo a extraer: **{campo}**
            Descripción: {descripcion}

            Normas estrictas:
            - Si encuentras el valor, responde únicamente con el dato.
            - Si no está, responde exactamente: null (sin comillas).
            - Para números, responde solo el número.
            - Para booleanos, responde true o false.
            - No expliques nada.
            - **No uses** etiquetas como <think>, ```json, ``` u otros marcadores Markdown.
            - No inventes datos.

            Historia clínica:
            {contexto}
            """
        )
    ])

    client = InferenceClient(
            provider="together",
            api_key=os.environ["HF_TOKEN"],
        )
    
    message = prompt.invoke({}).messages[0].content
    
    completion = client.chat.completions.create(
            model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ],
        )

    response = completion.choices[0].message
    #logging.info(response)

    try:
        # Postprocesado
        valor = response.content.strip()
        # logging.info('Se extrae campo ' + campo)
        # logging.info(valor)

        if valor.lower() == "null":
            return None

        if tipo_campo == int:
            return int(valor)
        if tipo_campo == bool:
            return valor.lower() in ("true", "sí", "si", "yes")

        return valor

    except Exception as e:
        logging.error(f"❌ Error extrayendo campo {campo}: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    mcp.run(transport="stdio")
