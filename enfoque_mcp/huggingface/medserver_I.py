import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["HF_TOKEN"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]

from mcp.server.fastmcp import FastMCP
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from huggingface_hub import InferenceClient
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser

from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from pandas import json_normalize
import logging
import json
import re


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

    historia_clinica = (
        "\n\n".join(doc.page_content for doc in documentos_relevantes if hasattr(doc, 'page_content') and doc.page_content)
        if documentos_relevantes else historia
    )

    if not historia_clinica.strip():
        raise ValueError("La historia clínica está vacía después de procesar los documentos relevantes.")

    # 🔵 Generar el formato JSON esperado (solo "properties")
    formato_resultado = json.dumps(
        {field: None for field in InformacionPaciente.model_fields},
        indent=2
    )    
    logging.info(f"Formato esperado: {formato_resultado}")

    # 🔵 Llamada al modelo
    try:
        prompt = ChatPromptTemplate.from_messages([
            (
                "human",
                """
                Eres un extractor clínico automático. Extrae únicamente los datos solicitados en formato JSON plano.

                ⚠️ Instrucciones estrictas:
                - Devuelve **solo** el objeto JSON, sin texto adicional, sin explicaciones ni comentarios.
                - **No uses** etiquetas como <think>, ```json, ``` u otros marcadores Markdown.
                - No devuelvas `properties`, `title` ni `type`. Solo el contenido real del objeto.
                - Si un dato no aparece, escribe `null`.
                - El resultado **debe** tener exactamente esta estructura:

                {formato_resultado}

                Historia clínica:
                {historia_clinica}

                Responde ahora con el JSON plano, sin envoltorios:
                """
            )
        ])


        client = InferenceClient(
            provider="together",
            api_key=os.environ["HF_TOKEN"],
        )

        message = prompt.invoke({
            "formato_resultado": formato_resultado,
            "historia_clinica": historia_clinica
        }).messages[0].content

        logging.info(message)

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
        logging.info(response)

    except Exception as e:
        logging.error(f"❌ Error al invocar el modelo: {e}", exc_info=True)
        raise

    # 🔵 Validar y parsear respuesta
    try:
        # Limpiar bloque markdown ```json ... ```
        raw_content = response.content.strip()
        cleaned_json_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.IGNORECASE).strip()

        # Parsear JSON
        parsed = json.loads(cleaned_json_str)
        resultado = InformacionPaciente.model_validate(parsed)

        # Guardar en Excel
        file_path = "ResultadosNousHermes2Mixtral.xlsx"
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df_existente = pd.read_excel(file_path, sheet_name='con_pydantic')
            df_nuevo = pd.json_normalize(resultado.model_dump())
            df_actualizado = pd.concat([df_existente, df_nuevo], ignore_index=True)
            df_actualizado.to_excel(writer, sheet_name="con_pydantic", index=False)

        logging.info("✅ Información clínica extraída y validada correctamente.")
        return resultado

    except Exception as e:
        logging.error(f"❌ Error validando o guardando la respuesta del modelo: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    mcp.run(transport="stdio")
