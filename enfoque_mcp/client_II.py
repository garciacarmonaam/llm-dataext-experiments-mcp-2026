import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# Crear el modelo
model = ChatOllama(model="llama3.2", base_url="http://127.0.0.1:11434")

server_params = StdioServerParameters(
    command="python3.13",
    args=["medserver_II.py"]
)

async def run_agent(pdf_path):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            agent = create_react_agent(model, tools)

            print("🤖 Solicitando extracción al agente...")

            instruction = f"""
            Tienes un documento de historia clínica en '{pdf_path}'.

            Debes realizar estrictamente las siguientes acciones:

            1. Ejecuta la herramienta `preparar_historia_clinica` usando el path proporcionado.
            2. Finalmente, ejecuta la herramienta `extraer_informacion_paciente` para extraer todos los campos clínicos estructurados.

            ⚠️ Normas obligatorias:
            - Solo puedes actuar invocando las herramientas disponibles.
            - No puedes responder escribiendo texto libre, resúmenes ni explicaciones.
            - No debes programar manualmente ni improvisar.
            - Si un paso falla, debes igualmente intentar completar el flujo hasta terminar.
            - Tu respuesta final debe ser únicamente el resultado del último tool call (`extraer_informacion_paciente`).

            ✅ Termina solo cuando completes las dos llamadas.

            No expliques nada más. Solo actúa invocando herramientas.
            """

            agent_response = await agent.ainvoke({"messages": instruction},config={"recursion_limit": 50})

            if isinstance(agent_response, dict) and "messages" in agent_response:
                mensajes = agent_response["messages"]
                for msg in reversed(mensajes):
                    if hasattr(msg, "content") and msg.content:
                        return msg.content
                return "⚠️ No encontré un mensaje con contenido."
            else:
                return "⚠️ El agente no devolvió un resultado esperado."


async def run_batch():
    resultados = []
    for i in range(1, 7):
        ruta_pdf = f"../Hist{i}.pdf"
        print(f"\n📂 Procesando: {ruta_pdf}")
        try:
            result = await run_agent(ruta_pdf)
            resultados.append((ruta_pdf, result))
            print("\n📄 Resultado extraído:\n")
            print(result)
        except Exception as e:
            print(f"❌ Error procesando {ruta_pdf}: {e}")
    return resultados

if __name__ == "__main__":
    asyncio.run(run_batch())