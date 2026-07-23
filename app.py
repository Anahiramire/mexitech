import os
import glob
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

# Librerías de LangChain y RAG
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Configuración de la interfaz
st.set_page_config(page_title="Roxxi", page_icon="🤖")
custom_css = """
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #06120E !important;
    }

    h1 {
        background: linear-gradient(135deg, #10B981, #34D399) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
        font-size: 3.5rem !important; 
        text-align: center !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stCaptionContainer"], h2, h3, .stMarkdown p {
        text-align: center;
    }
    
    [data-testid="stCaptionContainer"] {
        font-size: 1.25rem !important; 
        color: #A7F3D0 !important;
        margin-bottom: 2rem !important;
    }

    [data-testid="stChatMessage"] {
        background-color: #0F211B !important;
        border: 1px solid #1C382F !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
    }

    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] li, 
    [data-testid="stChatMessage"] div {
        color: #ECFDF5 !important;
        font-size: 1.18rem !important; 
        line-height: 1.6 !important; 
        text-align: left !important;
    }

    div[data-testid="stExpander"] {
        background-color: #091712 !important;
        border: 1px solid #1C382F !important;
        border-radius: 10px !important;
    }
    
    div[data-testid="stExpander"] summary {
        color: #34D399 !important;
        font-size: 1.1rem !important;
    }

    [data-testid="stBottom"], 
    [data-testid="stChatInputContainer"],
    div[data-testid="stBottom"] > div {
        background-color: transparent !important;
        background: transparent !important;
    }

    div[data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1px solid #1C382F !important;
        background-color: #0F211B !important;
    }

    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #ECFDF5 !important;
        font-size: 1.1rem !important; 
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #10B981 !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.3) !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.title("Roxxi")
st.caption("MexiTech Market")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.info("No se encontró la GROQ_API_KEY en el archivo .env")
    st.stop()

google_api_key = os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def inicializar_vectorstore():
    docs_dir = "documentos"
    faiss_index_path = "faiss_index"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if os.path.exists(faiss_index_path):
        return FAISS.load_local(
            faiss_index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

    documentos = []
    for filepath in glob.glob(os.path.join(docs_dir, "*")):
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext in [".txt", ".md"]:
                loader = TextLoader(filepath, encoding="utf-8")
                documentos.extend(loader.load())
            elif ext == ".pdf":
                loader = PyPDFLoader(filepath)
                documentos.extend(loader.load())
            elif ext == ".csv":
                loader = CSVLoader(filepath, encoding="utf-8")
                documentos.extend(loader.load())
            elif ext == ".json":
                loader = JSONLoader(filepath, jq_schema='.', text_content=False)
                documentos.extend(loader.load())
        except Exception as e:
            st.sidebar.warning(f"Error en {os.path.basename(filepath)}: {e}")

    # Chunking / Fragmentación
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documentos)

    # Indexación Vectorial con FAISS
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Guardar en disco local para evitar re-procesar en el futuro
    vectorstore.save_local(faiss_index_path)
    
    return vectorstore

try:
    with st.spinner("Cargando la base de conocimiento..."):
        vectorstore = inicializar_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    system_prompt = (
        "Eres el Agente Virtual de MexiTech Market.\n"
        "Responde a la pregunta del usuario utilizando ÚNICAMENTE el siguiente contexto retribuido. "
        "Si la información no está en el contexto, indica amablemente que no dispones de ese dato.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {input}"
    )

    prompt_template = ChatPromptTemplate.from_template(system_prompt)
    
    # Modelo ligero optimizado para rapidez y mayor cuota de uso
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Interfaz de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy el agente de MexiTech Market. ¿Qué duda tienes sobre nuestras políticas internas?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Consultando información..."):
                docs_relacionados = retriever.invoke(user_input)
            
                # Generar respuesta con la cadena RAG
                respuesta_texto = rag_chain.invoke(user_input)
            
                st.write(respuesta_texto)
                
                if docs_relacionados:
                    with st.expander("Fuentes encontradas en los documentos"):
                        # Diccionario para agrupar páginas por archivo
                        fuentes = {}
                        for doc in docs_relacionados:
                            archivo = os.path.basename(doc.metadata.get("source", "Desconocido"))
                            pagina = doc.metadata.get("page", None)
                            
                            if archivo not in fuentes:
                                fuentes[archivo] = []
                            if pagina is not None and (pagina + 1) not in fuentes[archivo]:
                                fuentes[archivo].append(pagina + 1)

                        for archivo, paginas in fuentes.items():
                            if paginas:
                                st.caption(f"• **Archivo:** `{archivo}` (Pág. {', '.join(map(str, paginas))})")
                            else:
                                st.caption(f"• **Archivo:** `{archivo}`")

                st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})

except Exception as e:
    st.error(f"Error configurando el RAG: {e}")