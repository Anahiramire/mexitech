# mexitech
MexiTech Market RAG Assistant — Enterprise AI Chatbot powered by LangChain, FAISS, and Groq LLMs. UI built with Streamlit and deployed on Oracle Cloud Infrastructure (OCI).

El sistema combina el poder del modelo de lenguaje **Google Gemini** con la búsqueda vectorial rápida de **FAISS** a través de **LangChain**, todo presentado en una interfaz interactiva creada con **Streamlit**.

---

## Características principales

* **Carga e indexación dinámica:** Procesamiento de documentos PDF y archivos JSON.
* **Búsqueda Semántica:** Generación de embeddings vectoriales indexados con FAISS para recuperar únicamente la información relevante.
* **Modelo LLM de Alta Precisión:** Integración con la API de Google Gemini (`langchain-google-genai`).
* **Interfaz de Chat:** Desarrollada con Streamlit para una experiencia sencilla y fluida.

---

## Requisitos previos

Antes de comenzar, asegúrate de tener instalado en tu sistema:

* **Python 3.10** o superior
* **Git**
* Una **Google Gemini API Key** (obtenida desde [Google AI Studio](https://aistudio.google.com/))

---

## Preparación del Ambiente e Instalación

Sigue detalladamente estos pasos para configurar el entorno de desarrollo:

### Paso 1: Clonar el repositorio
Abre tu terminal y clona el proyecto en tu máquina local:
`git clone [https://github.com/Anahiramire/mexitech.git](https://github.com/Anahiramire/mexitech.git)`  
`cd mexitech`

### Paso 2: Crear y activar el entorno virtual
Es indispensable usar un entorno virtual para aislar las dependencias del proyecto.

* **En Linux / macOS / Instancia Ubuntu OCI:**
  * `python3 -m venv venv`
  * `source venv/bin/activate`

* **En Windows (CMD / PowerShell):**
  * `python -m venv venv`
  * `venv\Scripts\activate`

*(Sabrás que está activo porque verás `(venv)` al inicio de la línea de comandos).*

### Paso 3: Actualizar pip e instalar dependencias
Con el entorno virtual activo, instala todos los paquetes necesarios ejecutando:

`pip install --upgrade pip`  
`pip install streamlit langchain langchain-google-genai faiss-cpu pypdf jq python-dotenv`

---

## Configuración de Variables de Entorno

Para habilitar la conexión con la API de Google Gemini:

### Paso 1: Crear el archivo `.env`
En la raíz del proyecto (`/mexitech/`), crea un archivo llamado `.env`:
`touch .env`

### Paso 2: Agregar la API Key
Abre el archivo `.env` e ingresa tu credencial de la siguiente manera:
`GOOGLE_API_KEY="tu_api_key_aqui"`

---

## Ejecución de la Aplicación

### Paso 1: Iniciar el servidor de Streamlit
Asegúrate de tener el entorno `venv` activo y ejecuta:
`streamlit run app.py`

### Paso 2: Abrir en el navegador
El sistema se abrirá automáticamente en tu navegador predeterminado en:  
`http://localhost:8501`

---

## Evidencia de Despliegue en la Nube (OCI)

A continuación se muestra la aplicación ejecutándose correctamente en una instancia **Ubuntu Server en Oracle Cloud Infrastructure (OCI)**:

![Demostración de MexiTech Market en OCI](./captura_oci.png)
