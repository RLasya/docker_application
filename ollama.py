# import ollama

# response = ollama.generate(model='gemma:2b',
# prompt='what is a qubit?')
# print(response['response'])


# from langchain_community.llms import Ollama

# llm = Ollama(model="llama2")

# llm.invoke("tell me about partial functions in python")

import streamlit as st
from langchain_community.llms import Ollama
from langchain import PromptTemplate
from langchain.chains import LLMChain
# from googletrans import Translator  
from langchain.document_loaders import YoutubeLoader
import warnings
from PIL import Image


warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core")

# Función para crear la base de datos a partir de la URL de un video de YouTube
def create_db_from_youtube_video_url(video_url: str, language: str):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True, language=[language])
    db = loader.load()
    return db

# Función para obtener la respuesta a partir de una consulta en la base de datos
def get_response_from_query(db, query, k=4):
    llm = Ollama(model="llama2")
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        Eres un asistente inteligente que puede responder preguntas sobre videos de YouTube 
        basado en la transcripción del video.

        Responde la siguiente pregunta: {question}
        Buscando en la transcripción del siguiente video: {docs}

        Utiliza únicamente la información factual de la transcripción para responder la pregunta.

        Si sientes que no tienes suficiente información para responder la pregunta, di "No lo sé".

        Tus respuestas deben ser detalladas y completas, no des detalles de más que no tengan que ver con la transcripción.
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=db)
    response = response.replace("\n", "")
    return response

# Streamlit UI
st.title("Asistente de Video YouTube")

# Cargar la imagen desde un archivo local
# image = Image.open("..\img\youtube_logo.png")

# # Mostrar la imagen en Streamlit
# st.image(image, width=200)

video_url = st.text_input("Ingrese el enlace del video de YouTube:")
language = st.selectbox("Seleccione el idioma de la transcripción:", options=['es', 'en'], index=0)

if video_url:
    db = create_db_from_youtube_video_url(video_url, language)

    query = st.text_input("Ingrese su pregunta sobre el video:")
    
    if st.button("Obtener respuesta") and query:
        response = get_response_from_query(db, query)
        st.write("Respuesta:", response)