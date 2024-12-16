import os
import whisper
import re
import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List

# Set your Gemini API key
os.environ["GEMINI_API_KEY"] = ""  # Replace with your actual key

def transcribe_video(video_path):
    print("Starting transcription...")
    try:
        model = whisper.load_model("base")
        result = model.transcribe(video_path)
        print("Transcription completed successfully.")
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise

def split_text(text: str):
    print("Splitting transcript into chunks...")
    try:
        split_text = re.split('\n \n', text)
        return [i for i in split_text if i != ""]
    except Exception as e:
        print(f"Error during text splitting: {str(e)}")
        raise

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        try:
            embeddings = genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]
            print("Embedding generated successfully.")
            return embeddings
        except Exception as e:
            print(f"Error during embedding generation: {str(e)}")
            raise

# def create_chroma_db(documents: List, path: str, name: str):
#     print(f"Creating Chroma DB at path: {path}, with collection name: {name}")
#     chroma_client = chromadb.PersistentClient(path=path)
#     try:
#         db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
#         for i, d in enumerate(documents):
#             db.add(documents=d, ids=str(i))
#         print("Chroma DB created successfully.")
#         return db, name
#     except Exception as e:
#         print(f"Error during Chroma DB creation: {str(e)}")
#         raise
def create_chroma_db(documents: List, path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))
        print(f"Document {i} added to Chroma DB.")  # Debug statement

    return db, name


# def load_chroma_collection(path, name):
#     print(f"Loading Chroma collection: {name} from path: {path}")
#     try:
#         chroma_client = chromadb.PersistentClient(path=path)
#         db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
#         print("Chroma collection loaded successfully.")
#         return db
#     except Exception as e:
#         print(f"Error during Chroma collection loading: {str(e)}")
#         raise

def load_chroma_collection(path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    collection = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    print(f"Chroma collection '{name}' loaded, contains {len(collection.get())} documents.")  # Debug statement
    return collection

# def get_relevant_passage(query, db, n_results):
#     print(f"Querying for relevant passages with query: {query}")
#     try:
#         passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
#         print(f"Found relevant passage: {passage}")
#         return passage
#     except Exception as e:
#         print(f"Error during passage retrieval: {str(e)}")
#         raise

def get_relevant_passage(query, db, n_results):
    passages = db.query(query_texts=[query], n_results=n_results)['documents']
    print(f"Retrieved passages: {passages}")  # Debug statement
    return passages[0] if passages else ""


def make_rag_prompt(query, relevant_passage):
    print(f"Creating prompt for query: {query} with relevant passage.")
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. \
    If the passage is irrelevant to the answer, you may ignore it.\n\n    QUESTION: '{query}'\n    PASSAGE: '{relevant_passage}'\n\n    ANSWER:\n    """).format(query=query, relevant_passage=escaped)
    return prompt

# def generate_prompt(prompt):
#     print(f"Generating content from prompt: {prompt[:100]}...")  # Display only the first 100 chars for debugging
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     if not gemini_api_key:
#         raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
#     genai.configure(api_key=gemini_api_key)
#     model = genai.GenerativeModel('gemini-pro')
#     try:
#         answer = model.generate_content(prompt)
#         print("Prompt generated successfully.")
#         return answer.text
#     except Exception as e:
#         print(f"Error during prompt generation: {str(e)}")
#         raise

def generate_prompt(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided.")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    print(f"Generated prompt: {prompt}")  # Debug statement
    print(f"Model response: {answer.text}")  # Debug statement
    return answer.text

# def generate_answer(db, query):
#     print(f"Generating answer for query: {query}")
#     try:
#         relevant_text = get_relevant_passage(query, db, n_results=3)
#         prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))
#         return generate_prompt(prompt)
#     except Exception as e:
#         print(f"Error during answer generation: {str(e)}")
#         raise

def generate_answer(db, query):
    relevant_text = get_relevant_passage(query, db, n_results=3)
    if not relevant_text:
        print(f"No relevant text found for query: {query}")  # Debug statement
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))
    answer = generate_prompt(prompt)
    print(f"Answer generated: {answer}")  # Debug statement
    return answer


# Streamlit app
st.title("Video-Based Chatbot")
st.sidebar.header("Options")
option = st.sidebar.radio("Provide Input:", ["Upload File", "Provide URL"])

if 'video_path' not in st.session_state:
    st.session_state.video_path = None  # Initialize the video path
if 'db' not in st.session_state:
    st.session_state.db = None

if option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload your video file", type=["mp4", "mkv", "avi"])
    if uploaded_file is not None:
        video_path = f"temp_{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.sidebar.success("File uploaded successfully!")
        st.session_state.video_path = video_path 

elif option == "Provide URL":
    video_url = st.sidebar.text_input("Enter video URL")
    if video_url:
        video_path = video_url  # Handle URL-based download (extend with downloader)
        st.sidebar.success("URL provided successfully!")
        st.session_state.video_path = video_path

if st.sidebar.button("Launch Chatbot"):
   try:
        video_path = st.session_state.video_path  # Retrieve video path from session_state
        if not video_path:
            st.error("Please provide a valid video file or URL.")
        else:
            st.info("Processing video, please wait...")

            # Transcribe video
            transcript = transcribe_video(video_path)
            chunked_text = split_text(transcript)

            # Chroma DB setup
            db_path = "/Users/ravlasya/Downloads"  # Persistent storage path
            collection_name = "rag_experiment"

            if st.session_state.db is None:
                try:
                    db = load_chroma_collection(path=db_path, name=collection_name)
                    st.session_state.db = db
                except Exception:
                    db, name = create_chroma_db(documents=chunked_text, path=db_path, name=collection_name)
                    st.session_state.db = db

            # try:
            #     db = load_chroma_collection(path=db_path, name=collection_name)
            # except Exception:
            #     db, name = create_chroma_db(documents=chunked_text, path=db_path, name=collection_name)
            #     db = load_chroma_collection(path=db_path, name=collection_name)

            st.success("Chatbot ready! Ask your questions below.")

            user_query = st.text_input("Enter your question:")
            print(f"User query: {user_query}")  # Debugging line

            if user_query:
                st.write(f"User query: {user_query}")  # Add a debug line to see if the input is being capture
                st.info("Generating answer...")
                print("Generating answer...")  # Debugging line
                answer = generate_answer(db, query=user_query)
                print(f"Generated answer: {answer}")  # Debugging line
                st.write(answer)
            else:
                print("No query entered.")  # Debugging line

    #         if user_query:
    #             st.info("Generating answer...")
    #             answer = generate_answer(db, query=user_query)
    #             st.write(answer)
   except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")
