# import os
# import whisper
# import re
# import google.generativeai as genai
# import chromadb
# from chromadb import Documents, EmbeddingFunction, Embeddings
# from typing import List
# os.environ["GEMINI_API_KEY"]="AIzaSyAiKL1IsQgrAcNTSzSF-iPIh6smDEyfwJM"

# def transcribe_video(video_path):
#     model = whisper.load_model("base")
#     result = model.transcribe(video_path)
#     return result['text']

# def split_text(text: str):
#     """
#     Splits a text string into a list of non-empty substrings based on the specified pattern.
#     The "\n \n" pattern will split the document para by para
#     Parameters:
#     - text (str): The input text to be split.

#     Returns:
#     - List[str]: A list containing non-empty substrings obtained by splitting the input text.

#     """
#     split_text = re.split('\n \n', text)
#     return [i for i in split_text if i != ""]



# class GeminiEmbeddingFunction(EmbeddingFunction):
#     """
#     Custom embedding function using the Gemini AI API for document retrieval.

#     This class extends the EmbeddingFunction class and implements the __call__ method
#     to generate embeddings for a given set of documents using the Gemini AI API.

#     Parameters:
#     - input (Documents): A collection of documents to be embedded.

#     Returns:
#     - Embeddings: Embeddings generated for the input documents.
#     """
#     def __call__(self, input: Documents) -> Embeddings:
#         gemini_api_key = os.getenv("GEMINI_API_KEY")
#         if not gemini_api_key:
#             raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
#         genai.configure(api_key=gemini_api_key)
#         model = "models/embedding-001"
#         title = "Custom query"
#         return genai.embed_content(model=model,
#                                    content=input,
#                                    task_type="retrieval_document",
#                                    title=title)["embedding"]
    

# def create_chroma_db(documents:List, path:str, name:str):
#     """
#     Creates a Chroma database using the provided documents, path, and collection name.

#     Parameters:
#     - documents: An iterable of documents to be added to the Chroma database.
#     - path (str): The path where the Chroma database will be stored.
#     - name (str): The name of the collection within the Chroma database.

#     Returns:
#     - Tuple[chromadb.Collection, str]: A tuple containing the created Chroma Collection and its name.
#     """
#     chroma_client = chromadb.PersistentClient(path=path)
#     db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

#     for i, d in enumerate(documents):
#         db.add(documents=d, ids=str(i))

#     return db, name

# def load_chroma_collection(path, name):
#     """
#     Loads an existing Chroma collection from the specified path with the given name.

#     Parameters:
#     - path (str): The path where the Chroma database is stored.
#     - name (str): The name of the collection within the Chroma database.

#     Returns:
#     - chromadb.Collection: The loaded Chroma Collection.
#     """
#     chroma_client = chromadb.PersistentClient(path=path)
#     db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

#     return db

# def get_relevant_passage(query, db, n_results):
#   passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
#   return passage

# def make_rag_prompt(query, relevant_passage):
#   escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
#   prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
#   Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
#   However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
#   strike a friendly and converstional tone. \
#   If the passage is irrelevant to the answer, you may ignore it.
#   QUESTION: '{query}'
#   PASSAGE: '{relevant_passage}'

#   ANSWER:
#   """).format(query=query, relevant_passage=escaped)

#   return prompt

# def generate_prompt(prompt):
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     if not gemini_api_key:
#         raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
#     genai.configure(api_key=gemini_api_key)
#     model = genai.GenerativeModel('gemini-pro')
#     answer = model.generate_content(prompt)
#     return answer.text

# def generate_answer(db,query):
#     #retrieve top 3 relevant text chunks
#     relevant_text = get_relevant_passage(query,db,n_results=3)
#     prompt = make_rag_prompt(query, 
#                              relevant_passage="".join(relevant_text)) # joining the relevant chunks to create a single passage
#     answer = generate_prompt(prompt)

#     return answer



# # Specify the path of the video file
# video_path = input("Enter the path to the video file (e.g., video.mp4): ")

# # Transcribe the video
# print("Transcribing video...")
# transcript = transcribe_video(video_path)
# print("Transcription complete.")

# chunked_text = split_text(text=transcript)

# # db,name =create_chroma_db(documents=chunked_text, 
# #                     path="/Users/ravlasya/Downloads/", #replace with your path
# #                     name="rag_exp")



# db=load_chroma_collection(path="/Users/ravlasya/Downloads", #replace with path of your persistent directory
#                     name="rag_experiment") #replace with the collection name

# answer = generate_answer(db,query="Give the pointers so that i will know what all information is present in the text ?")
# print(answer)

import os
import whisper
import re
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List
os.environ["GEMINI_API_KEY"] = "AIzaSyAiKL1IsQgrAcNTSzSF-iPIh6smDEyfwJM"

def transcribe_video(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result['text']

def split_text(text: str):
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]

def create_chroma_db(documents: List, path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))

    return db, name

def load_chroma_collection(path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    return db

def get_relevant_passage(query, db, n_results):
    passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    return passage

def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and converstional tone. \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)

    return prompt

def generate_prompt(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

def generate_answer(db, query):
    relevant_text = get_relevant_passage(query, db, n_results=3)
    prompt = make_rag_prompt(query, "".join(relevant_text))  # joining the relevant chunks to create a single passage
    answer = generate_prompt(prompt)
    return answer

# Main loop for continuous querying
def main():
    video_path = input("Enter the path to the video file (e.g., video.mp4): ")

    print("Transcribing video...")
    transcript = transcribe_video(video_path)
    print("Transcription complete.")

    chunked_text = split_text(text=transcript)

    db = load_chroma_collection(path="/Users/ravlasya/Downloads", name="rag_experiment")

    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting...")
            break
        print("\nGenerating answer...")
        answer = generate_answer(db, query)
        print("\nAnswer: ", answer)

if __name__ == "__main__":
    main()




    