import whisper
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embeddings and text splitter
embeddings = OllamaEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Function to transcribe video
def transcribe_video(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result['text']

# Main execution
try:
    # Specify the path of the video file
    video_path = input("Enter the path to the video file (e.g., video.mp4): ")

    # Transcribe the video
    print("Transcribing video...")
    transcript = transcribe_video(video_path)
    print("Transcription complete.")

    # Process the transcript
    documents = [Document(page_content=transcript)]  # Structure transcript for compatibility
    chunks = text_splitter.split_documents(documents)

    # Create FAISS vector store
    vectors = FAISS.from_documents(chunks, embeddings)
    retriever = vectors.as_retriever()

    # Prepare the language model and prompt template
    from langchain.llms.groq import ChatGroq  # Ensure this import is correct and available
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. Based on the video transcript provided,
        answer the following question with a clear and concise response.

        Context:
        {context}

        Question:
        {input}

        Answer the question clearly and succinctly:
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Input a question
    question = input("Enter your question: ")

    # Process the query
    response = retrieval_chain.invoke({"input": question})
    print("\nHere's what I've found based on the video:")
    print(response['answer'])

    # Display related sections from the transcript for transparency
    print("\nRelevant Transcript Sections:")
    for doc in response.get('context', []):
        print(doc['page_content'])  # Access 'page_content' instead of 'text'

except Exception as e:
    print(f"An error occurred: {e}")
