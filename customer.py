import streamlit as st
from docx import Document
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
import PyPDF2

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
INDEX_NAME = "gradient-cyber"

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to the Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

# System Instruction for the AI
SYSTEM_INSTRUCTION = """You are an AI assistant. Provide only accurate answers based on the given context. Do not make assumptions."""

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return [("PDF Document", text)]

# Function to count tokens
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to get embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def upsert_to_pinecone(text, source):
    chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]  # Split into 8000 character chunks
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        metadata = {
            "source": source,
            "text": chunk
        }
        vector_id = f"{source}_{i}"
        vectors.append((vector_id, embedding, metadata))
    index.upsert(vectors=vectors)
    time.sleep(1)

# Function to query Pinecone
def query_pinecone(query, top_k=10):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = []
    for match in results['matches']:
        if 'text' in match['metadata']:
            contexts.append(match['metadata']['text'])
        else:
            contexts.append(f"Content from {match['metadata'].get('source', 'unknown source')}")
    return " ".join(contexts)

def get_answer(query):
    context = query_pinecone(query)
    
    # Prepare the messages for the LLM call
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Query: {query}\n\nContext: {context}"}
    ]
    
    # Make a single call to the LLM
    response = client.chat.completions.create(
        model="gpt-4o",  # Ensure this model can handle your maximum context length
        messages=messages,
        max_tokens=1000  # Adjust as needed
    )
    
    return response.choices[0].message.content.strip()

# Streamlit Interface
st.set_page_config(page_title="Gradient Cyber Customer Touchpoint Assistant", layout="wide")
st.title("Gradient Cyber Customer Touchpoint Assistant")
st.markdown("Welcome to the Gradient Cyber Customer Touchpoint Assistant! I'm here to help you analyze customer interactions and data. Feel free to ask me a question below.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Customer Touchpoint Data")
    uploaded_file = st.file_uploader("Upload the PDF file", type="pdf")
    if uploaded_file:
        texts = extract_text_from_pdf(uploaded_file)
        total_token_count = 0
        for source, text in texts:
            token_count = num_tokens_from_string(text)
            total_token_count += token_count
            # Upsert to Pinecone
            upsert_to_pinecone(text, source)
            st.text(f"Uploaded: {source}")
        st.subheader("Uploaded Documents")
        st.text(f"Total token count: {total_token_count}")

# Main content area
st.header("Ask Your Question")
user_query = st.text_input("What would you like to know about Gradient Cyber's customer touchpoints?")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Generating answer..."):
            answer = get_answer(user_query)
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question before searching.")
