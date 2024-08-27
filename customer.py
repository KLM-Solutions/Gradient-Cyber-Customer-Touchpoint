import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import time
import json
import io

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
INDEX_NAME = "excel-data"
MAX_TOKENS_PER_CHUNK = 20000  # Adjust this value as needed

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

def chunk_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def upsert_to_pinecone(json_data, source):
    vectors = []
    vector_ids = []
    for customer_name, customer_data in json_data.items():
        text = json.dumps({customer_name: customer_data})
        chunks = chunk_text(text, MAX_TOKENS_PER_CHUNK)
        for j, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            chunk_source = f"{source}_{j+1}"
            metadata = {
                "source": chunk_source,
                "text": chunk,
                "chunk_index": j,
                "customer_name": customer_name
            }
            vector_id = f"{chunk_source}_{customer_name}"
            vectors.append((vector_id, embedding, metadata))
            vector_ids.append(vector_id)
    
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        index.upsert(vectors=batch)
        time.sleep(1)
    
    return vector_ids

def fetch_from_pinecone(vector_ids):
    fetch_response = index.fetch(ids=vector_ids)
    contexts = []
    for vector_id, vector_data in fetch_response['vectors'].items():
        if 'text' in vector_data['metadata']:
            contexts.append(vector_data['metadata']['text'])
        else:
            contexts.append(f"Content from {vector_data['metadata'].get('source', 'unknown source')}")
    return contexts

def get_answer(query, vector_ids):
    contexts = fetch_from_pinecone(vector_ids)
    full_context = " ".join(contexts)
    
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Query: {query}\n\nContext: {full_context}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4-0613",
        messages=messages,
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content.strip()
    return answer

# Streamlit Interface
st.set_page_config(page_title="Pinecone Multi-File Upserter and Querier", layout="wide")
st.title("Pinecone JSON Multi-File Upserter and Querier")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload JSON Data")
    uploaded_files = st.file_uploader("Upload your JSON files", type=["json"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Upsert to Pinecone"):
            with st.spinner("Upserting data to Pinecone..."):
                all_vector_ids = []
                for uploaded_file in uploaded_files:
                    try:
                        json_data = json.load(uploaded_file)
                        vector_ids = upsert_to_pinecone(json_data, uploaded_file.name)
                        all_vector_ids.extend(vector_ids)
                        st.success(f"Data from {uploaded_file.name} upserted successfully!")
                    except json.JSONDecodeError:
                        st.error(f"Invalid JSON file: {uploaded_file.name}. Skipping this file.")
                    except Exception as e:
                        st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")
                st.session_state['vector_ids'] = all_vector_ids
            st.success("All valid files have been processed and upserted to Pinecone.")

# Main content area
st.header("Query Your Data")
user_query = st.text_input("What would you like to know about?")

if st.button("Get Answer"):
    if user_query and 'vector_ids' in st.session_state:
        with st.spinner("Generating answer..."):
            answer = get_answer(user_query, st.session_state['vector_ids'])
            st.subheader("Answer:")
            st.write(answer)
    elif 'vector_ids' not in st.session_state:
        st.warning("Please upload and upsert data before querying.")
    else:
        st.warning("Please enter a question before searching.")
