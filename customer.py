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
SYSTEM_INSTRUCTION = """You are an AI assistant. Provide only accurate answers based on the given context. If the information is not available in the context, clearly state that. Do not make assumptions or provide information that is not supported by the given context."""

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
    for customer_name, customer_data in json_data.items():
        # Convert each customer's data to a string, including the customer name
        text = json.dumps({customer_name: customer_data})
        chunks = chunk_text(text, MAX_TOKENS_PER_CHUNK)
        for j, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            chunk_source = f"{source}_{j+1}"  # Include chunk number in source
            metadata = {
                "source": chunk_source,
                "text": chunk,
                "chunk_index": j,
                "customer_name": customer_name
            }
            vector_id = f"{chunk_source}_{customer_name}"
            vectors.append((vector_id, embedding, metadata))
    
    # Upsert in batches of 100
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        index.upsert(vectors=batch)
        time.sleep(1)  # To avoid hitting rate limits

# Improved function to query Pinecone
def improved_query_pinecone(query, top_k=20):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = []
    for match in results['matches']:
        if 'text' in match['metadata']:
            contexts.append({
                'text': match['metadata']['text'],
                'score': match['score'],
                'source': match['metadata'].get('source', 'unknown source')
            })
    
    # Sort contexts by relevance score
    contexts.sort(key=lambda x: x['score'], reverse=True)
    
    # Aggregate context, prioritizing higher-scored matches
    aggregated_context = "\n".join([f"Source: {c['source']} (Score: {c['score']:.2f})\n{c['text']}" for c in contexts])
    
    return aggregated_context, contexts

# Improved function to generate answers
def improved_get_answer(query):
    aggregated_context, contexts = improved_query_pinecone(query)
    
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Query: {query}\n\nContext: {aggregated_context}\n\nPlease provide a detailed and accurate answer based on the given context. If the information is not available in the context, clearly state that."}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",  # Use the most capable model available
        messages=messages,
        max_tokens=1000,
        temperature=0.2  # Lower temperature for more focused answers
    )
    
    answer = response.choices[0].message.content.strip()
    
    # Add source information to the answer
    answer += "\n\nSources: " + ", ".join(set([c['source'] for c in contexts if c['score'] > 0.7]))
    
    return answer

# Function to validate answers
def validate_answer(query, answer):
    validation_prompt = f"Query: {query}\n\nProposed Answer: {answer}\n\nPlease evaluate the above answer for accuracy and relevance to the query. If there are any inaccuracies or missing information, please provide corrections."
    
    validation_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a critical evaluator. Your task is to assess the accuracy and completeness of the given answer."},
            {"role": "user", "content": validation_prompt}
        ],
        max_tokens=500,
        temperature=0.2
    )
    
    validation_result = validation_response.choices[0].message.content.strip()
    return validation_result

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
                for uploaded_file in uploaded_files:
                    try:
                        json_data = json.load(uploaded_file)
                        upsert_to_pinecone(json_data, uploaded_file.name)
                        st.success(f"Data from {uploaded_file.name} upserted successfully!")
                    except json.JSONDecodeError:
                        st.error(f"Invalid JSON file: {uploaded_file.name}. Skipping this file.")
                    except Exception as e:
                        st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")
            st.success("All valid files have been processed and upserted to Pinecone.")

# Main content area
st.header("Query Your Data")
user_query = st.text_input("What would you like to know about?")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Generating answer..."):
            answer = improved_get_answer(user_query)
            validation = validate_answer(user_query, answer)
            
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Answer Validation:")
            st.write(validation)
    else:
        st.warning("Please enter a question before searching.")
