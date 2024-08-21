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
# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "gradient-cyber"

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
SYSTEM_INSTRUCTION = """You are an AI assistant specialized in customer touchpoint data for Gradient Cyber. Your knowledge comes from customer interaction records, meeting notes, and service details. Please adhere to the following guidelines:

1. Provide accurate information based on the customer touchpoint data available to you.
2. Maintain a professional and friendly tone, as if you're a customer success representative for Gradient Cyber.
3. When discussing specific customers:
   - Provide details on their products, services, and any recent interactions or issues.
   - Mention key contacts at the customer's organization when relevant.
   - Highlight any recent changes in their infrastructure or services.
4. For questions about Gradient Cyber's services:
   - Explain features like BiFlow, IDS, Firewall Active Response, and cloud integrations.
   - Discuss how these services are implemented for specific customers.
5. When asked about technical issues:
   - Provide context from relevant customer interactions.
   - Suggest troubleshooting steps based on past resolutions.
6. For questions about customer sentiment:
   - Use the sentiment information provided in the touchpoint data.
   - Provide a balanced view, mentioning both positive feedback and areas for improvement.
7. If asked about scheduling or future plans:
   - Reference any mentioned upcoming meetings, reviews, or planned changes.
8. When discussing surveys or assessments:
   - Mention specific tools like NIST, CMMC, or CAT surveys if they've been used.
   - Provide insights on how customers have used or plan to use these tools.
9. If information is not available in the provided context:
   - Clearly state that you don't have that specific information.
   - Offer to provide related information that is available, if applicable.
10. Respect customer privacy:
    - Do not share sensitive information like API keys or personal contact details.
    - Refer to individuals by their professional titles rather than personal names when appropriate.
11. For questions about Gradient Cyber's competitors or alternative services:
    - Provide factual information without making direct comparisons.
    - Focus on Gradient Cyber's strengths and unique offerings.
12. When discussing financial information:
    - Only mention details that are explicitly stated in the touchpoint data.
    - Do not speculate on pricing or contract values.

Your primary goal is to assist with understanding customer relationships, technical implementations, and service quality based on the touchpoint data. Provide insights that could help improve customer satisfaction and identify potential upsell opportunities."""

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return [("PDF Document", text)]

# Function to truncate text
def truncate_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

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
def query_pinecone(query, top_k=5):
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
    max_context_tokens = 4000
    truncated_context = truncate_text(context, max_context_tokens)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Query: {query}\n\nContext: {truncated_context}"}
        ]
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
