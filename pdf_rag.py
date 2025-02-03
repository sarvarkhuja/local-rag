import streamlit as st
import os
from typing import List, Tuple
from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document

# Constants
PDFS_DIRECTORY = Path('./pdfs')
CHUNK_SIZE = 1200  # Increased chunk size for better context
CHUNK_OVERLAP = 300  # Increased overlap for better continuity
EMBED_MODEL = "nomic-embed-text"  # Specialized embedding model
LLM_MODEL = "deepseek-coder:14b"  # LLM for text generation
TOP_K = 5  # Number of document chunks to retrieve

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(
        OllamaEmbeddings(model=EMBED_MODEL)  # Use dedicated embedding model
    )

# Cache model initialization
@st.cache_resource
def get_llm():
    return OllamaLLM(model=LLM_MODEL)

# Improved prompt template with clearer instructions
@st.cache_data
def get_prompt_template():
    template = """You are a professional document analysis assistant. Answer the question using ONLY the provided context. 
If the information is not explicitly in the context, state "This information is not available in the document."

Question: {question}

Context excerpts from the document:
{context}

Provide a precise answer based on the context."""

    return ChatPromptTemplate.from_template(template)

def split_text(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with improved parameters."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]  # Better content-aware splitting
    )
    return text_splitter.split_documents(documents)

def process_documents(documents: List[Document]):
    """Process documents with enhanced text splitting and metadata."""
    with st.spinner('Processing documents...'):
        chunked_documents = split_text(documents)
        
        # Add document metadata to chunks
        for i, doc in enumerate(chunked_documents):
            doc.metadata["chunk_id"] = i
            doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "unknown"))
        
        st.session_state.vector_store.add_documents(chunked_documents)

def answer_question(question: str) -> Tuple[str, List[Document]]:
    """Generate answer with improved context retrieval."""
    try:
        with st.spinner('Analyzing documents...'):
            # Retrieve more relevant chunks with score threshold
            documents = st.session_state.vector_store.similarity_search_with_score(
                question, 
                k=TOP_K,
                score_threshold=0.75  # Filter out low-confidence matches
            )
            # Filter and unpack documents
            filtered_docs = [doc for doc, score in documents if score < 0.75]
            
        if not filtered_docs:
            return "No relevant information found in documents.", []

        # Replace spinner with streaming response
        message_placeholder = st.empty()
        full_response = ""
        
        context = "\n\n---\n".join([
            f"Excerpt {i+1} (from {doc.metadata['source']}):\n{doc.page_content}"
            for i, doc in enumerate(filtered_docs)
        ])
        
        prompt = get_prompt_template()
        chain = prompt | get_llm()
        
        # Stream the response
        for chunk in chain.stream({"question": question, "context": context}):
            full_response += str(chunk)
            message_placeholder.write(full_response + "▌")
        
        # Write final response without cursor
        message_placeholder.write(full_response)
            
        return full_response, filtered_docs
            
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return f"Error: {str(e)}", []

# ... (rest of the functions remain similar with minor UI adjustments)
def get_existing_pdfs() -> List[Path]:
    """Get list of existing PDFs in directory with error handling."""
    try:
        PDFS_DIRECTORY.mkdir(exist_ok=True, parents=True)
        return list(PDFS_DIRECTORY.glob('*.pdf'))
    except Exception as e:
        st.error(f"Error accessing PDF directory: {str(e)}")
        return []

@st.cache_data(show_spinner=False)
def load_pdf(file_path: Path) -> List[Document]:
    """Load PDF with enhanced error handling and caching."""
    try:
        loader = PDFPlumberLoader(str(file_path))
        docs = loader.load()
        
        # Add source metadata to all documents
        for doc in docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["full_path"] = str(file_path)
            
        return docs
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {str(e)}")
        return []

def upload_pdf(file) -> Path:
    """Save uploaded PDF with conflict resolution."""
    PDFS_DIRECTORY.mkdir(exist_ok=True, parents=True)
    
    # Handle duplicate filenames
    original_name = file.name
    counter = 1
    while (PDFS_DIRECTORY / file.name).exists():
        name_parts = original_name.rsplit('.', 1)
        file.name = f"{name_parts[0]}_{counter}.{name_parts[1]}"
        counter += 1
    
    file_path = PDFS_DIRECTORY / file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def main():
    st.title("SYNEL AI chatbot")
    
    # Improved UI for document management
    with st.expander("Document Management"):
        existing_pdfs = get_existing_pdfs()
        if existing_pdfs:
            st.write("### Loaded Documents:")
            for pdf in existing_pdfs:
                st.write(f"- {pdf.name}")
            if st.button("Reprocess All Documents"):
                with st.spinner('Reloading documents...'):
                    all_documents = []
                    for pdf_path in existing_pdfs:
                        documents = load_pdf(pdf_path)
                        all_documents.extend(documents)
                    process_documents(all_documents)
                    st.success("Documents reprocessed successfully!")
    
    # Enhanced chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about the uploaded documents!"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "documents" in message:
                with st.expander("Sources"):
                    for doc in message["documents"]:
                        st.write(f"**Source**: {doc.metadata['source']}")
                        st.write(f"**Excerpt**: {doc.page_content[:300]}...")

    if question := st.chat_input():
        # Display user question
        with st.chat_message("user"):
            st.write(question)
            
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Generate and display answer
        with st.chat_message("assistant"):
            answer, source_docs = answer_question(question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "documents": source_docs
            })

if __name__ == "__main__":
    main()