import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI, RateLimitError
import tiktoken  # For token counting and chunking
from typing import List, Dict
import time
import logging
import json

from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment or .env file")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Directory with your 1377 text files
TEXT_FILES_DIR = "C:/healthflex/mcp server/texts"  # Update this path

# Chroma DB persistence path
CHROMA_DB_PATH = "C:/healthflex/mcp server/vector_db"  # Will create/store DB here

# Progress JSON path
PROGRESS_JSON_PATH = "C:/healthflex/mcp server/progress.json"

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# Max tokens per chunk (OpenAI limit is ~8192; we use 200 for smaller chunks)
MAX_TOKENS = 200
ENCODING = tiktoken.get_encoding("cl100k_base")  # For token counting

# Rate limiting variables
call_count = 0
call_start_time = time.time()

def chunk_text(text: str) -> List[str]:
    """Chunk text into pieces under MAX_TOKENS."""
    tokens = ENCODING.encode(text)
    chunks = []
    for i in range(0, len(tokens), MAX_TOKENS):
        chunk_tokens = tokens[i:i + MAX_TOKENS]
        chunks.append(ENCODING.decode(chunk_tokens))
    return chunks

def get_embedding(text: str) -> List[float]:
    """Generate embedding using OpenAI with basic rate limiting."""
    global call_count, call_start_time
    call_count += 1
    current_time = time.time()
    if current_time - call_start_time >= 60:
        call_count = 1
        call_start_time = current_time
    elif call_count > 50:  # Stay under 60 calls/minute
        wait_time = 60 - (current_time - call_start_time) + 1
        logger.info(f"Rate limit approaching. Waiting {wait_time:.1f}s...")
        time.sleep(wait_time)
        call_count = 1    
        call_start_time = current_time
    time.sleep(1.2)  # ~50 calls/minute

    try:
        response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        logger.info(f"HTTP Request: POST https://api.openai.com/v1/embeddings 'HTTP/1.1 200 OK' for chunk")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to embed chunk: {str(e)}")
        return None

def get_processed_chunks(collection) -> set:
    """Get set of chunk IDs already processed."""
    try:
        results = collection.get(include=["metadatas"])
        return {id for id in results["ids"]}
    except Exception:
        return set()

def load_progress() -> Dict:
    """Load progress from JSON file, or return empty dict if not found."""
    if os.path.exists(PROGRESS_JSON_PATH):
        with open(PROGRESS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_progress(progress: Dict):
    """Save progress to JSON file."""
    with open(PROGRESS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=4)

def get_remaining_files(total_files, progress):
    """Calculate remaining files to process."""
    processed = sum(1 for f in progress.values() if f["processed"] == f["total"])
    return total_files - processed

def main():
    global call_count, call_start_time
    call_count = 0
    call_start_time = time.time()
    logger.info("🚀 Starting embedding process...")

    # Initialize Chroma client with OpenAI embedder
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=EMBEDDING_MODEL
        )
        collection = chroma_client.get_or_create_collection(
            name="text_files_embeddings",
            embedding_function=openai_ef
        )

        # Load existing progress
        progress = load_progress()
        processed_chunks = get_processed_chunks(collection)

        # Get total files
        total_files = len([f for f in os.listdir(TEXT_FILES_DIR) if f.endswith(".txt")])
        remaining = get_remaining_files(total_files, progress)
        logger.info(f"Total files: {total_files}, Remaining files: {remaining}")

        # Process each text file
        for filename in os.listdir(TEXT_FILES_DIR):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(TEXT_FILES_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                # Chunk if necessary
                chunks = chunk_text(text)
                total_chunks = len(chunks)
                progress[filename] = progress.get(filename, {"total": total_chunks, "processed": 0})
                all_chunks_processed = all(f"{filename}_chunk_{i}" in processed_chunks for i in range(total_chunks))
                if all_chunks_processed:
                    logger.info(f"Skipping {filename}: all {total_chunks} chunks already processed")
                    progress[filename]["processed"] = total_chunks
                    save_progress(progress)
                    remaining = get_remaining_files(total_files, progress)
                    logger.info(f"Remaining files: {remaining}")
                    continue

                for i, chunk in enumerate(chunks):
                    doc_id = f"{filename}_chunk_{i}"
                    if doc_id in processed_chunks:
                        logger.info(f"Skipping existing chunk: {doc_id}")
                        progress[filename]["processed"] += 1
                        continue
                    
                    # Generate embedding
                    embedding = get_embedding(chunk)
                    if embedding is None:
                        logger.warning(f"Skipping chunk {doc_id} due to embedding failure")
                        continue
                    
                    # Add to Chroma DB
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[{"filename": filename, "chunk_id": i}],
                        ids=[doc_id]
                    )
                    logger.info(f"Embedded and stored: {doc_id}")
                    progress[filename]["processed"] += 1
                save_progress(progress)
                remaining = get_remaining_files(total_files, progress)
                logger.info(f"Remaining files: {remaining}")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                save_progress(progress)
                remaining = get_remaining_files(total_files, progress)
                logger.info(f"Remaining files: {remaining}")
                continue
    except Exception as e:
        logger.error(f"Critical error in main process: {str(e)}")
        save_progress(progress)
        remaining = get_remaining_files(total_files, progress)
        logger.info(f"Remaining files: {remaining}")
        raise

    logger.info(f"✅ All 1377 files embedded and stored in vector DB at {CHROMA_DB_PATH}")

if __name__ == "__main__":
    main()