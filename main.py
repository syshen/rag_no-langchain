import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import logging
from pathlib import Path
from typing import List, Dict, Optional
from utils import read_text_file, read_text_files_from_directory
from tokenization import count_tokens, split_text_by_tokens
import uuid
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_openai():
    """Initialize OpenAI client."""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        # Test the connection
        models = client.models.list()
        logger.info("Successfully connected to OpenAI")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        raise

def init_pinecone():
    """Initialize Pinecone client."""
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        # List indexes to verify connection
        indexes = pc.list_indexes()
        logger.info(f"Successfully connected to Pinecone. Available indexes: {indexes}")
        
        # Get the specified index
        index_name = os.getenv('PINECONE_INDEX_NAME')
        if index_name not in [idx.name for idx in indexes]:
            logger.error(f"Index {index_name} not found in available indexes: {indexes}")
            raise ValueError(f"Index {index_name} not found")
            
        index = pc.Index(index_name)
        logger.info(f"Successfully connected to index: {index_name}")
        return pc, index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {str(e)}")
        raise

def get_embeddings(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI's API.
    
    Args:
        client (OpenAI): OpenAI client
        texts (List[str]): List of texts to generate embeddings for
        model (str): Model to use for embeddings
        
    Returns:
        List[List[float]]: List of embeddings
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float"
        )
        embeddings = [data.embedding for data in response.data]
        logger.info(f"Generated embeddings for {len(texts)} texts using {model}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def process_and_index_documents(
    openai_client: OpenAI,
    pinecone_index,
    docs_path: str = "docs",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 100
):
    """
    Process documents from a directory and index them in Pinecone.
    
    Args:
        openai_client (OpenAI): OpenAI client
        pinecone_index: Pinecone index
        docs_path (str): Path to documents directory
        chunk_size (int): Maximum number of tokens per chunk
        chunk_overlap (int): Number of tokens to overlap between chunks
        batch_size (int): Number of embeddings to generate and upsert at once
    """
    try:
        # Read all documents
        docs = read_text_files_from_directory(docs_path)
        if not docs:
            logger.warning(f"No documents found in {docs_path}")
            return

        logger.info(f"Processing {len(docs)} documents")
        
        # Process documents in batches
        current_batch = []
        batch_metadata = []
        
        for filepath, content in tqdm(docs.items(), desc="Processing documents"):
            # Split the document into chunks
            chunks = split_text_by_tokens(
                content,
                max_tokens=chunk_size,
                overlap_tokens=chunk_overlap
            )
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Create metadata for the chunk
                metadata = {
                    "source": filepath,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                current_batch.append(chunk)
                batch_metadata.append(metadata)
                
                # If we've reached the batch size, process the batch
                if len(current_batch) >= batch_size:
                    embeddings = get_embeddings(openai_client, current_batch)
                    
                    # Prepare vectors for Pinecone
                    vectors = [
                        {
                            "id": str(uuid.uuid4()),
                            "values": embedding,
                            "metadata": {
                                **metadata,
                                "text": text
                            }
                        }
                        for embedding, metadata, text in zip(embeddings, batch_metadata, current_batch)
                    ]
                    
                    # Upsert to Pinecone
                    pinecone_index.upsert(vectors=vectors)
                    logger.info(f"Indexed batch of {len(vectors)} vectors")
                    
                    # Clear the batch
                    current_batch = []
                    batch_metadata = []
        
        # Process any remaining documents
        if current_batch:
            embeddings = get_embeddings(openai_client, current_batch)
            vectors = [
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {
                        **metadata,
                        "text": text
                    }
                }
                for embedding, metadata, text in zip(embeddings, batch_metadata, current_batch)
            ]
            pinecone_index.upsert(vectors=vectors)
            logger.info(f"Indexed final batch of {len(vectors)} vectors")
            
        logger.info("Document processing and indexing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing and indexing documents: {str(e)}")
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process and index documents for RAG system')
    parser.add_argument('-l', '--load',
                      help='Path to the documents directory to load and index',
                      type=str,
                      required=False)
    parser.add_argument('question',
                      help='Question to ask the AI',
                      type=str,
                      nargs='?',  # Makes the question optional
                      default=None)
    return parser.parse_args()

def get_relevant_context(
    openai_client: OpenAI,
    pinecone_index,
    question: str,
    num_results: int = 5
) -> List[str]:
    """
    Retrieve relevant context from Pinecone based on the question.
    
    Args:
        openai_client (OpenAI): OpenAI client
        pinecone_index: Pinecone index
        question (str): User's question
        num_results (int): Number of results to retrieve
        
    Returns:
        List[str]: List of relevant text chunks
    """
    try:
        # Generate embedding for the question
        question_embedding = get_embeddings(openai_client, [question])[0]
        
        # Query Pinecone
        query_response = pinecone_index.query(
            vector=question_embedding,
            top_k=num_results,
            include_metadata=True
        )
        
        # Log the number of matches from Pinecone
        logger.info(f"Received {len(query_response.matches)} matches from Pinecone")
        
        # Extract and sort the contexts by score
        contexts = []
        for match in query_response.matches:
            if match.score < 0.3:  # Skip if similarity is too low
                continue
            contexts.append({
                'text': match.metadata['text'],
                'score': match.score,
                'source': match.metadata['source']
            })
        
        # Sort by score
        contexts.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Retrieved {len(contexts)} relevant contexts")
        return contexts
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        raise

def generate_answer(
    openai_client: OpenAI,
    question: str,
    contexts: List[Dict],
    model: str = "gpt-4-turbo-preview"
) -> str:
    """
    Generate an answer using OpenAI's API with retrieved contexts.
    
    Args:
        openai_client (OpenAI): OpenAI client
        question (str): User's question
        contexts (List[Dict]): List of relevant contexts with scores
        model (str): Model to use for generation
        
    Returns:
        str: Generated answer
    """
    try:
        # Prepare context string
        context_str = "\n\n".join([
            f"[Source: {ctx['source']} (Relevance: {ctx['score']:.2f})]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful AI assistant that answers questions based on the provided context. 
Follow these rules:
1. Use ONLY the provided context to answer the question
2. If the context doesn't contain enough information to answer the question fully, say so
3. Always cite your sources when providing information
4. If different sources provide conflicting information, point this out
5. Be concise but thorough
6. Use markdown formatting for better readability

Here is the relevant context:
{context_str}"""
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        # Generate response
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        logger.info("Generated answer successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise

def main():
    """Main function to initialize all components and process documents."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    required_env_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise EnvironmentError(f"Missing required environment variables: {missing_vars}")

    try:
        # Initialize clients
        openai_client = init_openai()
        pinecone_client, pinecone_index = init_pinecone()
        
        logger.info("All clients initialized successfully")
        
        # Process and index documents if path is provided
        if args.load:
            docs_path = args.load
            if not os.path.isdir(docs_path):
                logger.error(f"The specified path '{docs_path}' is not a valid directory")
                raise NotADirectoryError(f"Invalid directory path: {docs_path}")
                
            logger.info(f"Processing documents from: {docs_path}")
            process_and_index_documents(
                openai_client=openai_client,
                pinecone_index=pinecone_index,
                docs_path=docs_path,
                chunk_size=500,
                chunk_overlap=50,
                batch_size=100
            )
        
        # Handle question if provided
        if args.question:
            logger.info(f"Processing question: {args.question}")
            
            # Get relevant context
            contexts = get_relevant_context(
                openai_client=openai_client,
                pinecone_index=pinecone_index,
                question=args.question
            )
            
            if not contexts:
                print("No relevant context found for your question.")
                return
            
            # Generate answer
            answer = generate_answer(
                openai_client=openai_client,
                question=args.question,
                contexts=contexts
            )
            
            # Print the answer
            print("\nAnswer:")
            print(answer)
        
        return openai_client, pinecone_client, pinecone_index
    
    except Exception as e:
        logger.error(f"Failed to initialize clients or process documents: {str(e)}")
        raise

if __name__ == "__main__":
    main()
