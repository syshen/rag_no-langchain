import logging
from typing import List
import tiktoken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text (str): The input text
        model (str): The model to use for tokenization (default: "gpt-3.5-turbo")
        
    Returns:
        int: Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        raise

def split_text_by_tokens(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    model: str = "gpt-3.5-turbo"
) -> List[str]:
    """
    Split a text string into chunks based on token count with optional overlap.
    
    Args:
        text (str): The input text to split
        max_tokens (int): Maximum number of tokens per chunk (default: 500)
        overlap_tokens (int): Number of tokens to overlap between chunks (default: 50)
        model (str): The model to use for tokenization (default: "gpt-3.5-turbo")
        
    Returns:
        List[str]: List of text chunks
        
    Raises:
        ValueError: If max_tokens is less than overlap_tokens
        Exception: For other tokenization errors
    """
    try:
        if max_tokens <= overlap_tokens:
            raise ValueError("max_tokens must be greater than overlap_tokens")
            
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        chunks = []
        
        if len(tokens) <= max_tokens:
            return [text]
            
        start = 0
        while start < len(tokens):
            # Get the tokens for this chunk
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            
            # Decode the chunk back to text
            chunk_text = encoding.decode(chunk_tokens)
            
            # If this isn't the last chunk and there's more text to process
            if end < len(tokens):
                # Find the last period or newline in the chunk to create a clean break
                last_period = max(chunk_text.rfind('. '), chunk_text.rfind('\n'))
                if last_period != -1:
                    chunk_text = chunk_text[:last_period + 1]
                    # Recalculate the actual tokens used
                    chunk_tokens = encoding.encode(chunk_text)
                    end = start + len(chunk_tokens)
            
            chunks.append(chunk_text)
            
            # Move the start position for the next chunk, accounting for overlap
            start = end - overlap_tokens
            
        logger.info(f"Split text into {len(chunks)} chunks with max {max_tokens} tokens each")
        return chunks
        
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        raise
