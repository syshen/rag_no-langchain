import logging
from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_text_file(file_path: str) -> str:
    """
    Read content from a single text file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Content of the text file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: For other reading errors
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logger.info(f"Successfully read file: {file_path}")
        return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def read_pdf_file(file_path: str) -> str:
    """
    Read content from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Content of the PDF file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: For other reading errors
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        logger.info(f"Successfully read PDF file: {file_path}")
        return text
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {str(e)}")
        raise

def read_text_files_from_directory(directory_path: str, file_extensions: List[str] = ['.txt', '.md', '.py', '.pdf']) -> Dict[str, str]:
    """
    Read all text files from a directory with specified extensions.
    
    Args:
        directory_path (str): Path to the directory
        file_extensions (List[str]): List of file extensions to include (default: ['.txt', '.md', '.py', '.pdf'])
        
    Returns:
        Dict[str, str]: Dictionary mapping file paths to their contents
        
    Raises:
        NotADirectoryError: If the directory doesn't exist
        Exception: For other reading errors
    """
    try:
        directory = Path(directory_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        file_contents = {}
        for ext in file_extensions:
            for file_path in directory.rglob(f"*{ext}"):
                try:
                    if file_path.suffix.lower() == '.pdf':
                        content = read_pdf_file(str(file_path))
                    else:
                        content = read_text_file(str(file_path))
                    file_contents[str(file_path)] = content
                except Exception as e:
                    logger.warning(f"Skipping file {file_path}: {str(e)}")
                    continue

        logger.info(f"Successfully read {len(file_contents)} files from {directory_path}")
        return file_contents
    except Exception as e:
        logger.error(f"Error reading directory {directory_path}: {str(e)}")
        raise
