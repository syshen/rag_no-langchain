# RAG Project (No LangChain)

This project implements a Retrieval-Augmented Generation (RAG) system without using LangChain. It uses the following core dependencies:

- OpenAI: For LLM capabilities
- Pinecone: Vector database for storing and retrieving embeddings
- Tiktoken: OpenAI's tokenizer
- Python-dotenv: For managing environment variables

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
```

## Usage

1. Index documents:
```bash
python main.py -l /path/to/documents
```

2. Ask a question:
```bash
python main.py "What is the best way to do great work?"
```

3. Both index and ask:
```bash
python main.py -l /path/to/documents "What is the best way to do great work?"
```

## Project Structure

The project consists of several key components:

- `main.py`: Core application logic and CLI interface
- `utils.py`: File reading utilities
- `tokenization.py`: Token-related functions for text processing
- `requirements.txt`: Project dependencies
- `.env.example`: Template for environment variables

## Features

- Document processing and chunking
- Semantic search using OpenAI embeddings
- Vector storage with Pinecone
- RAG-based question answering
- Source citations in responses
- Progress tracking for long operations

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
