# AI-Powered Legal Judgements Search System (POC)

This is a proof-of-concept (POC) application for an AI-powered search system for legal judgements using Retrieval-Augmented Generation (RAG). It handles up to 50,000 PDFs, extracts text, indexes them, and provides a query interface via Streamlit.

**Disclaimer:** This is for testing purposes only. It may not handle all edge cases, and performance is optimized for CPU. Verify results and use at your own risk. Not intended for production use without further optimization.

## Features
- Ingest PDFs from a local directory.
- Text extraction with fallback to OCR for scanned documents.
- Document chunking and embedding.
- Hybrid retrieval (semantic + keyword) with reranking.
- Response generation using a local LLM.
- Streamlit web UI for interaction.

## Requirements
- Python 3.12
- Dependencies listed in `requirements.txt`

## Installation
1. Clone the repository or copy the project files to your local machine.
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Unix/Mac
   venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. (Optional) For OCR support, install Tesseract OCR and set the path if needed:
   - Download from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
   - Add to PATH or set `pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'`

## Usage
1. Place your PDF judgements in `./judgements_pdfs/` (or specify another directory in the UI).
2. Run the Streamlit app:
   ```
   streamlit run main.py
   ```
3. In the web interface:
   - Enter the PDF directory and click "Build Index" to create the FAISS index.
   - Enter a query in the text box to search and get responses with sources.

## Project Structure
- `requirements.txt`: List of dependencies.
- `main.py`: Streamlit app logic.
- `rag_pipeline.py`: RAG pipeline functions (index building, retrieval, querying).
- `utils.py`: Helper functions for PDF handling.
- `README.md`: This file.

## Notes
- Index is saved to `./faiss_index/` and can be reused.
- For large datasets, building the index may take time; it processes in batches with progress bars.
- Uses local models; ensure sufficient RAM/CPU for 50,000 PDFs.
- Error handling is included, but monitor console for issues.

For questions or improvements, refer to the code comments.