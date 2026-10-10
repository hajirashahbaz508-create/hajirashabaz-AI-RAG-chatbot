PYTHON_RAG_CHATBOT - SIMPLE SETUP

FILES
- app.py: Streamlit interface, chat, and file upload
- chatbot.py: Retrieval + Ollama generation + file text extraction
- ingest.py: Indexes the original python notes.pdf
- python notes.pdf: Starting knowledge source
- requirements.txt: Python packages

1. Open a terminal in this folder.
2. (Recommended) Create and activate a virtual environment:
   py -m venv .venv
   .venv\Scripts\activate
3. Install packages:
   python -m pip install -r requirements.txt
4. Make sure Ollama is installed and running. In a terminal run:
   ollama pull llama3.2:1b
   ollama pull nomic-embed-text
   ollama list
5. Build the initial searchable index from the supplied PDF:
   python ingest.py
6. Start the app:
   python -m streamlit run app.py

OFFLINE USE
After both models and Python packages have been downloaded/installed, Ollama inference and embedding generation run locally. You do not need internet for normal chat or for adding files. Do not remove the downloaded models.

UPLOADING FILES
Use the left sidebar to upload PDF, TXT, or DOCX, then click 'Add file to knowledge base'. The file is indexed locally and can be queried immediately. Scanned PDFs containing only images may need OCR; this version extracts selectable PDF text only.

IMPORTANT
This is RAG, not model fine-tuning. Uploaded text is chunked and embedded into retrieval_data.pkl; the Llama model's weights are not retrained.

IF YOU CHANGE EMBEDDING MODEL
Rebuild the index by deleting retrieval_data.pkl and running python ingest.py again. Use the same embedding model for all documents and questions.
