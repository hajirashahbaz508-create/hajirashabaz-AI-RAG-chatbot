from pathlib import Path

from chatbot import BASE_DIR, add_text_to_knowledge_base, extract_text

PDF_FILE = BASE_DIR / "python notes.pdf"


def run_ingestion():
    if not PDF_FILE.exists():
        raise FileNotFoundError(
            f"'{PDF_FILE.name}' was not found. Put your Python notes PDF in this project folder."
        )
    text = extract_text(PDF_FILE.name, PDF_FILE.read_bytes())
    count = add_text_to_knowledge_base(text, PDF_FILE.name)
    print(f"Ingestion complete. Added {count} chunks to retrieval_data.pkl.")


if __name__ == "__main__":
    run_ingestion()
