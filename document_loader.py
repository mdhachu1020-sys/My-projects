import os
import re
from exceptions import UnsupportedFileFormatError

SUPPORTED_FORMATS = {".pdf", ".docx", ".txt"}

class DocumentLoader:
    def supported_formats(self):
        return list(SUPPORTED_FORMATS)

    def load(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise UnsupportedFileFormatError(
                f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext == ".docx":
            return self._load_docx(file_path)
        elif ext == ".txt":
            return self._load_txt(file_path)

    def _load_pdf(self, path):
        import fitz
        doc = fitz.open(path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return self._normalize(text)

    def _load_docx(self, path):
        from docx import Document
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return self._normalize(text)

    def _load_txt(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return self._normalize(f.read())

    def _normalize(self, text):
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
