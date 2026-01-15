import os
import shutil
import pytesseract


def configure_tesseract():
    """
    Auto-detect Tesseract OCR executable in a deployable way.
    Priority:
    1. System PATH
    2. Environment variable TESSERACT_CMD
    """

    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return

    env_path = os.getenv("TESSERACT_CMD")
    if env_path and os.path.exists(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path
        return

    raise RuntimeError(
        "Tesseract OCR not found.\n"
        "Install Tesseract and ensure it is in PATH, or set TESSERACT_CMD env variable.\n"
        "Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "Linux: sudo apt install tesseract-ocr\n"
        "macOS: brew install tesseract"
    )
