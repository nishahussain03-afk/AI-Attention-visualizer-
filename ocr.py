import os
import pytesseract


DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def configure_tesseract():
    """
    Configure Tesseract OCR for Windows.

    The function first checks the TESSERACT_CMD environment variable.
    If it is not set, it checks the common Windows installation path.
    Finally, it checks whether Tesseract is available through PATH.
    """
    custom_path = os.environ.get("TESSERACT_CMD")

    if custom_path and os.path.exists(custom_path):
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return custom_path

    if os.path.exists(DEFAULT_TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH
        return DEFAULT_TESSERACT_PATH

    return None


def get_tesseract_status():
    """
    Return a simple status dictionary for the Streamlit interface.
    """
    path = configure_tesseract()

    try:
        version = pytesseract.get_tesseract_version()
        return {
            "available": True,
            "path": path or "PATH",
            "version": str(version),
            "message": f"Tesseract detected: {version}"
        }
    except Exception:
        return {
            "available": False,
            "path": path,
            "version": None,
            "message": (
                "Tesseract was not found. Install Tesseract OCR and ensure "
                "tesseract.exe is in C:\\Program Files\\Tesseract-OCR\\ "
                "or add its location to the PATH environment variable."
            )
        }


def extract_text(image):
    """
    Extract text from a PIL image using Tesseract OCR.
    """
    configure_tesseract()
    return pytesseract.image_to_string(image)
