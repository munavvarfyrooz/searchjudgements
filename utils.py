import os
from typing import List
import pdfplumber
import pytesseract
from PIL import Image

def get_pdf_paths(directory: str) -> List[str]:
    """
    Retrieve all PDF file paths from the given directory and its subdirectories.
    
    Args:
        directory (str): The path to the directory containing PDFs.
    
    Returns:
        List[str]: List of PDF file paths.
    """
    pdf_paths = []
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(root, file))
        return pdf_paths
    except Exception as e:
        print(f"Error retrieving PDF paths from {directory}: {e}")
        return []

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file. Uses pdfplumber for text extraction,
    falls back to pytesseract OCR if no text is found on a page.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 0:
                    text += page_text + "\n\n"
                else:
                    # Fallback to OCR
                    try:
                        # Convert page to image
                        page_image = page.to_image(resolution=300).original
                        # Improved OCR with custom config for better accuracy
                        custom_config = r'--oem 3 --psm 3'
                        ocr_text = pytesseract.image_to_string(page_image, lang='eng', config=custom_config)
                        text += ocr_text + "\n\n"
                    except Exception as ocr_error:
                        print(f"OCR error on page {page.page_number} of {pdf_path}: {ocr_error}")
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text.strip()