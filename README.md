# AI Attention Visualizer

A beginner-friendly AI application that combines OCR, text embeddings,
scaled dot-product attention, and Streamlit visualization.

## Project Flow

Image → OCR → Clean Words → Embeddings → Q/K/V Attention → Visualization

## Technologies

- Python
- Streamlit
- Pillow
- Tesseract OCR
- pytesseract
- Sentence Transformers
- NumPy

## Folder Structure

```text
AI-Attention-Visualizer/
│
├── app.py
├── ocr.py
├── embedding.py
├── attention.py
├── requirements.txt
└── README.md
```

## Python Environment

Python 3.11 is recommended for this project.

Create a virtual environment:

```powershell
py -3.11 -m venv venv
```

If PowerShell activation is blocked, you can directly use the venv Python executable:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If activation works:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Tesseract OCR

`pytesseract` is only the Python wrapper. Tesseract OCR itself must also
be installed on Windows.

Common path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

This project automatically checks the common path and the system PATH.

You can also set a custom path with the `TESSERACT_CMD` environment variable.

## Run

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Or, after activation:

```powershell
streamlit run app.py
```

## First Run

The `all-MiniLM-L6-v2` model may be downloaded on its first use.
An internet connection is required for the first model download.

## Features

- Upload JPG, JPEG, and PNG study-note images
- Extract text using OCR
- Clean and select words
- Generate 384-dimensional embeddings
- Create Query, Key and Value representations
- Apply scaled dot-product attention
- Display normalized attention bars
- Display a summary table
- Identify the highest calculated attention word
- Show useful OCR and processing errors

## Important Technical Note

This is an educational attention demonstration.

The project uses `all-MiniLM-L6-v2` to create embeddings for individual
words. It does not expose the pretrained Transformer's internal token-level
attention.

The Q, K and V projection matrices in `attention.py` are randomly initialized
with a fixed seed. Therefore, the highest-attention word should not be
interpreted as a trained model's semantic importance prediction.

## Testing

1. Upload a clear printed-English study-notes image.
2. Check whether OCR text appears.
3. Check that cleaned words appear in the attention section.
4. Check that every word receives a progress bar.
5. Check the attention summary table.
6. Check the highest calculated attention word.
7. Test a handwritten image and observe OCR limitations.
8. Test an image without readable text and verify the error message.
9. Test an image containing more than 20 usable words.

## Limitations

- OCR quality depends on image quality, font, orientation and language.
- Only the first 20 cleaned words are visualized.
- The embedding model is applied to individual words for simplicity.
- The Q/K/V matrices are not learned from a training dataset.
- The result is a demonstration of attention mechanics, not true Transformer
  internal attention.

## Future Enhancements

- Use a Transformer that exposes actual token-level attention.
- Add sentence summarization.
- Highlight words directly on the OCR text.
- Use OCR bounding boxes on the original image.
- Support Tamil and other languages.
- Add keyword extraction and topic classification.
- Download an attention report.
- Add a study-notes summarizer.
- Add question answering for uploaded notes.
