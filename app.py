import streamlit as st
import numpy as np
from PIL import Image

from ocr import extract_text, get_tesseract_status
from embedding import create_embeddings
from attention import calculate_attention

st.set_page_config(
    page_title="AI Attention Visualizer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Attention Visualizer")
st.write(
    "Upload a study-notes image to extract text using OCR, "
    "convert the words into embeddings, and visualize a basic "
    "scaled dot-product attention mechanism."
)

with st.expander("ℹ️ How this project works"):
    st.write("Image → OCR → Clean Words → Embeddings → Q/K/V Attention → Visualization")

    st.markdown("""
    **OCR:** Converts text in an image into machine-readable text.

    **Embedding:** Converts each selected word into a numerical vector.

    **Attention:** Creates Query, Key and Value representations and calculates
    scaled dot-product attention.

    **Visualization:** Displays relative attention scores using progress bars.
    """)

status = get_tesseract_status()

if not status["available"]:
    st.warning(
        "Tesseract OCR was not detected. Install Tesseract OCR and make sure "
        "tesseract.exe is available before uploading an image."
    )
    st.code(status["message"])

uploaded_file = st.file_uploader(
    "📷 Upload a study-notes image",
    type=["jpg", "jpeg", "png"],
    help="Use a clear image containing printed English text."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.subheader("🖼️ Uploaded Image")
        st.image(image, use_container_width=True)

        if not status["available"]:
            st.error("OCR cannot start because Tesseract OCR is not configured.")
            st.stop()

        with st.spinner("Reading text from the image..."):
            text = extract_text(image)

        st.subheader("📝 Extracted Text")

        if not text.strip():
            st.error(
                "No readable text was found. Try a clearer image with larger "
                "printed text and better lighting."
            )
            st.stop()

        st.text_area(
            "OCR Result",
            text.strip(),
            height=180
        )

        words = []
        for word in text.split():
            cleaned = word.strip(".,!?;:()[]{}\"'`")
            if len(cleaned) > 2:
                words.append(cleaned)

        words = words[:20]

        if not words:
            st.error("No suitable words were found for attention visualization.")
            st.stop()

        st.info(
            f"Using {len(words)} cleaned word(s) for the classroom attention demonstration. "
            "A maximum of 20 words is used."
        )

        with st.spinner("Generating embeddings..."):
            embeddings = create_embeddings(words)

        with st.spinner("Calculating attention..."):
            scores = calculate_attention(embeddings)

        st.subheader("🧠 Word Attention")

        max_score = float(np.max(scores))

        if max_score <= 0:
            display_scores = np.zeros_like(scores, dtype=float)
        else:
            display_scores = scores / max_score

        for word, score in zip(words, display_scores):
            st.write(f"**{word}** — {float(score):.4f}")
            st.progress(float(np.clip(score, 0.0, 1.0)))

        top_index = int(np.argmax(scores))
        top_word = words[top_index]

        st.success(f"⭐ Highest Calculated Attention: **{top_word}**")

        st.subheader("📊 Attention Summary")

        summary_data = {
            "Word": words,
            "Attention Score": [round(float(score), 6) for score in scores],
            "Display Score": [round(float(score), 4) for score in display_scores]
        }

        st.dataframe(
            summary_data,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Educational note: the Q, K and V projection matrices are randomly "
            "initialized. Therefore, the highest-attention word is not a reliable "
            "measure of semantic importance."
        )

    except Exception as error:
        st.error("Something went wrong while processing the image.")
        st.exception(error)
