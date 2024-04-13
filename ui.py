import streamlit as st
from text_summ import extract_text_from_pdf, generate_summary_and_sentiment
from gtts import gTTS
import os

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-container {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .section-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .section-icon {
        margin-right: 1rem;
        font-size: 1.5rem;
    }
    .btn-primary {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .btn-primary:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to display section header
def section_header(icon, title):
    st.markdown(f'<div class="section-header"><span class="section-icon">{icon}</span><h2>{title}</h2></div>', unsafe_allow_html=True)

def main():
    st.title("Text Summarization and Sentiment Analysis")

    # Sidebar
    # st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload PDF file", type="pdf")

    # Main content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Text Extraction Section
    section_header("📄", "Text Extraction")
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        st.write(text)

        # Text Summarization Section
        section_header("📝", "Text Summarization")
        with st.spinner('Generating summary...'):
            summary, sentiment_label, sentiment_score = generate_summary_and_sentiment(text)
            st.write("Summary:", summary)

        # Audio Summarization Section
        section_header("🔊", "Audio Summarization")
        language = 'en'
        tts = gTTS(text=summary, lang=language, slow=False)
        tts.save("output.mp3")
        st.audio("output.mp3", format='audio/mp3')

        # Sentiment Analysis Section
        section_header("😊", "Sentiment Analysis")
        st.write("Overall Sentiment:", sentiment_label)
        st.write("Sentiment Score:", sentiment_score)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
