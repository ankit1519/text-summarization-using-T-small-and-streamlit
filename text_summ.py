from transformers import T5Tokenizer, T5ForConditionalGeneration,pipeline
import PyPDF2




def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text


'''pdf_path = "C:/farewell/rabbit.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
print(pdf_text)'''


def generate_summary_and_sentiment(input_text, chunk_size=512):
    # Split the input text into chunks of specified size
    chunks = [input_text[i:i + chunk_size] for i in range(0, len(input_text), chunk_size)]

    # Initialize variables to store results
    combined_summary = ''
    average_sentiment_score = 0.0

    # Sentiment analysis pipeline
    sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

    # Choose a Text Summarization Model
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Analyze sentiment and generate summary for each chunk
    for chunk in chunks:
        # Analyze sentiment of the current chunk
        sentiment_result = sentiment_analyzer(chunk)[0]
        sentiment_score = sentiment_result['score']
        average_sentiment_score += sentiment_score

        # Tokenize the current chunk
        tokenized_input = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)

        # Generate Summarization for the current chunk
        summary_ids = model.generate(tokenized_input, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Combine summaries from different chunks
        combined_summary += summary

    # Calculate overall sentiment based on the average scoreṇ
    average_sentiment_score /= len(chunks)
    overall_sentiment_label = 'positive' if average_sentiment_score >= 0.5 else 'negative'

    return combined_summary, overall_sentiment_label, average_sentiment_score




#pdf_path = "C:/farewell/rabbit.pdf"
#pdf_text = extract_text_from_pdf(pdf_path)
#summary, sentiment_label, sentiment_score = generate_summary_and_sentiment(pdf_text)
#print(f"Summary: {summary}\nSentiment: {sentiment_label} (Score: {sentiment_score:.2f})")




'''from gtts import gTTS
import os
language = 'en'

# Passing the text and language to the engine
tts = gTTS(text=summary, lang=language, slow=False)

# Saving the converted audio in a file
tts.save("output.mp3")

# Playing the converted file
os.system("start output.mp3")'''