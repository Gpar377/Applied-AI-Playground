import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def rank_resumes(job_description, resumes):
    job_desc_processed = preprocess_text(job_description)
    resumes_processed = [preprocess_text(resume) for resume in resumes]

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform([job_desc_processed] + resumes_processed)

    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_indices = cosine_similarities.argsort()[::-1]
    return ranked_indices, cosine_similarities

st.title("Resume Scanner using NLP")

job_description = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader("Upload Resumes (txt files)", accept_multiple_files=True, type=['txt'])

if st.button("Rank Resumes"):
    if not job_description:
        st.error("Please enter a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        resumes = [file.getvalue().decode("utf-8") for file in uploaded_files]
        ranked_indices, scores = rank_resumes(job_description, resumes)

        st.write("### Ranked Resumes:")
        for rank, idx in enumerate(ranked_indices, start=1):
            st.write(f"**Rank {rank}** - Similarity Score: {scores[idx]:.2f}")
            st.text_area(f"Resume {rank}", resumes[idx], height=200)
