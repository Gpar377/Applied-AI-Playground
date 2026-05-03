# AI-Powered Resume Screening & Job Matching System

An intelligent NLP-based system that analyzes resumes and matches them with job descriptions using semantic similarity, TF-IDF vectorization, and skill extraction.

## Features

- **Resume Parsing**: Extract text from PDF, DOCX, and TXT files
- **NLP Preprocessing**: Tokenization, lemmatization, stopword removal
- **Skill Extraction**: Identify technical and role-based skills
- **Semantic Matching**: TF-IDF + Cosine Similarity scoring
- **Resume Ranking**: Rank multiple candidates by relevance
- **Explainable AI**: Show matched/missing skills and score breakdown
- **REST API**: FastAPI endpoints for integration
- **Web UI**: Streamlit interface for easy interaction

## Technologies

- Python 3.8+
- NLP: NLTK, spaCy
- ML: scikit-learn (TF-IDF, Cosine Similarity)
- API: FastAPI
- UI: Streamlit
- Text Extraction: PyPDF2, python-docx

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Run Streamlit UI
```bash
streamlit run app.py
```

### Run API Server
```bash
python api.py
```

API Endpoints:
- `POST /upload_resume` - Upload and parse resume
- `POST /match_score` - Calculate match score for single resume
- `POST /rank_resumes` - Rank multiple resumes

## Project Structure

```
resume_screening_system/
├── api.py              # FastAPI backend
├── app.py              # Streamlit UI
├── parser.py           # Resume text extraction
├── preprocessor.py     # NLP preprocessing pipeline
├── matcher.py          # Matching & ranking engine
└── requirements.txt    # Dependencies
```

## How It Works

1. **Text Extraction**: Parse resumes from various formats
2. **Preprocessing**: Clean and normalize text using NLP
3. **Skill Extraction**: Identify technical skills from predefined database
4. **Vectorization**: Convert text to TF-IDF vectors
5. **Similarity Scoring**: Calculate cosine similarity (60%) + skill match (40%)
6. **Ranking**: Sort candidates by final score
7. **Explainability**: Display matched/missing skills

## Sample Output

```
Resume: john_doe.pdf
Match Score: 87.5%
Matched Skills: python, machine learning, sql, docker
Missing Skills: kubernetes, aws
```

## Real-World Applications

- Applicant Tracking Systems (ATS)
- HR Tech Platforms
- Recruitment Automation
- Talent Acquisition Tools

## Author

Built as an intermediate AI/ML project demonstrating NLP, semantic similarity, and production-ready system design.
