import PyPDF2
from docx import Document
import re

class ResumeParser:
    @staticmethod
    def extract_text_from_pdf(file_path):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join([page.extract_text() for page in reader.pages])
    
    @staticmethod
    def extract_text_from_docx(file_path):
        doc = Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])
    
    @staticmethod
    def extract_text(file_path):
        if file_path.endswith('.pdf'):
            return ResumeParser.extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            return ResumeParser.extract_text_from_docx(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    @staticmethod
    def extract_skills(text):
        skills_db = ['python', 'java', 'javascript', 'sql', 'machine learning', 'deep learning', 
                     'nlp', 'tensorflow', 'pytorch', 'react', 'node', 'aws', 'docker', 'kubernetes',
                     'git', 'api', 'rest', 'html', 'css', 'mongodb', 'postgresql', 'flask', 'django']
        text_lower = text.lower()
        return [skill for skill in skills_db if skill in text_lower]
