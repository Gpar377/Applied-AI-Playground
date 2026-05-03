from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MatchingEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def calculate_match_score(self, resume_text, job_description, resume_skills, jd_skills):
        texts = [resume_text, job_description]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        skill_match = len(set(resume_skills) & set(jd_skills)) / len(jd_skills) if jd_skills else 0
        
        final_score = (similarity * 0.6 + skill_match * 0.4) * 100
        return round(final_score, 2)
    
    def rank_resumes(self, resumes_data, job_description, jd_skills):
        results = []
        for resume in resumes_data:
            score = self.calculate_match_score(
                resume['text'], job_description, resume['skills'], jd_skills
            )
            matched_skills = list(set(resume['skills']) & set(jd_skills))
            missing_skills = list(set(jd_skills) - set(resume['skills']))
            
            results.append({
                'filename': resume['filename'],
                'score': score,
                'matched_skills': matched_skills,
                'missing_skills': missing_skills
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
