from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
from parser import ResumeParser
from preprocessor import TextPreprocessor
from matcher import MatchingEngine

app = FastAPI(title="Resume Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

preprocessor = TextPreprocessor()
parser = ResumeParser()
matcher = MatchingEngine()

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    text = parser.extract_text(file_path)
    skills = parser.extract_skills(text)
    os.remove(file_path)
    
    return {"filename": file.filename, "skills": skills, "text_length": len(text)}

@app.post("/match_score")
async def match_score(resume: UploadFile = File(...), job_description: str = Form(...)):
    file_path = f"temp_{resume.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)
    
    resume_text = parser.extract_text(file_path)
    resume_skills = parser.extract_skills(resume_text)
    jd_skills = parser.extract_skills(job_description)
    
    processed_resume = preprocessor.preprocess(resume_text)
    processed_jd = preprocessor.preprocess(job_description)
    
    score = matcher.calculate_match_score(processed_resume, processed_jd, resume_skills, jd_skills)
    
    os.remove(file_path)
    
    return {
        "filename": resume.filename,
        "score": score,
        "matched_skills": list(set(resume_skills) & set(jd_skills)),
        "missing_skills": list(set(jd_skills) - set(resume_skills))
    }

@app.post("/rank_resumes")
async def rank_resumes(resumes: List[UploadFile] = File(...), job_description: str = Form(...)):
    jd_skills = parser.extract_skills(job_description)
    processed_jd = preprocessor.preprocess(job_description)
    
    resumes_data = []
    for resume in resumes:
        file_path = f"temp_{resume.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
        
        text = parser.extract_text(file_path)
        skills = parser.extract_skills(text)
        processed_text = preprocessor.preprocess(text)
        
        resumes_data.append({
            'filename': resume.filename,
            'text': processed_text,
            'skills': skills
        })
        
        os.remove(file_path)
    
    ranked = matcher.rank_resumes(resumes_data, processed_jd, jd_skills)
    return {"ranked_resumes": ranked}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
