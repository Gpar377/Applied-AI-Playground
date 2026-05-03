import streamlit as st
from parser import ResumeParser
from preprocessor import TextPreprocessor
from matcher import MatchingEngine

st.set_page_config(page_title="Resume Screening System", layout="wide")

st.title("🎯 AI-Powered Resume Screening & Job Matching")

preprocessor = TextPreprocessor()
parser = ResumeParser()
matcher = MatchingEngine()

tab1, tab2 = st.tabs(["Single Resume Match", "Rank Multiple Resumes"])

with tab1:
    st.header("Match Resume with Job Description")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_resume = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])
    
    with col2:
        job_desc = st.text_area("Job Description", height=200)
    
    if st.button("Calculate Match Score") and uploaded_resume and job_desc:
        with open(f"temp_{uploaded_resume.name}", "wb") as f:
            f.write(uploaded_resume.getbuffer())
        
        resume_text = parser.extract_text(f"temp_{uploaded_resume.name}")
        resume_skills = parser.extract_skills(resume_text)
        jd_skills = parser.extract_skills(job_desc)
        
        processed_resume = preprocessor.preprocess(resume_text)
        processed_jd = preprocessor.preprocess(job_desc)
        
        score = matcher.calculate_match_score(processed_resume, processed_jd, resume_skills, jd_skills)
        
        st.success(f"Match Score: {score}%")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Matched Skills")
            matched = list(set(resume_skills) & set(jd_skills))
            st.write(matched if matched else "None")
        
        with col2:
            st.subheader("❌ Missing Skills")
            missing = list(set(jd_skills) - set(resume_skills))
            st.write(missing if missing else "None")

with tab2:
    st.header("Rank Multiple Resumes")
    
    uploaded_resumes = st.file_uploader("Upload Resumes", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
    job_desc_rank = st.text_area("Job Description", height=150, key="jd_rank")
    
    if st.button("Rank Resumes") and uploaded_resumes and job_desc_rank:
        jd_skills = parser.extract_skills(job_desc_rank)
        processed_jd = preprocessor.preprocess(job_desc_rank)
        
        resumes_data = []
        for resume in uploaded_resumes:
            with open(f"temp_{resume.name}", "wb") as f:
                f.write(resume.getbuffer())
            
            text = parser.extract_text(f"temp_{resume.name}")
            skills = parser.extract_skills(text)
            processed_text = preprocessor.preprocess(text)
            
            resumes_data.append({
                'filename': resume.name,
                'text': processed_text,
                'skills': skills
            })
        
        ranked = matcher.rank_resumes(resumes_data, processed_jd, jd_skills)
        
        st.subheader("📊 Ranked Results")
        for i, result in enumerate(ranked, 1):
            with st.expander(f"#{i} - {result['filename']} - Score: {result['score']}%"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Matched Skills:**", result['matched_skills'])
                with col2:
                    st.write("**Missing Skills:**", result['missing_skills'])
