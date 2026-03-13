from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Optional
from parser import ResumeParser
from ranker import CandidateRanker
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Recruitment Engine")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, # Set to False if using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = ResumeParser()

# Initialize ranker with Gemini if API key is present
use_gemini = os.getenv("GEMINI_API_KEY") is not None
ranker = CandidateRanker(use_gemini=use_gemini)

class JobSpec(BaseModel):
    description: str
    required_skills: str
    required_experience: Optional[str] = "0"
    skill_weight: Optional[int] = 50
    experience_weight: Optional[int] = 30
    description_weight: Optional[int] = 20

class ResumeData(BaseModel):
    id: Optional[int] = 0
    text: Optional[str] = ""
    file_url: Optional[str] = None

class RankRequest(BaseModel):
    job: JobSpec
    resumes: List[ResumeData]

class ProcessRequest(BaseModel):
    job_description: str
    required_skills: str
    required_experience: Optional[str] = "0"
    resume_text: Optional[str] = ""
    file_url: Optional[str] = None
    skill_weight: Optional[int] = 50
    experience_weight: Optional[int] = 30
    description_weight: Optional[int] = 20

@app.post("/parse")
async def parse_resume(resume: ResumeData):
    text = resume.text
    if not text and resume.file_url:
        try:
            import urllib.request
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(resume.file_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                content = response.read()
                text = parser.extract_text(content, resume.file_url.split('?')[0])
        except Exception as e:
            print(f"Error parsing from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch or parse file from URL: {str(e)}")

    if not text:
        return {"skills": [], "education": [], "experience_years": 0, "cleaned_text": ""}

    skills = parser.extract_skills(text)
    education = parser.extract_education(text)
    experience = parser.extract_experience_years(text)
    return {
        "skills": skills,
        "education": education,
        "experience_years": experience,
        "cleaned_text": parser.clean_text(text)
    }

import asyncio
import urllib.request
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=20)

async def fetch_and_parse(r: ResumeData):
    text = r.text
    if not text and r.file_url:
        try:
            # Run blocking I/O in a thread
            loop = asyncio.get_event_loop()
            def download():
                headers = {'User-Agent': 'Mozilla/5.0'}
                req = urllib.request.Request(r.file_url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    return response.read()
            
            content = await loop.run_in_executor(executor, download)
            text = parser.extract_text(content, r.file_url.split('?')[0])
        except Exception as e:
            print(f"Error parsing {r.file_url}: {e}")
            text = ""
    
    return {
        "id": r.id,
        "text": parser.clean_text(text or ""),
        "skills": parser.extract_skills(text or ""),
        "experience_years": parser.extract_experience_years(text or "")
    }

@app.post("/rank")
async def rank_resumes(request: RankRequest):
    # Process all resumes in parallel
    resumes_processed = await asyncio.gather(*(fetch_and_parse(r) for r in request.resumes))
    
    rankings = ranker.rank_candidates(
        {
            "description": parser.clean_text(request.job.description), 
            "required_skills": request.job.required_skills,
            "required_experience": request.job.required_experience,
            "skill_weight": request.job.skill_weight,
            "experience_weight": request.job.experience_weight,
            "description_weight": request.job.description_weight
        },
        resumes_processed
    )
    return rankings

@app.post("/process")
async def process_single(request: ProcessRequest):
    # Reuse the logic but for a single request
    dummy_resume = ResumeData(id=0, text=request.resume_text, file_url=request.file_url)
    processed = await fetch_and_parse(dummy_resume)
    
    job = {
        "description": parser.clean_text(request.job_description),
        "required_skills": request.required_skills,
        "required_experience": request.required_experience,
        "skill_weight": getattr(request, 'skill_weight', 50),
        "experience_weight": getattr(request, 'experience_weight', 30),
        "description_weight": getattr(request, 'description_weight', 20)
    }
    
    return ranker.rank_candidates(job, [processed])[0]
@app.post("/ats-check")
async def ats_check(file: UploadFile = File(...), job_role: Optional[str] = Form(None)):
    if not file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF, DOCX, or TXT.")
    
    try:
        content = await file.read()
        text = parser.extract_text(content, file.filename)
        
        if not text:
            raise HTTPException(status_code=400, detail="The file appears to be empty or contains no extractable text. If it's a PDF, ensure it's not an image-only scan.")
        
        return ranker.analyze_ats(text, job_role)
    except Exception as e:
        print(f"ATS Check Error: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

@app.post("/optimize")
async def optimize(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF, DOCX, or TXT.")
    
    try:
        content = await file.read()
        text = parser.extract_text(content, file.filename)
        
        if not text:
            raise HTTPException(status_code=400, detail="The file appears to be empty or contains no extractable text.")
        
        return ranker.optimize_resume(text)
    except Exception as e:
        print(f"Optimization Error: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
