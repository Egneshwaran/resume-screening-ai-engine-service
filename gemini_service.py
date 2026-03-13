import os
import google.generativeai as genai
from typing import Dict, List, Optional
import json

class GeminiMatcher:
    def __init__(self, api_key: Optional[str] = None):
        # Allow passing key or reading from environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Configure the official Gemini API endpoint
        # The library uses 'https://generativelanguage.googleapis.com' by default.
        # If the user is hitting 'daily-cloudcode-pa.googleapis.com', it's likely an env override.
        # We explicitly use the default or a configured override below if needed.
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def analyze_match(self, job_description: str, resume_text: str, 
                     required_skills: str = "", required_experience: str = "",
                     skill_weight: int = 50, exp_weight: int = 30, desc_weight: int = 20) -> Dict:
        if not self.model:
            return {
                "error": "Gemini API key not configured. Using basic ranking fallback.",
                "total_score": 0.0,
                "explanation": "Gemini service unavailable."
            }
        
        prompt = f"""
        You are an AI Resume Screening and Matching Engine.
        
        Your task is to compare the Job Description with the Candidate's Resume and accurately identify matches across the following components:
        • Key Skills
        • Required Experience
        • Detailed Job Description

        The final matching score must be calculated using customizable weight percentages defined by the recruiter.
        Your goal is to analyze the entire resume text and identify ALL possible matches without limiting the number of results.

        Instructions:
        1. Extract and normalize the following information from the Job Description:
           • Key Skills: {required_skills}
           • Required Experience: {required_experience}
           • Detailed Job Description: {job_description}
           • Recruiter-defined weight percentages: Skills={skill_weight}%, Experience={exp_weight}%, Description={desc_weight}%

        2. Compare these requirements with the candidate’s resume across ALL sections including:
           • Skills, Experience, Projects, Certifications, Tools & Technologies, Summary or Profile.

        3. Perform case-insensitive matching (example: Python = python).
        4. Support synonym and variation matching (e.g., AWS = Amazon Web Services).
        5. Apply context-based matching: Count skills if they appear in projects, internships, or experience.
        6. Remove duplicate skills before evaluation.
        7. Detect experience relevance: Project work or internships related to the domain count as matched experience.
        8. Extract keywords and technologies from the Detailed Job Description and match with resume context.
        9. Only classify items as Skill Gaps or Experience Gaps if they do not appear anywhere in the resume.
        10. Do NOT limit the number of matched items. Return every valid match found.

        Score Calculation Rules:
        1. Key Skills Match % = (Matched Key Skills ÷ Total Key Skills) × 100
        2. Experience Match % = (Matched Experience Items ÷ Total Experience Requirements) × 100
        3. Description Keyword Match % = (Matched Description Keywords ÷ Total Description Keywords) × 100
        4. Final Weighted Match Score = (Key Skills Match % * {skill_weight}/100) + (Experience Match % * {exp_weight}/100) + (Description Match % * {desc_weight}/100)

        Candidate Resume:
        {resume_text}

        Provide a structured JSON response with exactly these keys:
        1. "total_score": (float, 0-100 - the Final Weighted Match Score)
        2. "skill_score": (float, 0-100 - Key Skills Match %)
        3. "experience_score": (float, 0-100 - Experience Match %)
        4. "description_score": (float, 0-100 - Description Keyword Match %)
        5. "matched_skills": (list of strings - ALL matching skills found)
        6. "matched_experience": (list of strings - relevant roles, internships, or projects)
        7. "keyword_matches": (list of strings - matched technologies/concepts from JD description)
        8. "skill_gaps": (list of strings - required skills not found)
        9. "experience_gaps": (list of strings - missing experience requirements)
        10. "short_explanation": (string - a concise summary of the match results)

        RETURN ONLY RAW JSON.
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini")
                
            return json.loads(response.text)
            
        except Exception as e:
            error_msg = str(e)
            return {
                "error": f"Gemini API error: {error_msg}",
                "total_score": 0.0,
                "explanation": "Failed to generate AI insights."
            }

    def analyze_ats(self, resume_text: str, job_role: Optional[str] = None) -> Dict:
        if not self.model:
            return {
                "error": "Gemini API key not configured.",
                "total_score": 0,
                "ats_compatibility": 0,
                "accuracy_percentage": 0,
                "formatting": 0,
                "strengths": [],
                "improvements": ["Please configure Gemini API key for detailed analysis."],
                "sections": []
            }
        
        role_context = f"\nKeep the suggestions tailored to relevance for the job role: {job_role}" if job_role else ""

        prompt = f"""
        Perform a comprehensive ATS (Applicant Tracking System) evaluation for the following resume.
        Evaluate it based on industry standards for formatting, keyword density, and structural integrity.{role_context}
        
        Resume Text:
        {resume_text}
        
        Provide a structured JSON response with:
        1. total_score (0-100)
        2. ats_compatibility (0-100)
        3. accuracy_percentage (0-100)
        4. formatting (0-100)
        5. strengths (list of strings)
        6. improvements (list of strings)
        7. sections (list of objects with 'name', 'score', and 'feedback')

        RETURN ONLY RAW JSON.
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            return {
                "error": str(e),
                "total_score": 50,
                "ats_compatibility": 50,
                "sections": []
            }

    def optimize_resume(self, resume_text: str) -> Dict:
        if not self.model:
            return {
                "error": "Gemini API key not configured.",
                "optimized_summary": "Please configure Gemini API key to get optimized summary.",
                "bullet_points": []
            }
        
        prompt = f"""
        Based on the following resume text, provide an ATS-optimized professional summary and 3-5 high-impact bullet points.
        
        Resume Text:
        {resume_text}
        
        Provide a structured JSON response with:
        1. optimized_summary (string)
        2. bullet_points (list of strings)
        3. general_tips (list of 3 strings)

        RETURN ONLY RAW JSON.
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            return {
                "error": str(e),
                "optimized_summary": "Failed to generate AI-optimized summary.",
                "bullet_points": [],
                "general_tips": ["Use action verbs", "Quantify results", "Include keywords"]
            }

