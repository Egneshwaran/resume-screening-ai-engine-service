from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from gemini_service import GeminiMatcher

class CandidateRanker:
    def __init__(self, use_gemini: bool = False):
        self.vectorizer = TfidfVectorizer()
        self.gemini = GeminiMatcher() if use_gemini else None

    def calculate_similarity(self, job_desc, resume_text):
        if not job_desc or not resume_text:
            return 0.0
        try:
            # Ensure they are strings
            job_desc = str(job_desc)
            resume_text = str(resume_text)
            tfidf_matrix = self.vectorizer.fit_transform([job_desc, resume_text])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            print(f"Similarity Calculation Error: {e}")
            return 0.0

    def analyze_skill_gap(self, required_skills, candidate_skills):
        if not required_skills:
            return {"match_percentage": 0, "missing_skills": [], "matched_skills": []}
            
        required = set([s.strip().lower() for s in str(required_skills).split(",") if s.strip()])
        candidate = set([str(s).lower() for s in (candidate_skills or [])])
        
        missing = required - candidate
        match_pct = (len(required - missing) / len(required)) * 100 if required else 0
        
        return {
            "match_percentage": match_pct,
            "missing_skills": list(missing),
            "matched_skills": list(required & candidate)
        }
    
    def analyze_experience(self, required_range, candidate_years):
        # required_range e.g. "2-4 years", "5+ years", "0-1 years"
        # If None or empty, assume 0
        if not required_range:
            return 100.0
            
        min_exp = 0.0
        max_exp = 100.0
        
        # Check for "N-M years" or "N+ years"
        range_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', required_range)
        plus_match = re.search(r'(\d+)\+', required_range)
        
        if range_match:
            min_exp = float(range_match.group(1))
            max_exp = float(range_match.group(2))
        elif plus_match:
            min_exp = float(plus_match.group(1))
            max_exp = 100.0 # Open ended
        else:
            # Fallback if just a number is given "2 years"
            simple_match = re.search(r'(\d+)', required_range)
            if simple_match:
                min_exp = float(simple_match.group(1))
            else:
                try:
                     # Try to parse if it's just a number string "2"
                     min_exp = float(required_range)
                except:
                     min_exp = 0.0

        if candidate_years >= min_exp:
            # If within range or above min (and implied max is high if not strictly ranged)
            # Actually if explicit range 2-4, and candidate has 10, are they overqualified?
            # Usually for this context, more is better or equal. 
            # If strictly 2-4, 5 might be 100% or slightly penalized? 
            # PROMPT: "minimum years of experience". So N+ is implied logic mostly.
            # But user said "range-based input".
            # I will assume >= min is 100%.
            return 100.0
        else:
            # Calculate gap
            if min_exp == 0: return 100.0
            score = (candidate_years / min_exp) * 100.0
            return max(0.0, min(100.0, score))

    def generate_explanation(self, analysis, similarity_score, exp_score, required_exp, candidate_exp):
        explanation = f"Skill match: {analysis['match_percentage']:.1f}%. "
        
        if required_exp:
            explanation += f"Experience: {candidate_exp} yrs (Req: {required_exp}). "
        
        if analysis['missing_skills']:
            explanation += f"Missing: {', '.join(analysis['missing_skills'])}. "
        
        explanation += f"Content relevance: {similarity_score * 100:.1f}%."
        return explanation
    
    def rank_candidates(self, job, resumes):
        # job: dict {description, required_skills, required_experience}
        results = []
        for resume in resumes:
            analysis = self.analyze_single(job, resume)
            results.append(analysis)
        
        return sorted(results, key=lambda x: x['total_score'], reverse=True)

    def analyze_single(self, job, resume):
        # job: {description, required_skills, required_experience, skill_weight, experience_weight, description_weight}
        # resume: {id, text, skills, experience_years}
        
        # Get weights or use defaults
        w_skill = job.get('skill_weight', 50) / 100.0
        w_exp = job.get('experience_weight', 30) / 100.0
        w_desc = job.get('description_weight', 20) / 100.0

        gemini_result = None
        if self.gemini:
            gemini_result = self.gemini.analyze_match(
                job_description=job['description'], 
                resume_text=resume['text'],
                required_skills=job.get('required_skills', ''),
                required_experience=job.get('required_experience', '0'),
                skill_weight=int(w_skill * 100),
                exp_weight=int(w_exp * 100),
                desc_weight=int(w_desc * 100)
            )
        
        if gemini_result and "error" not in gemini_result:
            s_skill = gemini_result.get("skill_score", 0.0)
            s_exp = gemini_result.get("experience_score", 0.0)
            s_desc = gemini_result.get("description_score", 0.0)
            total_score = gemini_result.get("total_score", 0.0)
            
            # Construct a detailed explanation following the user's requested format
            explanation = f"Matched Key Skills: {', '.join(gemini_result.get('matched_skills', []))}\n\n"
            explanation += f"Matched Experience: {', '.join(gemini_result.get('matched_experience', []))}\n\n"
            explanation += f"Keyword Matches from Detailed Job Description: {', '.join(gemini_result.get('keyword_matches', []))}\n\n"
            explanation += f"Skill Gaps: {', '.join(gemini_result.get('skill_gaps', []))}\n\n"
            explanation += f"Experience Gaps: {', '.join(gemini_result.get('experience_gaps', []))}\n\n"
            explanation += f"Category Match Scores:\n"
            explanation += f"Key Skills Match %: {s_skill}%\n"
            explanation += f"Experience Match %: {s_exp}%\n"
            explanation += f"Description Match %: {s_desc}%\n\n"
            explanation += f"Final Weighted Match Score: {total_score}%\n\n"
            explanation += f"AI Summary: {gemini_result.get('short_explanation', '')}"

            return {
                "resume_id": resume.get('id', 0),
                "total_score": round(total_score, 2),
                "skill_score": round(s_skill, 2),
                "experience_score": round(s_exp, 2),
                "education_score": 90.0,
                "matched_skills": ", ".join(gemini_result.get("matched_skills", [])),
                "missing_skills": ", ".join(gemini_result.get("skill_gaps", [])),
                "matched_experience": gemini_result.get("matched_experience", []),
                "keyword_matches": gemini_result.get("keyword_matches", []),
                "experience_gaps": gemini_result.get("experience_gaps", []),
                "explanation": explanation
            }
        else:
            # Fallback
            sim_score = self.calculate_similarity(job['description'], resume['text'])
            gap_analysis = self.analyze_skill_gap(job['required_skills'], resume.get('skills', []))
            candidate_exp = resume.get('experience_years', 0.0)
            exp_score = self.analyze_experience(job.get('required_experience', '0'), candidate_exp)
            
            # Weighted Score
            total_score = (gap_analysis['match_percentage'] * w_skill) + (exp_score * w_exp) + (sim_score * 100 * w_desc)
            
            explanation = self.generate_explanation(gap_analysis, sim_score, exp_score, job.get('required_experience'), candidate_exp)
            if gemini_result and "error" in gemini_result:
                explanation += f" (Note: Gemini unavailable: {gemini_result['error']})"

            return {
                "resume_id": resume.get('id', 0),
                "total_score": round(total_score, 2),
                "skill_score": round(gap_analysis['match_percentage'], 2),
                "experience_score": round(exp_score, 2),
                "education_score": 90.0,
                "matched_skills": ", ".join(gap_analysis['matched_skills']),
                "missing_skills": ", ".join(gap_analysis['missing_skills']),
                "explanation": explanation
            }
    def analyze_ats(self, resume_text, job_role=None):
        from parser import ResumeParser
        parser = ResumeParser()
        
        if self.gemini:
            return self.gemini.analyze_ats(resume_text, job_role)
        
        # Improved Fallback basic logic
        extracted_skills = parser.extract_skills(resume_text)
        experience_years = parser.extract_experience_years(resume_text)
        education = parser.extract_education(resume_text)
        
        # Calculate scores
        # Skill score: 10+ skills is a good target
        skill_score = min(100, (len(extracted_skills) / 10) * 100)
        # Compatibility score based on structure (simple check for common headers)
        headers = ["experience", "education", "skills", "summary", "projects", "contact", "objective"]
        found_headers = [h for h in headers if h in resume_text.lower()]
        compat_score = (len(found_headers) / len(headers)) * 100
        
        # Formatting score (heuristic based on bullets and length)
        bullets = resume_text.count('•') + resume_text.count('*') + resume_text.count('- ')
        formatting_score = min(100, (bullets / 15) * 100) if bullets > 0 else 50
        
        keyword_score = skill_score # Use skill count as proxy for keywords in fallback
        
        total_score = (skill_score * 0.4) + (compat_score * 0.3) + (formatting_score * 0.3)
        
        # Dynamic Strengths
        strengths = []
        if len(extracted_skills) >= 5:
            strengths.append(f"Strong skill density with {len(extracted_skills)} key competencies identified")
        if experience_years > 0:
            strengths.append(f"Clear career progression with approximately {experience_years} years of experience")
        if education:
            strengths.append(f"Formal education background ({', '.join(education)}) is well-documented")
        if not strengths:
            strengths = ["Text successfully extracted", "Basic structure identified"]
            
        # Dynamic Improvements 
        improvements = []
        if job_role:
            improvements.append(f"Highlight your direct experience and impact handling {job_role} responsibilities")
        if len(extracted_skills) < 8:
            improvements.append("Consider adding more industry-specific technical skills to pass ATS filters")
        if bullets < 10:
            improvements.append("Use more bullet points to describe your achievements (aim for 3-5 per role)")
        if "summary" not in found_headers:
            improvements.append("Add a professional summary section to introduce your value proposition")
        if experience_years == 0:
            improvements.append("Ensure your work experience dates are clearly formatted for the parser")
        if not improvements:
             improvements = ["Consider quantifying achievements with specific metrics and results"]

        return {
            "total_score": round(total_score, 2),
            "ats_compatibility": round(compat_score, 2),
            "keyword_relevance": round(keyword_score, 2),
            "formatting": round(formatting_score, 2),
            "strengths": strengths[:3],
            "improvements": improvements[:3],
            "sections": [
                {
                    "name": "Summary", 
                    "score": 80 if "summary" in found_headers else 40, 
                    "feedback": "Professional summary identified." if "summary" in found_headers else "Missing professional summary."
                },
                {
                    "name": "Skills", 
                    "score": round(skill_score, 2), 
                    "feedback": f"Found {len(extracted_skills)} relevant keywords."
                },
                {
                    "name": "Experience", 
                    "score": 90 if experience_years > 0 else 50, 
                    "feedback": f"Experience section parsed ({experience_years} years)." if experience_years > 0 else "Experience section is thin or poorly formatted."
                },
                {
                    "name": "Education", 
                    "score": 90 if education else 50, 
                    "feedback": f"Education: {', '.join(education)}" if education else "Education details not clearly identified."
                }
            ]
        }
    def optimize_resume(self, resume_text):
        from parser import ResumeParser
        parser = ResumeParser()
        if self.gemini:
            return self.gemini.optimize_resume(resume_text)
        
        extracted_skills = parser.extract_skills(resume_text)
        experience_years = parser.extract_experience_years(resume_text)
        
        # Dynamic summary suggestion
        summary = ""
        if experience_years > 0:
            summary = f"Results-driven professional with {experience_years} years of experience."
            if extracted_skills:
                summary += f" Skilled in {', '.join(extracted_skills[:3])}."
            summary += " Ready to leverage background and drive growth."
        else:
            summary = "Motivated and adaptable professional"
            if extracted_skills:
                summary += f" with a strong foundation in {', '.join(extracted_skills[:3])}."
            summary += " Eager to contribute to a collaborative team and deliver value."

        # Dynamic bullet points based on skills detected
        bullets = []
        if "python" in [s.lower() for s in extracted_skills]:
            bullets.append("Developed efficient Python automation scripts, reducing processing time by 30%.")
        elif "react" in [s.lower() for s in extracted_skills]:
            bullets.append("Built responsive user interfaces with React, improving user engagement by 20%.")
        elif "sql" in [s.lower() for s in extracted_skills]:
            bullets.append("Optimized SQL queries to improve database retrieval speed by 40%.")
        else:
            bullets.append("Spearheaded project initiatives that delivered key milestones 2 weeks ahead of schedule.")

        if experience_years > 2:
            bullets.append("Mentored junior team members and fostered a culture of continuous learning and growth.")
        else:
            bullets.append("Collaborated with cross-functional teams to streamline workflows and improve productivity.")

        # General Tips
        general_tips = [
            "Start bullet points with strong action verbs (e.g., 'Spearheaded', 'Engineered')",
            "Always quantify your achievements with metrics ($, %, impact)"
        ]
        if len(extracted_skills) < 5:
            general_tips.append("Add a dedicated highly visible 'Skills' section with more relevant industry keywords.")

        return {
            "optimized_summary": f"💡 (AI basic fallback): {summary}",
            "bullet_points": bullets,
            "general_tips": general_tips
        }
