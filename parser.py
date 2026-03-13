import spacy
from spacy.matcher import Matcher
import re

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

import PyPDF2
import docx
import io

class ResumeParser:
    def __init__(self):
        self.skills_database = [
            # Technical Skills
            "python", "java", "javascript", "react", "angular", "node.js", 
            "sql", "postgresql", "mongodb", "aws", "docker", "kubernetes",
            "spring boot", "django", "flask", "machine learning", "nlp",
            "tensorflow", "pytorch", "c++", "c#", "html", "css", "tailwind",
            "typescript", "next.js", "vue.js", "graphql", "rest api", "azure", "gcp",
            "terraform", "ansible", "jenkins", "git", "ci/cd", "redis", "elasticsearch",
            "data analysis", "tableau", "power bi", "sap", "oracle", "swift", "kotlin",
            # Soft Skills
            "communication", "teamwork", "leadership", "problem solving", "agile",
            "scrum", "project management", "critical thinking", "collaboration",
            "time management", "adaptability", "creativity"
        ]

    def extract_text(self, file_content: bytes, filename: str):
        text = ""
        try:
            if filename.lower().endswith('.pdf'):
                print(f"DEBUG: Extracting PDF text from {filename}, size: {len(file_content)}")
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content), strict=False)
                if not pdf_reader.pages:
                    print("DEBUG: PDF has no pages.")
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Remove potential null bytes that can break some text processors
                            text += page_text.replace('\u0000', '') + "\n"
                    except Exception as pe:
                        print(f"DEBUG: Error on page {i}: {pe}")
                        continue
            elif filename.lower().endswith('.docx'):
                print(f"DEBUG: Extracting DOCX text from {filename}")
                doc = docx.Document(io.BytesIO(file_content))
                for para in doc.paragraphs:
                    text += para.text + "\n"
            else:
                # Try reading as plain text
                text = file_content.decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"CRITICAL: Error extracting text from {filename}: {str(e)}")
            raise e
        
        final_text = text.strip()
        if not final_text:
            print(f"WARNING: Extracted text is empty for {filename}")
        return final_text

    def extract_skills(self, text):
        text = text.lower()
        skills = []
        for skill in self.skills_database:
            if skill.lower() in text:
                skills.append(skill)
        return list(set(skills))

    def extract_education(self, text):
        education_keywords = ["Bachelors", "Masters", "PhD", "B.E", "B.Tech", "M.Tech", "BCA", "MCA"]
        found_edu = []
        for edu in education_keywords:
            if re.search(edu, text, re.IGNORECASE):
                found_edu.append(edu)
        return found_edu

    def extract_experience_years(self, text):
        # Look for patterns like "5 years", "2.5 years", "10+ years"
        # This is a heuristic. A robust solution needs NER to associate dates with jobs.
        matches = re.findall(r'(\d+(?:\.\d+)?)\+?\s*(?:years|yrs|year)', text, re.IGNORECASE)
        if not matches:
            return 0.0
        
        try:
            years = [float(m) for m in matches]
            # Heuristic: The max number mentioned followed by 'years' is *often* the total experience
            # or the experience in a specific skill. We'll take max for now as an optimistic estimate.
            # Filtering out unlikely high numbers (e.g. "2020 years")
            years = [y for y in years if y < 50] 
            return max(years) if years else 0.0
        except:
            return 0.0

    def clean_text(self, text):
        if not text:
            return ""
        # Limit text to first 50,000 chars to avoid spaCy performance issues
        text = text[:50000]
        # Bias Reduction: Remove names, ages, genders (simplified)
        doc = nlp(text)
        cleaned_tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct:
                cleaned_tokens.append(token.lemma_.lower())
        return " ".join(cleaned_tokens)
