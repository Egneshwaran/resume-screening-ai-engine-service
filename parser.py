import spacy
import re

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
        self.skills_database = ["python", "java", "javascript", "react", "sql", "aws", "docker"]

    def extract_text(self, file_content: bytes, filename: str):
        text = ""
        if filename.lower().endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif filename.lower().endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            text = file_content.decode('utf-8', errors='ignore')
        return text.strip()

    def extract_skills(self, text):
        text = text.lower()
        skills = [s for s in self.skills_database if s in text]
        return list(set(skills))

    def extract_education(self, text):
        return []

    def extract_experience_years(self, text):
        return 0.0

    def clean_text(self, text):
        return text[:1000]
