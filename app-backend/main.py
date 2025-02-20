from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from starlette.status import HTTP_403_FORBIDDEN
import google.generativeai as genai
import numpy as np
import joblib
import pdfplumber
import re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os


# FastAPI App Initialization
app = FastAPI()

load_dotenv()

# API Key Setup
API_KEY = os.getenv("FAST_API_KEY")  # Replace with a strong API key
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key")

# Configure Gemini API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Pre-trained SentenceTransformer Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# Load GDPR Embeddings and Labels
gdpr_embeddings = np.load("sentencegdpr_embeddings.npy")
gdpr_labels = joblib.load("sentencegdpr_labels.pkl")

# Initialize Gemini Model
model = genai.GenerativeModel('gemini-pro')

# Cache for GDPR summaries
cache = {}

# Utility Functions
def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF file."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def split_into_regulations(text):
    """Splits extracted text into individual regulations based on numbering patterns."""
    pattern = r"\d+\.\s+(.*?)(?=\n\d+\.|\Z)"
    return [reg.strip() for reg in re.findall(pattern, text, re.DOTALL)]

def find_best_match(text):
    """Finds the best matching GDPR rule using cosine similarity."""
    global gdpr_embeddings
    input_embedding = embed_model.encode([text], convert_to_numpy=True)
    if len(input_embedding.shape) == 1:
        input_embedding = input_embedding.reshape(1, -1)
    if len(gdpr_embeddings.shape) == 1:
        gdpr_embeddings = gdpr_embeddings.reshape(1, -1)
    similarities = cosine_similarity(input_embedding, gdpr_embeddings)[0]
    best_match_idx = int(np.argmax(similarities))
    return gdpr_labels[best_match_idx], similarities[best_match_idx]

def get_gdpr_summary(gdpr_label):
    """Fetches a well-structured GDPR summary or generates one using Gemini AI."""
    if gdpr_label in cache:
        return cache[gdpr_label]
    
    prompt = f"Summarize the GDPR compliance rule {gdpr_label} in simple terms."
    response = model.generate_content(prompt)
    summary = response.text.strip()
    cache[gdpr_label] = summary
    return summary

def validate_match(org_text, gdpr_label, gdpr_summary):
    """Validates compliance of the regulation with GDPR and provides recommendations."""
    prompt = f"""
    Compare the following organizational regulation with GDPR compliance:
    
    **Organizational Regulation:** {org_text}
    **Matched GDPR Rule:** {gdpr_label}
    **GDPR Summary:** {gdpr_summary}
    
    Does this regulation comply with GDPR? Explain why or why not, and suggest improvements if needed.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def process_regulations(regulations):
    """Processes and validates each extracted regulation against GDPR."""
    results = []
    for reg in regulations:
        best_match, similarity_score = find_best_match(reg)
        gdpr_summary = get_gdpr_summary(best_match)
        llm_analysis = validate_match(reg, best_match, gdpr_summary)
        results.append({
            "Regulation": reg,
            "Best Matched GDPR Rule": best_match,
            "Similarity Score": float(round(similarity_score, 4)),
            "GDPR Summary": gdpr_summary,
            "Gemini Analysis": llm_analysis
        })
    return results

def convert_numpy_floats(obj):
    """Convert numpy float types to Python floats recursively"""
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)  # Convert to Python float
    elif isinstance(obj, list):
        return [convert_numpy_floats(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_floats(value) for key, value in obj.items()}
    return obj  # Return as-is if no conversion is needed

# FastAPI Endpoint for File Upload
@app.post("/gdpr/upload", dependencies=[Depends(verify_api_key)])
async def gdpr_compliance_check(file: UploadFile = File(...)):
    """Uploads a PDF or TXT file, extracts regulations, and checks GDPR compliance."""
    
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file.file)
    elif file.filename.endswith(".txt"):
        text = (await file.read()).decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PDF or TXT file.")

    # Extract and process regulations
    regulations = split_into_regulations(text)
    if not regulations:
        raise HTTPException(status_code=400, detail="No regulations found in the file.")

    results = process_regulations(regulations)

    return {"file": file.filename, "compliance_results": results}
