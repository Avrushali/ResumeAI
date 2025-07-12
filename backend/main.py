from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import re
import string
import json
import os
import io
import pdfplumber
import random

# --- Gensim Import for Word2Vec ---
from gensim.models import KeyedVectors

# --- NLTK Imports and Downloads (Backend) ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available for preprocessing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'stopwords' not found, downloading...")
    nltk.download('stopwords')
    print("NLTK 'stopwords' downloaded.")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK 'wordnet' not found, downloading...")
    nltk.download('wordnet')
    print("NLTK 'wordnet' downloaded.")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' not found, downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' downloaded.")

# --- FastAPI App Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    title="Resume Classification & Job Recommendation Backend",
    description="API for classifying resumes (using Word2Vec embeddings and a scikit-learn classifier) and recommending jobs."
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import kagglehub

# --- Model Paths (Relative to main.py) ---
WORD2VEC_MODEL_PATH = "models/GoogleNews-vectors-negative300.bin"
CLASSIFIER_MODEL_PATH = "models/voting_classifier.pkl"
JOBS_PERSONAS_DATA_PATH = "jobs.json"

# --- Download model from Kaggle Hub ---
print("Downloading dataset from Kaggle Hub...")
path = kagglehub.dataset_download("leadbest/googlenewsvectorsnegative300")
print(f"Dataset downloaded to: {path}")
WORD2VEC_MODEL_PATH = os.path.join(path, "GoogleNews-vectors-negative300.bin")


# --- Global Variables for Models and Data ---
w2v_model = None
classifier_model = None
job_personas_data = []  # Stores the persona mapping data from jobs.json

# --- Preprocessing Functions ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemma_and_filter_tokens(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()
    ]
    return lemmatized_tokens


def get_word2vec_document_vector(tokens: list[str], w2v_model_instance: KeyedVectors, vector_dim: int) -> np.ndarray:
    vectors = []
    for word in tokens:
        if word in w2v_model_instance.key_to_index:
            vectors.append(w2v_model_instance[word])

    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_dim)


def extract_years_of_experience(resume_text: str) -> str:
    text = resume_text.lower()
    pattern1 = r'(\d+(\.\d+)?)\s*(?:plus|\+)?\s*(?:-|to)?\s*(\d+(\.\d+)?)?\s*years?\s+of\s+experience'
    pattern2 = r'(\d+(\.\d+)?)\s*(?:plus|\+)?\s*(?:-|to)?\s*(\d+(\.\d+)?)?\s*yrs?'
    pattern3 = r'experience:\s*(\d+(\.\d+)?)\s*years?'
    pattern4 = r'(\d+(\.\d+)?)\s*YOE'

    match = re.search(pattern1, text)
    if match:
        if match.group(3):
            return f"{match.group(1)}-{match.group(3)} years"
        return f"{match.group(1)} years"

    match = re.search(pattern2, text)
    if match:
        if match.group(3):
            return f"{match.group(1)}-{match.group(3)} years"
        return f"{match.group(1)} years"

    match = re.search(pattern3, text)
    if match:
        return f"{match.group(1)} years"

    match = re.search(pattern4, text)
    if match:
        return f"{match.group(1)} years"

    return "Not specified"


def get_experience_numerical_range(exp_str: str) -> tuple[int, int]:
    exp_str_lower = exp_str.lower().strip()
    if "not specified" in exp_str_lower or "entry-level" in exp_str_lower or "fresher" in exp_str_lower:
        return (0, 1)  # Fresher/Entry-level typically 0-1 year

    match_range = re.search(
        r'(\d+)\s*(?:-|to)\s*(\d+)\s*years?', exp_str_lower)
    if match_range:
        min_exp = int(match_range.group(1))
        max_exp = int(match_range.group(2))
        return (min_exp, max_exp)

    match_single = re.search(r'(\d+)\s*\+?\s*years?', exp_str_lower)
    if match_single:
        min_exp = int(match_single.group(1))
        return (min_exp, 99)  # Assume 99 as a high upper bound for "X+"

    return (0, 99)  # Default broad range if no specific pattern is matched


def map_experience_to_persona(years_experience_str: str) -> str:
    """
    Maps the extracted years of experience string to a predefined persona type.
    """
    min_exp, max_exp = get_experience_numerical_range(years_experience_str)

    if max_exp <= 1:
        return "starter"
    elif min_exp >= 2 and min_exp <= 4:
        return "mid_level_pro"
    elif min_exp >= 5 and min_exp <= 7:
        return "expert"
    elif min_exp >= 8:
        return "specialist"
    else:
        return "mid_level_pro"  # Default to mid-level if ambiguous


# --- Load Models and Data on Startup ---
@app.on_event("startup")
async def load_models_and_data():
    global w2v_model, classifier_model, job_personas_data
    print("--- APP STARTUP: Loading models and data ---")

    if not os.path.exists(WORD2VEC_MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"Word2Vec model not found at: {WORD2VEC_MODEL_PATH}. "
                            "Please ensure 'GoogleNews-vectors-negative300.bin' is in the 'models/' directory.")
    try:
        w2v_model = KeyedVectors.load_word2vec_format(
            WORD2VEC_MODEL_PATH, binary=True)
        print(f"Word2Vec model loaded from: {WORD2VEC_MODEL_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading Word2Vec model: {e}. "
                            "Ensure it's a valid Word2Vec binary model file and gensim is installed correctly.")

    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"Classifier model not found at: {CLASSIFIER_MODEL_PATH}. "
                            "Please ensure 'voting_classifier.pkl' is in the 'models/' directory.")
    try:
        with open(CLASSIFIER_MODEL_PATH, 'rb') as f:
            classifier_model = joblib.load(f)
        print(
            f"Classifier (VotingClassifier) model loaded from: {CLASSIFIER_MODEL_PATH}")

        if not hasattr(classifier_model, 'predict'):
            print("Warning: Loaded model does not have a 'predict' method. "
                  "Ensure it's a valid scikit-learn classifier.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading classifier model: {e}. "
                            "Ensure it's a valid joblib/pickle file (e.g., a scikit-learn classifier).")

    if not os.path.exists(JOBS_PERSONAS_DATA_PATH):
        print(f"Warning: Job personas data file not found at {JOBS_PERSONAS_DATA_PATH}. "
              "Persona-based job recommendations will be affected. Please ensure 'jobs.json' contains the persona mapping.")
    else:
        try:
            with open(JOBS_PERSONAS_DATA_PATH, 'r', encoding='utf-8') as f:
                job_personas_data = json.load(f)
            print(
                f"Loaded {len(job_personas_data)} categories with personas from {JOBS_PERSONAS_DATA_PATH}")
        except json.JSONDecodeError as e:
            print(
                f"Error decoding JSON from {JOBS_PERSONAS_DATA_PATH}: {e}. Ensure it's valid JSON.")
        except Exception as e:
            print(
                f"Error loading job personas data: {e}. Persona-based recommendations might be affected.")

    print("--- APP STARTUP: Models and data loaded successfully ---")


# --- Pydantic Request/Response Models ---
class ResumeClassificationInput(BaseModel):
    resume_content: str


class ClassificationResponse(BaseModel):
    category: str
    years_experience: str


class JobRecommendationInput(BaseModel):
    category: str
    candidate_years_experience: str


class JobPosting(BaseModel):
    title: str
    company: str
    location: str
    description: str
    required_skills: str
    category: str
    experience_required: str
    url: str


# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "FastAPI backend for Resume Analyzer and Job Recommender is running."}


@app.post("/classify", response_model=ClassificationResponse)
async def classify_resume(data: ResumeClassificationInput):
    print("Received /classify request.")
    if w2v_model is None or classifier_model is None:
        raise HTTPException(
            status_code=503, detail="Models not loaded. Server is not ready.")

    try:
        print("Attempting to extract years of experience...")
        years_experience = extract_years_of_experience(data.resume_content)
        print(f"Years extracted: {years_experience}")

        print("Preprocessing resume text for classification...")
        cleaned_text = clean_text(data.resume_content)
        tokens = lemma_and_filter_tokens(cleaned_text)
        print("Preprocessing complete. Generating embedding...")

        vector_dim = 300
        document_vector = get_word2vec_document_vector(
            tokens, w2v_model, vector_dim)
        document_vector = document_vector.reshape(1, -1)

        predicted_category = classifier_model.predict(document_vector)[0]
        print(f"Predicted category: {predicted_category}. Sending response.")

        return {"category": predicted_category, "years_experience": years_experience}

    except Exception as e:
        print(f"Error during classify endpoint processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Classification failed due to an internal error: {e}")


@app.post("/recommend_jobs", response_model=list[JobPosting])
async def recommend_jobs(data: JobRecommendationInput):
    """
    Recommends jobs based on category and candidate's experience persona from local jobs.json.
    Generates search URLs for Indeed, Naukri, or LinkedIn.
    """
    print(
        f"Received /recommend_jobs request for category: {data.category}, candidate experience: {data.candidate_years_experience}")

    persona_type = map_experience_to_persona(data.candidate_years_experience)
    print(f"Candidate mapped to persona: {persona_type}")

    category_data = next((item for item in job_personas_data if item["category"].lower(
    ) == data.category.lower()), None)

    recommended_jobs_list = []
    if category_data and persona_type in category_data["personas"]:
        job_titles_for_persona = category_data["personas"].get(
            persona_type, [])  # Use .get with default empty list

        if job_titles_for_persona:  # Check if list is not empty
            num_jobs_to_recommend = min(5, len(job_titles_for_persona))
            # Handle case where job_titles_for_persona might have fewer than 5 items
            selected_titles = random.sample(
                job_titles_for_persona, num_jobs_to_recommend)

            # Define base search URLs for different platforms
            search_platforms = {
                "Indeed": "https://in.indeed.com/jobs?q={keywords}&l={location}",
                # User-friendly search URL
                "Naukri": "https://www.naukri.com/{keywords}-jobs-in-{location}",
                "LinkedIn": "https://www.linkedin.com/jobs/search/?keywords={keywords}&location={location}"
            }

            # List of possible locations for dummy data
            possible_locations = ["Bangalore", "Mumbai", "Hyderabad",
                                  "Delhi", "Pune", "Chennai", "Gurugram", "Noida"]

            for title in selected_titles:
                chosen_platform = random.choice(list(search_platforms.keys()))

                # Sanitize keywords for URL
                keywords_for_url = title.replace(" ", "+")

                # Randomly pick a location for the search URL
                location_for_url = random.choice(possible_locations)
                # Sanitize location for URL, especially for Naukri
                location_for_url_naukri = location_for_url.lower().replace(" ", "-")

                job_url = ""
                if chosen_platform == "Indeed":
                    job_url = search_platforms["Indeed"].format(
                        keywords=keywords_for_url, location=location_for_url)
                elif chosen_platform == "Naukri":
                    job_url = search_platforms["Naukri"].format(
                        keywords=keywords_for_url.lower(), location=location_for_url_naukri)
                elif chosen_platform == "LinkedIn":
                    job_url = search_platforms["LinkedIn"].format(
                        keywords=keywords_for_url, location=location_for_url)
                else:
                    job_url = "https://example.com/search-not-found"  # Fallback

                # --- THIS IS WHERE DUMMY DATA IS GENERATED ---
                company_name = f"Company {random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'])} {random.randint(100, 999)}"
                # Use the same random location for the job card as for the URL for consistency
                job_card_location = location_for_url
                description_text = (
                    f"Join our dynamic team as a {title}! We are seeking a talented professional in the {data.category} domain "
                    f"with {persona_type.replace('_', ' ').lower()} experience. This role offers exciting challenges and "
                    f"opportunities for growth in a collaborative environment. Key responsibilities include "
                    f"leading projects, innovating solutions, and contributing to strategic initiatives. "
                    f"Strong problem-solving skills and a passion for technology are highly valued."
                )
                skills_text = f"{title.split(' ')[-1]} skills, {data.category} knowledge, Problem-solving, Communication, Teamwork"

                recommended_jobs_list.append({
                    "title": title,
                    "company": company_name,
                    "location": job_card_location,  # Use the generated location
                    "description": description_text,
                    "required_skills": skills_text,
                    "category": data.category,
                    # Display the persona type
                    "experience_required": persona_type.replace('_', ' ').title(),
                    "url": job_url  # This is the generated search URL
                })
        else:
            print(
                f"DEBUG: No job titles defined for persona '{persona_type}' in category '{data.category}'.")
    else:
        print(
            f"DEBUG: No persona data found for category '{data.category}' or persona type '{persona_type}'.")

    print(f"Generated {len(recommended_jobs_list)} dummy jobs with search URLs for category '{data.category}' and persona '{persona_type}'.")
    return recommended_jobs_list


@app.post("/extract_pdf_text")
async def extract_pdf_text(file: UploadFile = File(...)):
    print(
        f"Received /extract_pdf_text request for file: {file.filename}, content_type: {file.content_type}")
    if file.content_type != "application/pdf":
        print(
            f"Error: Received non-PDF file with content type {file.content_type}")
        raise HTTPException(
            status_code=400, detail="Only PDF files are allowed.")

    try:
        print("Reading file content...")
        file_content = await file.read()
        pdf_bytes_io = io.BytesIO(file_content)
        print(f"File content read. Size: {len(file_content)} bytes.")

        text = ""
        print("Opening PDF with pdfplumber...")
        with pdfplumber.open(pdf_bytes_io) as pdf:
            print(f"PDF opened. Number of pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(layout=True)
                if page_text:
                    text += page_text + "\n"
                print(
                    f"Extracted text from page {i+1}. Length: {len(page_text or '')}")

        print(f"Total extracted text length from PDF: {len(text)} characters.")
        if not text.strip():
            print("Warning: Extracted text is empty or only whitespace.")
            return {"extracted_text": ""}
        return {"extracted_text": text}
    except Exception as e:
        print(f"Error during PDF extraction in /extract_pdf_text: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error extracting text from PDF: {e}")
