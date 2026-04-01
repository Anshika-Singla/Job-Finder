from fastapi import FastAPI
from pydantic import BaseModel
import requests
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os


load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


kw_model = KeyBERT()

custom_stopwords = {
    "know", "knowledge", "skill", "skills", "ability", "abilities",
    "experience", "experienced", "working", "work", "works",
    "good", "strong", "excellent", "motivated", "driven",
    "passionate", "dedicated", "committed", "innovative",
    "creative", "responsible", "team", "player", "focused"
}


class JobSearchRequest(BaseModel):
    description: str
    city: str = ""
    state: str = ""
    country: str = "in"
    date_posted: str = "all"


def extract_keywords(text, top_n=5):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n * 4
    )
    unique_keywords, seen = [], set()
    for kw, score in keywords:
        kw_clean = kw.lower().strip()
        if any(stop in kw_clean.split() for stop in custom_stopwords):
            continue
        if kw_clean not in seen:
            unique_keywords.append(kw_clean)
            seen.add(kw_clean)
    return unique_keywords[:top_n]


def fetch_jobs_from_api(description, city, state, country, date_posted):
    keywords = extract_keywords(description)
    query = " ".join(keywords)
    if city:
        query += f" {city}"

    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    querystring = {
        "query": query,
        "page": "1",
        "num_pages": "1",
        "date_posted": date_posted if date_posted else "all",
        "country": country if country else "in",
        "language": "en"
    }

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        return []


def match_jobs_semantic(user_input, jobs, title_key="job_title"):
    if not jobs:
        return []
    titles = [job.get(title_key, "") for job in jobs]
    docs = [user_input] + titles

    vectorizer = TfidfVectorizer().fit_transform(docs)
    vectors = vectorizer.toarray()
    scores = cosine_similarity([vectors[0]], vectors[1:])[0]

    for i, job in enumerate(jobs):
        job["match_score"] = round(float(scores[i]) * 100, 2)
    jobs.sort(key=lambda x: x["match_score"], reverse=True)
    return jobs


def format_date(iso_date_str):
    try:
        dt = datetime.fromisoformat(iso_date_str.replace("Z", ""))
        return dt.strftime("%d %b %Y")
    except Exception:
        return iso_date_str


@app.get("/")
def home():
    return {"message": "Job finder API is running "}

@app.post("/search")
async def search_jobs(request: JobSearchRequest):
    if not request.description.strip():
        return {"error": "Please enter a job description."}

    jobs_api = fetch_jobs_from_api(request.description, request.city, request.state, request.country, request.date_posted)
    matched_jobs_api = match_jobs_semantic(request.description, jobs_api, title_key="job_title")

    for job in matched_jobs_api:
        job["source"] = "API"
        job["date_posted"] = format_date(job.get("job_posted_at_datetime_utc", "N/A"))

    top_jobs = matched_jobs_api[:10]

    return {
        "description": request.description,
        "city": request.city,
        "state": request.state,
        "country": request.country,
        "date_posted": request.date_posted,
        "results": top_jobs
    }