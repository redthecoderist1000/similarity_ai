from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware
# from sympy import true
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
similarity_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

@app.get("/test")
def test():
    return {"success": 'response'}

class SimilarityRequest(BaseModel):
    question: str
    blooms_category: str
    repository: str
    lesson_id: str

@app.post("/similarity")
def compute_similarity(request: SimilarityRequest):
    try:
        # get questions in db
        response = supabase.table('tbl_question')\
            .select('id, question')\
            .eq('blooms_category',request.blooms_category)\
            .eq('repository',request.repository)\
            .eq('lesson_id',request.lesson_id)\
            .execute()


        if not response.data or len(response.data) == 0:
            return { 
                "status": "success",
                "count": 0,
                "results": []
            }

        # format response
        all_questions = [item["question"] for item in response.data]
        all_ids = [item["id"] for item in response.data]

        main_emb = similarity_model.encode(request.question, convert_to_tensor=True)
        other_embs = similarity_model.encode(all_questions, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(main_emb, other_embs)[0].tolist()

        # Combine question, id, and similarity score
        results = [
            {"id": qid, "question": q, "similarity": sim}
            for qid, q, sim in zip(all_ids, all_questions, similarities)
            if sim >= 0.7 #will return items with greater .5 similarity 
        ]

        # Sort results by similarity score in descending order
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)

        return {
            "status": "success",
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))