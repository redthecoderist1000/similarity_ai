from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware
# from sympy import true
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import gc
import math
import numpy as np
import torch

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
        # fetch only needed columns and limit results to protect memory
        MAX_RESULTS = int(os.getenv("MAX_RESULTS", "500"))  # tune as needed
        response = supabase.table('tbl_question')\
            .select('id, question')\
            .eq('blooms_category', request.blooms_category)\
            .eq('repository', request.repository)\
            .eq('lesson_id', request.lesson_id)\
            .limit(MAX_RESULTS)\
            .execute()

        if not response.data:
            return {"status": "success", "count": 0, "results": []}

        all_questions = [item["question"] for item in response.data]
        all_ids = [item["id"] for item in response.data]

        # batch encode to avoid OOM; prefer numpy arrays on CPU
        batch_size = int(os.getenv("EMB_BATCH_SIZE", "64"))
        # encode main query
        with torch.no_grad():
            main_emb = similarity_model.encode(request.question, convert_to_numpy=True)

            # encode others in batches and collect numpy embeddings
            other_embs_list = []
            for i in range(0, len(all_questions), batch_size):
                batch = all_questions[i : i + batch_size]
                emb = similarity_model.encode(batch, convert_to_numpy=True)
                other_embs_list.append(emb)
            other_embs = np.vstack(other_embs_list)

        # compute cosine similarities (numpy) to reduce torch overhead
        # normalize
        def normalize(a):
            norms = np.linalg.norm(a, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return a / norms

        main_norm = main_emb / (np.linalg.norm(main_emb) or 1.0)
        other_norms = normalize(other_embs)
        similarities = (other_norms @ main_norm).tolist()

        # combine, filter and sort
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        results = [
            {"id": qid, "question": q, "similarity": float(sim)}
            for qid, q, sim in zip(all_ids, all_questions, similarities)
            if sim >= threshold
        ]
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)

        # cleanup large objects
        if 'other_questions' in locals():
            del other_questions
        del other_embs_list
        del other_embs
        del main_emb
        del all_questions, all_ids, similarities
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"status": "success", "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))