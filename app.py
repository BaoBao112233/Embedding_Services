from typing import List, Optional, Union
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# The app.py file lives at the repository root, so use the file's parent
# directory as the repo root. Using parents[1] moved one directory too high
# and caused lookups like '/home/baobao/Projects/models/...' which don't exist.
repo_root = Path(__file__).resolve().parent
local_snapshot = None
MODEL_NAME_DEFAULT = repo_root / "models" / "embeddinggemma-300m"

app = FastAPI(title="Embedding Service (mini)")


class EmbedRequest(BaseModel):
    query: Optional[str] = None
    docs: Optional[List[str]] = None


class EmbedResponse(BaseModel):
    similarities: List[List[float]]


class QueryEmbResponse(BaseModel):
    query_embeddings: List[float]


class DocumentsEmbResponse(BaseModel):
    document_embeddings: List[List[float]]


class FullEmbeddingsResponse(BaseModel):
    query_embeddings: List[float]
    document_embeddings: List[List[float]]
    similarities: List[List[float]]

# simple cache so we don't reload the model on every request
_model_cache = None


def get_similarity():
    """Return a loaded SentenceTransformer model.

    Behavior:
    - If LOCAL_GEMMA_PATH env var is set, use that path.
    - Else if `MODEL_NAME_DEFAULT` points to a local model folder with a
      `snapshots/*` subfolder, use the first snapshot found.
    - Otherwise treat `MODEL_NAME_DEFAULT` as a Hub id/string.

    Always convert Path objects to str before passing to SentenceTransformer
    to avoid TypeError (SentenceTransformer expects a string/identifier).
    """
    from sentence_transformers import SentenceTransformer
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    # allow explicit override via env var
    env_path = os.environ.get("LOCAL_GEMMA_PATH")
    model_path = None
    if env_path:
        model_path = env_path
    else:
        # MODEL_NAME_DEFAULT might be a Path pointing to a local model.
        # Be resilient: try MODEL_NAME_DEFAULT itself and also look for the
        # model folder in parent directories in case repo_root calculation
        # differs between running contexts.
        candidate_paths = []
        try:
            candidate_paths.append(Path(MODEL_NAME_DEFAULT))
        except Exception:
            pass

        # also try searching upward a couple levels from this file's parent
        for root in list(repo_root.parents)[:2]:
            candidate_paths.append(root / "models" / "embeddinggemma-300m")

        for candidate in candidate_paths:
            if isinstance(candidate, Path) and candidate.exists():
                for p in candidate.rglob("snapshots/*"):
                    if p.is_dir():
                        model_path = str(p)
                        break
            if model_path:
                break

    if model_path is None:
        # fall back to using the default (convert to str to avoid Path issues)
        model_path = str(MODEL_NAME_DEFAULT)

    _model_cache = SentenceTransformer(str(model_path))
    return _model_cache


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embeddings", response_model=FullEmbeddingsResponse)
def embed(req: EmbedRequest):
    if req.query is None or req.docs is None:
        raise HTTPException(status_code=400, detail="Provide 'query' or 'docs' in body")

    model = get_similarity()

    query = req.query
    documents = req.docs

    try:
        query_embeddings = model.encode_query(query)
        document_embeddings = model.encode_document(documents)
        # ensure lists of floats
        similarities = model.similarity(query_embeddings, document_embeddings)
    
        return {
            "query_embeddings": query_embeddings.tolist(),
            "document_embeddings": document_embeddings.tolist(),
            "similarities": similarities.tolist() if hasattr(similarities, 'tolist') else similarities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/query_embeddings")
def query_embeddings(query: str):
    model = get_similarity()
    try:
        query_embeddings = model.encode_query(query)
        return {"query_embeddings": query_embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents_embeddings")
def documents_embeddings(documents: List[str]):
    model = get_similarity()
    try:
        document_embeddings = model.encode_document(documents)    
        return {"document_embeddings": document_embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/similarity", response_model=EmbedResponse)
def similarity(query_embeddings: List[float], documents_embeddings: List[float]):
    model = get_similarity()
    try:
        # ensure lists of floats
        similarities = model.similarity(query_embeddings, documents_embeddings)
        return {"similarities": similarities.tolist() if hasattr(similarities, 'tolist') else similarities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Preload the model at application startup so we fail fast and log any
# issues (version mismatch, missing local snapshot, auth) rather than
# returning obscure 500s on the first request.
@app.on_event("startup")
def _startup_load_model():
    try:
        # This will populate the _model_cache by calling get_similarity()
        get_similarity()
        logging.info("Model loaded successfully at startup")
    except Exception as e:
        # Log the exception; container will continue to run but the error
        # will be visible in logs and the service will return 500s until
        # the problem is fixed. You can optionally raise here to crash the
        # container and let orchestration restart it.
        logging.exception("Failed to load model at startup: %s", e)
