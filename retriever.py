import os
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

try:
    import streamlit as st
    HAS_ST = True
except ImportError:
    HAS_ST = False

# --- Configuration ---
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
RERANKER_MODEL = 'BAAI/bge-reranker-base'
TOP_K_RETRIEVAL = 25
TOP_K_FINAL = 5

# Cache models to prevent reloading
_EMBEDDING_MODEL = None
_RERANKER_MODEL = None

def get_models():
    global _EMBEDDING_MODEL, _RERANKER_MODEL
    if _EMBEDDING_MODEL is None:
        print("[Retriever] Loading models...")
        _EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        _RERANKER_MODEL = CrossEncoder(RERANKER_MODEL)
    return _EMBEDDING_MODEL, _RERANKER_MODEL

def get_db_connection():
    try:
        # Priority: Streamlit Secrets -> OS Environment
        if HAS_ST and "NEON_DB_URL" in st.secrets:
            dsn = st.secrets["NEON_DB_URL"]
        else:
            dsn = os.environ.get("NEON_DB_URL")
            
        return psycopg2.connect(dsn)
    except Exception as e:
        print(f"[Retriever] DB Connection Failed: {e}")
        return None

def query_vector_db(query_text):
    embedding_model, reranker_model = get_models()
    
    # Prefix required by BGE models
    query_vector = embedding_model.encode(f"Represent this sentence for searching relevant passages: {query_text}")
    
    conn = get_db_connection()
    if not conn:
        return ["Database connection error."]
        
    candidate_texts = set()
    
    try:
        with conn.cursor() as cur:
            register_vector(cur)
            
            # 1. Vector Search
            cur.execute(f"""
                SELECT search_text FROM events 
                ORDER BY embedding <-> %s LIMIT {TOP_K_RETRIEVAL}
            """, (query_vector,))
            candidate_texts.update(row[0] for row in cur.fetchall())
            
            # 2. Keyword Search
            cur.execute(f"""
                SELECT search_text FROM events, plainto_tsquery('english', %s) query
                WHERE query @@ to_tsvector('english', search_text)
                ORDER BY ts_rank_cd(to_tsvector('english', search_text), query) DESC 
                LIMIT {TOP_K_RETRIEVAL}
            """, (query_text,))
            candidate_texts.update(row[0] for row in cur.fetchall())
            
    except Exception as e:
        print(f"[Retriever] Error: {e}")
        return [f"Error querying database: {e}"]
    finally:
        conn.close()
    
    # 3. Reranking
    if not candidate_texts:
        return ["No relevant results found."]
        
    pairs = [[query_text, text] for text in candidate_texts]
    scores = reranker_model.predict(pairs)
    ranked_results = sorted(zip(candidate_texts, scores), key=lambda x: x[1], reverse=True)
    
    return [text for text, score in ranked_results[:TOP_K_FINAL]]

def query_relational_db(sql_query):
    conn = get_db_connection()
    if not conn:
        return [("Database error",)]
    try:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            return cur.fetchall()
    except Exception as e:
        return [f"SQL Error: {e}"]
    finally:
        conn.close()
