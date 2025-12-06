import os
import psycopg2
from typing import Dict, Any
import streamlit as st
from sentence_transformers import SentenceTransformer

# Constants
MODEL_NAME = "BAAI/bge-base-en-v1.5"

@st.cache_resource(show_spinner=False)
def _load_model():
    return SentenceTransformer(MODEL_NAME)

def _get_db_connection():
    try:
        if "NEON_DB_URL" in st.secrets:
            return psycopg2.connect(st.secrets["NEON_DB_URL"])
        return psycopg2.connect(os.environ.get("NEON_DB_URL"))
    except Exception as e:
        print(f"DB Error: {e}")
        return None

def add_new_event(form_data: Dict[str, Any]):
    # ... (Logic remains similar to previous, simplified for brevity) ...
    # This function is called by new_event.py
    # Re-using your logic for embedding generation + SQL insert
    try:
        model = _load_model()
        text = f"{form_data['name_of_event']} {form_data['description_insights']}"
        embedding = model.encode(text).tolist()
        
        conn = _get_db_connection()
        if not conn: return {"status": "error", "message": "No DB Connection"}
        
        with conn:
            with conn.cursor() as cur:
                # Simplified Insert for brevity - ensure your full columns match DB
                cur.execute("""
                    INSERT INTO events (event_id, name_of_event, description_insights, search_text, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    form_data['name_of_event'], 
                    form_data['name_of_event'], 
                    form_data['description_insights'],
                    text,
                    embedding
                ))
        return {"status": "success", "message": "Event Added"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
