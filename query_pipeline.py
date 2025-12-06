import os
import json
import re
import textwrap 
import google.generativeai as genai
import retriever  # Imports your local retriever.py

# --- Configuration ---
# 1. Select the Model
# Try 'gemini-2.5-flash' (Stable) or 'gemini-pro' (Legacy/Backup)
MODEL_NAME = 'gemini-2.5-flash' 

try:
    import streamlit as st
    HAS_ST = True
except ImportError:
    HAS_ST = False

def get_api_key():
    if HAS_ST and "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY")

API_KEY = get_api_key()

# Initialize Model
generation_model = None
if API_KEY:
    genai.configure(api_key=API_KEY)
    generation_model = genai.GenerativeModel(MODEL_NAME)
else:
    print("Warning: GEMINI_API_KEY is missing.")

def parse_json_response(response_text):
    # Clean the response to find the first JSON object
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        return {"intent": "error", "query": "No JSON found"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"intent": "error", "query": "Invalid JSON"}

def handle_user_query(user_question):
    if not generation_model:
        return "System Error: Gemini API Key is missing."

    # --- Step 1: Parse Intent ---
    parsing_prompt = textwrap.dedent(f"""
    You are a query parsing agent for a university club database.
    Schema: events(name_of_event, event_domain, date_of_event, description_insights, ...)
    
    RULES:
    1. structured: for dates, counts, names, specific facts. Output SQL.
    2. semantic: for concepts, "about", "describe", "summary". Output keywords.
    
    User: "{user_question}"
    
    Output JSON ONLY: {{"intent": "...", "query": "..."}}
    """)

    try:
        parse_resp = generation_model.generate_content(parsing_prompt)
        parsed = parse_json_response(parse_resp.text)
    except Exception as e:
        # Fallback if the specific model fails
        return f"Model Error ({MODEL_NAME}): {e}\nTry running check_models.py to see available models."

    # --- Step 2: Retrieve ---
    intent = parsed.get("intent")
    query = parsed.get("query")
    context = ""
    
    if intent == "semantic":
        context = "\n\n".join(retriever.query_vector_db(query))
    elif intent == "structured":
        context = str(retriever.query_relational_db(query))
    else:
        context = "Could not parse intent."

    # --- Step 3: Generate Answer ---
    final_prompt = f"""
    Answer based ONLY on context.
    Question: {user_question}
    Context: {context}
    """
    try:
        final_resp = generation_model.generate_content(final_prompt)
        return final_resp.text
    except Exception as e:
        return f"Generator Error: {e}"

if __name__ == "__main__":
    # Simple CLI for testing
    print(f"--- Club Knowledge Agent ({MODEL_NAME}) ---")
    while True:
        q = input("You: ")
        if q.lower() in ["quit", "exit"]: break
        print("Agent:", handle_user_query(q))
