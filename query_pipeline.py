import os
import json
import re
import textwrap 
import google.generativeai as genai
import retriever  # Imports the file from the same directory

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
if API_KEY:
    genai.configure(api_key=API_KEY)
    generation_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    generation_model = None

def parse_json_response(response_text):
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        return {"intent": "error", "query": "No JSON found"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"intent": "error", "query": "Invalid JSON"}

def handle_user_query(user_question):
    if not generation_model:
        return "Error: Gemini API Key not configured."

    # --- Step 1: Parse Intent ---
    parsing_prompt = textwrap.dedent(f"""
    You are a query parsing agent for a university club database.
    Schema: events(name_of_event, event_domain, date_of_event, description_insights, ...)
    RULES:
    1. structured: for dates, counts, names. Output SQL.
    2. semantic: for concepts, "about", "describe". Output keywords.
    User: "{user_question}"
    Output JSON: {{"intent": "...", "query": "..."}}
    """)

    try:
        parse_resp = generation_model.generate_content(parsing_prompt)
        parsed = parse_json_response(parse_resp.text)
    except Exception as e:
        return f"Parser Error: {e}"

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

    # --- Step 3: Generate ---
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
