import os
import json
import re
import textwrap 
import google.generativeai as genai
import retriever  # Imports your local retriever.py

# --- Configuration ---
# Use the model that you confirmed works
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

if API_KEY:
    genai.configure(api_key=API_KEY)
    generation_model = genai.GenerativeModel(MODEL_NAME)
else:
    generation_model = None

def clean_json_string(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

def parse_json_response(response_text):
    print(f"DEBUG - RAW LLM OUTPUT: {response_text}") # SEE THIS IN LOGS
    cleaned_text = clean_json_string(response_text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        return {"intent": "semantic", "query": response_text}

def handle_user_query(user_question):
    if not generation_model:
        return "System Error: Gemini API Key is missing."

    # --- Step 1: Parse Intent (IMPROVED PROMPT) ---
    parsing_prompt = textwrap.dedent(f"""
    You are a PostgreSQL expert and database agent.
    
    Table: events
    Columns: 
    - name_of_event (Text)
    - date_of_event (Date, format YYYY-MM-DD)
    - event_domain (Text, e.g., 'AI', 'Coding')
    - description_insights (Text)
    - venue (Text)
    
    TASK: Classify the query and generate the payload.
    
    RULES FOR INTENT:
    1. "structured": Use this for ANY query involving Dates, Counts, Lists of names, or Filtering by domain.
       - You MUST write a valid PostgreSQL query in the 'query' field.
       - For MONTH/YEAR: Use `EXTRACT(MONTH FROM date_of_event) = X`.
       - Example: "Events in September 2025" -> SELECT name_of_event, date_of_event FROM events WHERE EXTRACT(MONTH FROM date_of_event) = 9 AND EXTRACT(YEAR FROM date_of_event) = 2025;
    
    2. "semantic": Use this ONLY for abstract questions like "What is the vibe?", "Tell me about AI", "Summary of perks".
       - The 'query' field should be keywords.

    User Question: "{user_question}"
    
    Output JSON ONLY: {{ "intent": "...", "query": "..." }}
    """)

    try:
        parse_resp = generation_model.generate_content(parsing_prompt)
        parsed = parse_json_response(parse_resp.text)
    except Exception as e:
        return f"Model Error: {e}"

    # --- Step 2: Retrieve ---
    intent = parsed.get("intent", "semantic")
    query = parsed.get("query", user_question)
    
    print(f"DEBUG - INTENT: {intent}") 
    print(f"DEBUG - QUERY: {query}")
    
    context = ""
    if intent == "structured":
        # Run SQL
        try:
            results = retriever.query_relational_db(query)
            context = f"Database Results: {str(results)}"
        except Exception as e:
            context = f"SQL Error: {e}"
            
    elif intent == "semantic":
        # Run Vector Search
        results = retriever.query_vector_db(query)
        context = "\n\n".join(results) if results else "No relevant documents found."
    
    else:
        context = "Could not parse intent."

    # --- Step 3: Generate Answer ---
    final_prompt = f"""
    User Question: {user_question}
    
    Context (Database Data):
    {context}
    
    Instructions:
    - If the context contains a list of database tuples (e.g. [('Event A', '2025-09-12')]), format them into a nice readable list.
    - If the context is empty or says "No results", tell the user politely.
    """
    
    try:
        final_resp = generation_model.generate_content(final_prompt)
        return final_resp.text
    except Exception as e:
        return f"Generator Error: {e}"
