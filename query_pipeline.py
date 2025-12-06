import os
import json
import re
import textwrap 
import google.generativeai as genai
import retriever  # Imports your local retriever.py

# --- Configuration ---
# We use the stable 1.5-flash-001 or the newer 2.5-flash if available
# If 2.5 fails for you, switch this string to 'gemini-1.5-flash-001'
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
    """
    Cleans the model response to extract just the JSON.
    Removes markdown code blocks (```json ... ```).
    """
    # Remove markdown code blocks
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    
    # Find the first opening brace and last closing brace
    start = text.find("{")
    end = text.rfind("}")
    
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

def parse_json_response(response_text):
    print(f"RAW PARSER OUTPUT: {response_text}") # Debug print for logs
    
    cleaned_text = clean_json_string(response_text)
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Fallback: If JSON fails, assume semantic search
        print("JSON Decode Failed. Defaulting to Semantic Search.")
        return {"intent": "semantic", "query": response_text}

def handle_user_query(user_question):
    if not generation_model:
        return "System Error: Gemini API Key is missing. Check your Streamlit secrets."

    # --- Step 1: Parse Intent ---
    parsing_prompt = textwrap.dedent(f"""
    You are a query parsing agent for a university club database.
    
    Task: Convert the user's question into a JSON object.
    
    Rules:
    1. If the user asks for specific data (dates, counts, names), use "structured" intent and write a PostgreSQL query.
       - Table: events
       - Columns: event_id, name_of_event, event_domain, date_of_event, faculty_coordinators, venue
       - Example: "Who is running the AI event?" -> {{"intent": "structured", "query": "SELECT faculty_coordinators FROM events WHERE event_domain ILIKE '%AI%'"}}
    
    2. If the user asks about concepts, summaries, or "what is", use "semantic" intent and extract keywords.
       - Example: "Tell me about the hackathon" -> {{"intent": "semantic", "query": "hackathon details"}}
    
    User Question: "{user_question}"
    
    Output strictly valid JSON. No markdown. No explanations.
    """)

    try:
        parse_resp = generation_model.generate_content(parsing_prompt)
        parsed = parse_json_response(parse_resp.text)
    except Exception as e:
        return f"Model Error ({MODEL_NAME}): {e}"

    # --- Step 2: Retrieve ---
    intent = parsed.get("intent", "semantic") # Default to semantic if missing
    query = parsed.get("query", user_question)
    
    print(f"INTENT: {intent} | QUERY: {query}") # Debug log
    
    context = ""
    if intent == "semantic":
        # Search vector DB
        results = retriever.query_vector_db(query)
        if not results or results == ["No relevant results found."]:
            context = "No relevant documents found in the database."
        else:
            context = "\n\n".join(results)
            
    elif intent == "structured":
        # Search SQL DB
        results = retriever.query_relational_db(query)
        context = f"Database returned: {str(results)}"
    
    else:
        context = "Could not parse intent."

    # --- Step 3: Generate Answer ---
    final_prompt = f"""
    You are a helpful Club Assistant. Answer the user's question based ONLY on the context below.
    
    User Question: {user_question}
    
    Context from Database:
    {context}
    
    If the context says "No relevant documents" or is empty, politely say you don't have that info.
    """
    
    try:
        final_resp = generation_model.generate_content(final_prompt)
        return final_resp.text
    except Exception as e:
        return f"Generator Error: {e}"

if __name__ == "__main__":
    # Test locally
    while True:
        q = input("Question: ")
        print(handle_user_query(q))
