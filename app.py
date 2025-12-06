import streamlit as st
import query_pipeline

st.set_page_config(page_title="Club Knowledge Agent", layout="centered")

st.title("🤖 Club Knowledge Agent")
st.caption("Ask me about upcoming events, workshops, and club details.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle input
if prompt := st.chat_input("What is the next AI event?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                response = query_pipeline.handle_user_query(prompt)
            except Exception as e:
                response = f"An error occurred: {e}"
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
