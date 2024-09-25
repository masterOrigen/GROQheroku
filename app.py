import os
from dotenv import find_dotenv, load_dotenv
import streamlit as st
from typing import Generator
from groq import Groq

_ = load_dotenv(find_dotenv())
st.set_page_config(page_icon="📃", layout="wide", page_title="IA Chat Bot...")

client = Groq(
    api_key=os.environ['GROQ_API_KEY'],
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define model details
model = "llama3-70b-8192"
max_tokens = 8192  # Maximum tokens for LLaMA3-70b

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Initial greeting
if not st.session_state.messages:
    initial_message = "Bienvenido, cuéntame que información detallada necesitas que busque y filtre para ti en internet?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})
    with st.chat_message("assistant"):
        st.markdown(initial_message)

if prompt := st.chat_input("Ingresa tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True,
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="❌")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response}
        )

# Add a "Clear Chat" button
if st.button("Limpiar Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
