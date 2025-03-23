import streamlit as st
import requests
import json

# Set the base URL of your FastAPI server. Adjust if needed.
PROCESS_URL = "https://techiespark-stock-prediction-process-management.hf.space/process-management/process-input"
TTS_URL = "https://techiespark-stock-predictor-hindi-tts-management.hf.space/synthesize"

st.title("Stock Predictor")

st.markdown("## 1. Scrape Articles and Download JSON Response")
with st.form("articles_form"):
    search_query = st.text_input("Search Query")
    author = st.text_input("Author")
    from_date = st.text_input("From Date (YYYY-MM-DD)")
    to_date = st.text_input("To Date (YYYY-MM-DD)")
    max_articles = st.number_input("Max Articles", value=10, step=1)
    
    submitted_articles = st.form_submit_button("Get Articles")

if submitted_articles:
    if not search_query:
        st.error("Please enter a search query.")
    else:
        payload = {
            "search_query": search_query,
            "author": author if author else None,
            "from_date": from_date if from_date else None,
            "to_date": to_date if to_date else None,
            "max_articles": max_articles
        }
        st.info("Requesting articles...")
        try:
            response = requests.post(PROCESS_URL, json=payload)
            if response.status_code == 200:
                file_bytes = response.content
                json_data = json.loads(file_bytes)
                # Store data in session_state
                st.session_state.json_data = json_data
                st.session_state.articles_file_bytes = file_bytes
                st.success("Articles JSON is ready!")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Place the download button outside the form.
if "articles_file_bytes" in st.session_state:
    st.download_button(
        label="Download Articles JSON",
        data=st.session_state.articles_file_bytes,
        file_name=f"news_summary_{search_query}.json",
        mime="application/json"
    )

st.markdown("---")
st.markdown("## 2. TTS Synthesis and Download Audio")
if "json_data" not in st.session_state or not st.session_state.json_data.get("hindi_translation"):
    st.warning("Please retrieve articles with a valid Hindi translation before synthesizing speech.")
else:
    with st.form("tts_form"):
        tts_filename = st.text_input("Output Filename", "output.wav")
        submitted_tts = st.form_submit_button("Synthesize Speech")
    
    if submitted_tts:
        hindi_text = st.session_state.json_data["hindi_translation"]
        payload = {"text": hindi_text, "filename": tts_filename}
        st.info("Synthesizing speech...")
        try:
            response = requests.post(TTS_URL, json=payload)
            if response.status_code == 200:
                file_bytes = response.content
                # Store synthesized audio bytes in session_state
                st.session_state.audio_file_bytes = file_bytes
                st.success("Audio file is ready!")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    # Place the download button outside the form.
    if "audio_file_bytes" in st.session_state:
        st.download_button(
            label="Download Audio File",
            data=st.session_state.audio_file_bytes,
            file_name=tts_filename,
            mime="audio/wav"
        )
