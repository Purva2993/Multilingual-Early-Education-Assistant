"""
Streamlit frontend for the Multilingual AI Voice Assistant.
"""

import streamlit as st
import requests
import json
from datetime import datetime
import time
import base64
from typing import Dict, Any, Optional
from io import BytesIO

import streamlit_webrtc
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
from pydub import AudioSegment

from ..config import settings


# Page configuration
st.set_page_config(
    page_title="Multilingual AI Voice Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"


def get_system_status() -> Optional[Dict[str, Any]]:
    """Get system status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Could not connect to API: {str(e)}")
        return None


def query_api(query: str, include_audio: bool = False, max_sources: int = 5) -> Optional[Dict[str, Any]]:
    """Send query to API."""
    try:
        payload = {
            "query": query,
            "include_audio": include_audio,
            "max_sources": max_sources
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error querying API: {str(e)}")
        return None


def get_supported_languages() -> Dict[str, str]:
    """Get supported languages from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/languages", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"en": "English"}
    except:
        return {"en": "English"}


def trigger_crawl() -> Optional[Dict[str, Any]]:
    """Trigger data crawling."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/crawl",
            json={"force_recrawl": True},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error triggering crawl: {str(e)}")
        return None


def display_sources(sources: list):
    """Display source information."""
    if not sources:
        return
    
    st.subheader("📚 Sources")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source.get('title', 'Untitled')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Source:** {source.get('source_name', 'Unknown')}")
                if source.get('url'):
                    st.write(f"**URL:** {source['url']}")
                st.write(f"**Preview:** {source.get('text_preview', '')}")
            
            with col2:
                similarity = source.get('similarity_score', 0)
                st.metric("Relevance", f"{similarity:.2%}")


def display_query_info(query_info: Dict[str, Any]):
    """Display query processing information."""
    with st.expander("🔍 Query Processing Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Original Query:** {query_info.get('original_query', '')}")
            st.write(f"**Detected Language:** {query_info.get('language_name', 'Unknown')} ({query_info.get('detected_language', 'unknown')})")
            st.write(f"**Confidence:** {query_info.get('language_confidence', 0):.2%}")
        
        with col2:
            if query_info.get('translated_query'):
                st.write(f"**Translated Query:** {query_info['translated_query']}")
            st.write(f"**Search Query:** {query_info.get('search_query', '')}")
            st.write(f"**Processed At:** {query_info.get('processed_at', '')}")


def audio_recorder_component():
    """Audio recording component using streamlit-webrtc."""
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    webrtc_ctx = webrtc_streamer(
        key="speech-recognition",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"audio": True, "video": False},
    )
    
    if webrtc_ctx.audio_receiver:
        st.info("🎤 Recording... Click 'Stop' when done speaking.")
        
        audio_frames = []
        while True:
            try:
                audio_frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                if audio_frame is None:
                    break
                audio_frames.append(audio_frame)
            except:
                break
        
        if audio_frames:
            # Process audio frames
            st.success("✅ Audio recorded! Processing...")
            return audio_frames
    
    return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🎓 Multilingual AI Voice Assistant")
    st.markdown("*Ask questions about early childhood education in any language*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # System status
        status = get_system_status()
        if status:
            if status['ollama_status'] == 'connected':
                st.success("🟢 System Online")
            else:
                st.error("🔴 LLM Offline")
            
            # Database info
            db_info = status.get('database', {})
            if db_info:
                st.info(f"📊 Database: {db_info.get('total_vectors', 0)} documents indexed")
        else:
            st.error("🔴 API Disconnected")
        
        # Settings
        st.subheader("Query Settings")
        max_sources = st.slider("Max Sources", 1, 10, 5)
        include_audio = st.checkbox("Generate Audio Response", value=False)
        
        # Language selection
        languages = get_supported_languages()
        selected_lang = st.selectbox(
            "Preferred Language",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            index=0
        )
        
        # Data management
        st.subheader("Data Management")
        if st.button("🔄 Update Knowledge Base"):
            with st.spinner("Updating knowledge base..."):
                result = trigger_crawl()
                if result:
                    st.success("✅ Update started!")
                    st.json(result)
                else:
                    st.error("❌ Update failed")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🎤 Voice Query", "📊 Analytics"])
    
    with tab1:
        st.header("Text Query")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "sources" in message:
                    display_sources(message["sources"])
        
        # Query input
        if prompt := st.chat_input("Ask about early childhood education..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = query_api(prompt, include_audio, max_sources)
                
                if response and not response.get('error'):
                    # Display answer
                    st.markdown(response['answer'])
                    
                    # Display query info
                    display_query_info(response['query_info'])
                    
                    # Display sources
                    display_sources(response['sources'])
                    
                    # Audio playback
                    if response.get('audio_url') and include_audio:
                        audio_url = f"{API_BASE_URL}{response['audio_url']}"
                        st.audio(audio_url)
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response['answer'],
                        "sources": response['sources'],
                        "processing_time": response['processing_time']
                    })
                    
                    # Performance info
                    st.caption(f"⏱️ Processed in {response['processing_time']:.2f} seconds")
                    
                else:
                    error_msg = response.get('error', 'Unknown error') if response else 'API unavailable'
                    st.error(f"❌ Error: {error_msg}")
    
    with tab2:
        st.header("Voice Query")
        st.info("🎤 Click the button below to start recording your question")
        
        # Voice recording interface
        recording_placeholder = st.empty()
        
        if st.button("🎤 Start Recording", key="voice_record"):
            with recording_placeholder:
                st.warning("🎤 Recording... (Feature in development)")
                st.info("Voice recording will be available in the next update. Please use the text interface for now.")
        
        # File upload option
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=['wav', 'mp3', 'ogg', 'm4a'],
            help="Upload an audio file with your question"
        )
        
        if uploaded_file:
            st.audio(uploaded_file)
            
            if st.button("🔍 Process Audio"):
                with st.spinner("Processing audio..."):
                    # TODO: Implement audio processing
                    st.info("Audio processing will be implemented in the next update.")
    
    with tab3:
        st.header("System Analytics")
        
        if status:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                db_info = status.get('database', {})
                st.metric(
                    "Documents Indexed",
                    db_info.get('total_vectors', 0)
                )
            
            with col2:
                st.metric(
                    "Supported Languages",
                    len(status.get('supported_languages', []))
                )
            
            with col3:
                ollama_status = status.get('ollama_status', 'unknown')
                st.metric(
                    "LLM Status",
                    "Online" if ollama_status == 'connected' else "Offline"
                )
            
            # System details
            st.subheader("System Details")
            st.json(status)
        else:
            st.error("Unable to load system status")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using FastAPI, Streamlit, and open-source AI models | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
