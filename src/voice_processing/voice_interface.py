"""
Voice processing module for speech recognition and text-to-speech.
"""

import os
import io
import wave
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path
import subprocess

import speech_recognition as sr
import pygame
from pydub import AudioSegment
from TTS.api import TTS
from gtts import gTTS
from loguru import logger

from ..config import settings, SUPPORTED_LANGUAGES


class SpeechRecognizer:
    """Speech recognition using multiple engines."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        logger.info("Speech recognizer initialized")
    
    def listen_for_speech(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """
        Listen for speech from microphone.
        
        Args:
            timeout: Seconds to wait for speech to start
            phrase_time_limit: Maximum seconds for the phrase
            
        Returns:
            Recognized text or None
        """
        try:
            logger.info("Listening for speech...")
            
            with self.microphone as source:
                # Listen for audio
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            # Recognize speech
            try:
                # Try with Google Speech Recognition first
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Recognized speech: {text}")
                return text
                
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Error with speech recognition service: {str(e)}")
                
                # Fallback to offline recognition if available
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    logger.info(f"Recognized speech (offline): {text}")
                    return text
                except:
                    logger.error("Offline speech recognition also failed")
                    return None
                    
        except sr.WaitTimeoutError:
            logger.warning("No speech detected within timeout")
            return None
        except Exception as e:
            logger.error(f"Error during speech recognition: {str(e)}")
            return None
    
    def transcribe_audio_file(self, audio_path: str, language: str = 'en') -> Optional[str]:
        """
        Transcribe audio from file.
        
        Args:
            audio_path: Path to audio file
            language: Language code for recognition
            
        Returns:
            Transcribed text or None
        """
        try:
            # Load audio file
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            
            # Map language codes for Google Speech Recognition
            lang_map = {
                'en': 'en-US',
                'es': 'es-ES',
                'fr': 'fr-FR',
                'de': 'de-DE',
                'it': 'it-IT',
                'pt': 'pt-PT',
                'ru': 'ru-RU',
                'zh': 'zh-CN',
                'ja': 'ja-JP',
                'ko': 'ko-KR',
                'ar': 'ar-SA',
                'hi': 'hi-IN'
            }
            
            google_lang = lang_map.get(language, 'en-US')
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio, language=google_lang)
            logger.info(f"Transcribed audio: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Error transcribing audio file: {str(e)}")
            return None


class TextToSpeech:
    """Text-to-speech using multiple engines."""
    
    def __init__(self):
        self.tts_engine = settings.TTS_ENGINE
        self.audio_dir = Path(settings.AUDIO_DIR)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Initialize Coqui TTS if specified
        if self.tts_engine == 'coqui':
            self._init_coqui_tts()
        
        logger.info(f"TTS engine initialized: {self.tts_engine}")
    
    def _init_coqui_tts(self):
        """Initialize Coqui TTS model."""
        try:
            self.coqui_tts = TTS(model_name=settings.TTS_MODEL)
            logger.info("Coqui TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS model: {str(e)}")
            self.tts_engine = 'gtts'  # Fallback to gTTS
    
    def generate_speech(self, text: str, language: str = 'en', output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            language: Language code
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file or None
        """
        if not text:
            return None
        
        if not output_path:
            timestamp = int(datetime.now().timestamp())
            output_path = self.audio_dir / f"tts_output_{timestamp}.wav"
        
        try:
            if self.tts_engine == 'coqui' and hasattr(self, 'coqui_tts'):
                return self._generate_coqui_speech(text, language, output_path)
            else:
                return self._generate_gtts_speech(text, language, output_path)
                
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return None
    
    def _generate_coqui_speech(self, text: str, language: str, output_path: str) -> Optional[str]:
        """Generate speech using Coqui TTS."""
        try:
            # Coqui TTS supports multiple languages
            self.coqui_tts.tts_to_file(
                text=text,
                file_path=str(output_path)
            )
            
            logger.info(f"Generated speech with Coqui TTS: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Coqui TTS generation failed: {str(e)}")
            # Fallback to gTTS
            return self._generate_gtts_speech(text, language, output_path)
    
    def _generate_gtts_speech(self, text: str, language: str, output_path: str) -> Optional[str]:
        """Generate speech using Google TTS."""
        try:
            # Map language codes for gTTS
            gtts_lang_map = {
                'en': 'en',
                'es': 'es',
                'fr': 'fr',
                'de': 'de',
                'it': 'it',
                'pt': 'pt',
                'ru': 'ru',
                'zh': 'zh',
                'ja': 'ja',
                'ko': 'ko',
                'ar': 'ar',
                'hi': 'hi'
            }
            
            gtts_lang = gtts_lang_map.get(language, 'en')
            
            # Generate speech
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            
            # Save to temporary MP3 file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_mp3_path = temp_file.name
                tts.save(temp_mp3_path)
            
            # Convert MP3 to WAV using pydub
            audio = AudioSegment.from_mp3(temp_mp3_path)
            audio.export(output_path, format="wav")
            
            # Clean up temporary file
            os.unlink(temp_mp3_path)
            
            logger.info(f"Generated speech with gTTS: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"gTTS generation failed: {str(e)}")
            return None
    
    def play_audio(self, audio_path: str) -> bool:
        """
        Play audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if playback started successfully
        """
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            logger.info(f"Playing audio: {audio_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
            return False
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return pygame.mixer.music.get_busy()
    
    def stop_audio(self):
        """Stop audio playback."""
        pygame.mixer.music.stop()


class VoiceInterface:
    """Complete voice interface combining STT and TTS."""
    
    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.tts = TextToSpeech()
        self.listening = False
    
    def listen_and_transcribe(self, timeout: int = 5, phrase_time_limit: int = 10) -> Dict[str, Any]:
        """
        Listen for speech and return transcription with metadata.
        
        Returns:
            Dictionary with transcription results
        """
        result = {
            'success': False,
            'text': '',
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.listening = True
            text = self.speech_recognizer.listen_for_speech(timeout, phrase_time_limit)
            
            if text:
                result['success'] = True
                result['text'] = text
            else:
                result['error'] = 'No speech recognized'
                
        except Exception as e:
            result['error'] = str(e)
        finally:
            self.listening = False
        
        return result
    
    def speak_text(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            language: Language code
            
        Returns:
            Dictionary with TTS results
        """
        result = {
            'success': False,
            'audio_path': None,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Generate speech
            audio_path = self.tts.generate_speech(text, language)
            
            if audio_path:
                # Play audio
                if self.tts.play_audio(audio_path):
                    result['success'] = True
                    result['audio_path'] = audio_path
                else:
                    result['error'] = 'Failed to play audio'
            else:
                result['error'] = 'Failed to generate speech'
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def is_listening(self) -> bool:
        """Check if currently listening for speech."""
        return self.listening
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self.tts.is_playing()
    
    def stop_speaking(self):
        """Stop current speech playback."""
        self.tts.stop_audio()
    
    def cleanup_old_audio_files(self, max_files: int = 10):
        """Clean up old audio files to save space."""
        try:
            audio_files = list(self.tts.audio_dir.glob("tts_output_*.wav"))
            audio_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Remove old files
            for file_to_remove in audio_files[max_files:]:
                try:
                    file_to_remove.unlink()
                    logger.info(f"Removed old audio file: {file_to_remove}")
                except Exception as e:
                    logger.error(f"Error removing audio file: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error during audio cleanup: {str(e)}")


def main():
    """Test voice interface."""
    voice_interface = VoiceInterface()
    
    print("Voice Interface Test")
    print("1. Say something...")
    
    # Test speech recognition
    result = voice_interface.listen_and_transcribe()
    
    if result['success']:
        text = result['text']
        print(f"You said: {text}")
        
        # Test text-to-speech
        response = f"You said: {text}. This is a test of the text-to-speech system."
        print(f"Response: {response}")
        
        tts_result = voice_interface.speak_text(response)
        
        if tts_result['success']:
            print("Speech generated and playing...")
            
            # Wait for speech to finish
            import time
            while voice_interface.is_speaking():
                time.sleep(0.1)
            
            print("Speech finished.")
        else:
            print(f"TTS failed: {tts_result['error']}")
    else:
        print(f"Speech recognition failed: {result['error']}")


if __name__ == "__main__":
    from datetime import datetime
    main()
