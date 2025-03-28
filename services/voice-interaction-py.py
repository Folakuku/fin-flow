import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

class VoiceInteractionService:
    """
    Voice interaction service for speech-to-text and text-to-speech
    """
    def __init__(self):
        """
        Initialize voice interaction service
        """
        self.recognizer = sr.Recognizer()
    
    def text_to_speech(self, text: str) -> str:
        """
        Convert text to speech audio file
        
        :param text: Text to convert to speech
        :return: Path to generated audio file
        """
        try:
            # Ensure visualizations directory exists
            os.makedirs('audio_outputs', exist_ok=True)
            
            # Generate unique filename
            audio_path = os.path.join('audio_outputs', f'tts_output_{hash(text)}.mp3')
            
            # Create text-to-speech object
            tts = gTTS(text=text, lang='en')
            
            # Save the audio file
            tts.save(audio_path)
            
            return audio_path
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return ""
    
    def speech_to_text(self, audio_file: str) -> str:
        """
        Convert speech audio file to text
        
        :param audio_file: Path to audio file
        :return: Transcribed text
        """
        try:
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                
                # Record the audio
                audio_data = self.recognizer.record(source)
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio_data)
                
                return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return "Sorry, I couldn't understand that."
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return "Sorry, there was an error processing the audio."
        except Exception as e:
            print(f"Unexpected error in speech recognition: {e}")
            return "Sorry, an unexpected error occurred."
