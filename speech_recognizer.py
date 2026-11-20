import speech_recognition as sr
from exceptions import SpeechRecognitionError, MicrophoneNotFoundError

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        try:
            mics = sr.Microphone.list_microphone_names()
            if not mics:
                raise MicrophoneNotFoundError("No microphone detected.")
        except OSError as e:
            raise MicrophoneNotFoundError(f"Microphone error: {e}")

    def listen(self, timeout: float = 5.0) -> str:
        with sr.Microphone() as source:
            print("🎤 Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"🎤 Listening (up to {timeout}s)...")
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
            except sr.WaitTimeoutError:
                raise SpeechRecognitionError("No speech detected within timeout.")
        try:
            return self.recognizer.recognize_google(audio).strip().lower()
        except sr.UnknownValueError:
            raise SpeechRecognitionError("Could not understand audio.")
        except sr.RequestError as e:
            raise SpeechRecognitionError(f"STT service error: {e}")
