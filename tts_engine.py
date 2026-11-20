import os

class TTSEngine:
    def __init__(self, engine_type: str = "pyttsx3"):
        self.engine_type = engine_type.lower()
        self._engine = None
        if self.engine_type == "pyttsx3":
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 165)
            voices = self._engine.getProperty("voices")
            if voices:
                self._engine.setProperty("voice", voices[0].id)
            print("✅ TTS engine ready.")

    def speak(self, text: str):
        print(f"\n🔊 {text}\n")
        if self.engine_type == "pyttsx3":
            self._engine.say(text)
            self._engine.runAndWait()
        elif self.engine_type == "gtts":
            self._speak_gtts(text)

    def _speak_gtts(self, text: str):
        from gtts import gTTS
        import tempfile, pygame
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tmp = fp.name
        tts.save(tmp)
        pygame.mixer.init()
        pygame.mixer.music.load(tmp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
        os.remove(tmp)
