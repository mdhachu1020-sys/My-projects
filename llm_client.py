import time
from exceptions import LLMError

class LLMClient:
    def __init__(self, provider: str, api_key: str):
        self.provider = provider.lower()
        self.api_key = api_key
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            print("✅ Gemini LLM ready.")
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            print("✅ OpenAI LLM ready.")
        else:
            raise ValueError(f"Unknown provider '{self.provider}'. Use 'gemini' or 'openai'.")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        last_err = None
        for attempt in range(1, 4):
            try:
                if self.provider == "gemini":
                    resp = self._client.models.generate_content(
                        model="gemini-flash-lite-latest",
                        contents=prompt,
                    )
                    return resp.text.strip()
                else:
                    resp = self._client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                print(f"⚠️  LLM attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    time.sleep(2 ** attempt)
        raise LLMError(f"LLM failed after 3 attempts. Last: {last_err}")
