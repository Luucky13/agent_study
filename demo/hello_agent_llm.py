import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class HelloAgentsLLM:
    def __init__(self, model: str = None, url: str = None, api_key: str = None, timeout: int = 60):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL_ID") 
        self.url = url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.client = OpenAI(api_key=self.api_key, base_url=self.url, timeout = self.timeout)

    def think(self, prompt: List[Dict[str, str]], temperature: float = 0):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=temperature,
                stream=True,
            )
            print("大模型响应成功！！！")
            collected_resp = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                collected_resp.append(content)
            print()
            return "".join(collected_resp)
        except Exception as e:
            print(f"Error in using llm: {e}")
            return None

if __name__ == "__main__":
    try:
        llm = HelloAgentsLLM()
        test_response = llm.think([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the meaning of life?"}])
        print("------正在调用llm-------")
        if test_response:
            print("响应如下：")
            print(test_response) 
    except ValueError as e:
        print(f"Invalid value: {e}")

   