from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "ingu627/exaone4.0:32b"
SYSTEM_PROMPT = "You are a friendly Korean assistant helping students, who mostly learn formal and academic Korean in the classroom, learn and practice colloquial and conversational Korean /no_think."
history = []

class Message(BaseModel):
    text: str
@app.post("/chat")

def chat(msg: Message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                {"role": "user", "content": msg.text}]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.7, max_tokens=512,
    )
    answer = resp.choices[0].message.content
    history.append({"role": "user", "content": msg.text})
    history.append({"role": "assistant", "content": answer})
    return {"reply": answer}
