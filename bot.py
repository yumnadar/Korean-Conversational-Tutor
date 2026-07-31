from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "ingu627/exaone4.0:32b"
SYSTEM_PROMPT = "You are a concise, friendly assistant. /no_think"

def reply(history, user_msg):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                {"role": "user", "content": user_msg}]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.8, max_tokens=512,
    )
    return resp.choices[0].message.content

history = []

while True:
    user_msg = input("You: ")
    answer = reply(history, user_msg)
    print("Bot:", answer)
    history += [{"role": "user", "content": user_msg},
                {"role": "assistant", "content": answer}]
