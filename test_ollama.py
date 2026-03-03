from bot.llm.ollama_service import OllamaService
from rich import print

service = OllamaService(
    host="http://172.10.168.100:11434"
)

query = "How's the weather for Taiwan Taipei today? answer it in zhtw chinese."

messages = [
    {"role": "user", "content": query}
]

result = service.run(
    messages=messages,
    model="qwen3:14b",
    enable_tools=["web_search", "web_fetch"],
    think=True
)

print("Thinking:", result["thinking"])
print("Answer:", result["content"])