from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

llm = ChatOllama(
        model="gemma3:12b-it-qat",
        base_url="http://localhost:11434",
        temperature=0
)

messages = [
        (
            "system",
            "You are a chicken that doesn't know english.",
        ),
        ("human", "Why did the chicken cross the road?"),
]

ai_msg = llm.invoke(messages)
print(ai_msg)
