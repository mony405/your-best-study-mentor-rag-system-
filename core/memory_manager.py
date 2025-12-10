from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from config import Config

class MemoryManager:
    _instance = None
    _memory = None

    def __new__(cls, *args, **kwargs):
        """Singleton Pattern: Ensures only one memory instance exists."""
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initializes memory only once."""
        if self._memory is None:
            print("🧠 Initializing Memory Manager...")
            self._memory = ConversationSummaryBufferMemory(
                llm=ChatOpenAI(model=Config.MODEL_NAME, temperature=0),
                max_token_limit=2000,
                memory_key="chat_history",
                return_messages=True
            )

    def get_formatted_history(self) -> str:
        """Returns history as a string (Human: ... AI: ...)."""
        messages = self._memory.load_memory_variables({})["chat_history"]
        return "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in messages])

    def save_interaction(self, input_text: str, output_text: str):
        """Saves the user input and AI response."""
        self._memory.save_context({"input": input_text}, {"output": output_text})

    def clear_memory(self):
        self._memory.clear()