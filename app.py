import gradio as gr
from dotenv import load_dotenv
from config import Config
from core.database import DataBase
from core.memory_manager import MemoryManager
from logic.routing import QueryRouter
from logic.handlers import TaskRegistry

# Load Environment Variables (API Keys)
load_dotenv()

class StudyAssistantApp:
    def __init__(self):
        print("🚀 Initializing AI Study Mentor...")
        
        # 1. Initialize Core Systems
        self.db = DataBase()
        self.memory_manager = MemoryManager()
        
        # 2. Initialize Logic Systems
        self.router = QueryRouter()
        self.registry = TaskRegistry(self.db)

    def process_query(self, message, history):
        """
        The main pipeline: Get History -> Route -> Handle -> Save.
        """
        # A. Get formatted history for the Router & Prompts
        history_str = self.memory_manager.get_formatted_history()

        # B. Route the Query
        task = self.router.route_query(message, history_str)
        print(f"🚦 Routing Query: '{message}' -> Task: {task}")

        # C. Get the correct Handler
        handler = self.registry.get_handler(task)

        # D. Execute the Handler
        response_text = handler.handle(message, history_str)

        # E. Save interaction to Memory
        self.memory_manager.save_interaction(message, response_text)

        return response_text

# --- Launch Code ---
if __name__ == "__main__":
    app_instance = StudyAssistantApp()
    
    interface = gr.ChatInterface(
        fn=app_instance.process_query,
        title="AI Study Mentor 🧠",
        description="I can answer questions, summarize PDFs, and explain concepts.",
        theme="soft"
    )
    
    interface.launch(share=False)