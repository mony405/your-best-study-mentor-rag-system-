from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from config import Config

# --- Data Model for Output ---
class RouteQuerySchema(BaseModel):
    task: Literal["qa", "summary", "explanation", "comparison", "follow_up", "default"] = Field(
        ..., 
        description="The classification of the user's query."
    )

class QueryRouter:
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0)
        self.structured_llm = self.llm.with_structured_output(RouteQuerySchema)
        
        system_prompt = """
        You are the Master Router. Classify the User Question into ONE category:
        
        1. RAG TASKS (Need Context):
        - qa: Specific questions about facts.
        - summary: Requests to summarize documents.
        - explanation: Requests to explain concepts.
        - comparison: Comparing two concepts.
        
        2. NO RAG (Chat History Only):
        - follow_up: "Why?", "Explain that again", "Give an example".
        - default: Greetings, chit-chat, off-topic.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Chat History:\n{history}\n\nUser Question:\n{question}")
        ])

    def route_query(self, user_query: str, history_text: str) -> str:
        try:
            chain = self.prompt | self.structured_llm
            result = chain.invoke({"question": user_query, "history": history_text})
            return result.task
        except Exception as e:
            print(f"⚠️ Router Error: {e}. Defaulting to 'qa'.")
            return "qa"