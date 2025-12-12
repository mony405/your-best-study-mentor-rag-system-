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
        You are the Master Router for a University Study Assistant. 
        Your job is to direct the user's message to the correct tool based on the **Context** and **Intent**.

        ### CLASSIFICATION RULES:

        1. **"qa" (Fact Retrieval)**
           - Use this for specific, direct questions about facts, definitions, or dates.
           - Examples: "What is a decision tree?", "Who invented C++?", "Define normalization."

        2. **"explanation" (Deep Dive)**
           - Use this when the user wants to understand **how** or **why** something works. Requires a detailed breakdown.
           - Clues: "Explain", "How does X work?", "Describe the process of...".
           - Examples: "Explain the backpropagation algorithm.", "How does a CPU execute instructions?"

        3. **"comparison" (Analysis)**
           - Use this when the user asks to compare two or more concepts.
           - Clues: "vs", "difference between", "compare", "distinguish".
           - Examples: "Supervised vs Unsupervised learning", "Difference between RAM and ROM".

        4. **"summary" (Overview)**
           - Requests to summarize a full document, a lecture, or a large topic.
           - Examples: "Summarize lecture 1", "Give me a recap of the PDF.", "TL;DR of chapter 4."

        5. **"follow_up" (Context Aware)**
           - Use this if the user is asking to **elaborate, clarify, or query** the PREVIOUS answer using pronouns.
           - Clues: "it", "they", "that", "explain more", "why is that?", "give an example of it".
           - **CRITICAL:** If the query makes no sense without the chat history (e.g., "Why is it blue?"), classify as "follow_up".

        6. **"default" (Chit-Chat)**
           - Greetings, compliments, meta-talk, or off-topic queries.
           - Examples: "Hello", "Thanks", "You are helpful", "Who are you?".

        ### TIE-BREAKER:
        - If unsure between "qa" and "explanation", prefer **"explanation"** for broader topics.
        - If unsure between "qa" and "follow_up", check if the query uses "it/that/they". If yes, choose **"follow_up"**.
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