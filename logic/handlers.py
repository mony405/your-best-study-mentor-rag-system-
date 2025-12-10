from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from config import Config

# --- Base Interface ---
class BaseHandler(ABC):
    def __init__(self, db_instance):
        self.db = db_instance
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)

    @abstractmethod
    def handle(self, query: str, history_str: str) -> str:
        pass

# --- 1. QA Handler (Standard RAG) ---
class QATaskHandler(BaseHandler):
    def handle(self, query, history_str):
        retriever = self.db.get_db().as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
        
        template = """Answer the question based on the Context and History.
        
        Context: {context}
        History: {history}
        Question: {question}
        
        If the answer is not in the context, say so.
        Answer:"""
        
        chain = (
            RunnablePassthrough.assign(context=lambda x: retriever.invoke(x["question"]))
            | ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"question": query, "history": history_str})

# --- 2. Summary Handler (RAG with different prompt) ---
class SummaryTaskHandler(BaseHandler):
    def handle(self, query, history_str):
        # We might want more docs for a summary
        retriever = self.db.get_db().as_retriever(search_kwargs={"k": Config.SUMMARY_K})
        
        template = """Summarize the provided context relevant to the user's request.
        
        Context: {context}
        Request: {question}
        
        Summary:"""
        
        chain = (
            RunnablePassthrough.assign(context=lambda x: retriever.invoke(x["question"]))
            | ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"question": query})

# --- 3. Chat Handler (No Retrieval) ---
class ChatTaskHandler(BaseHandler):
    def handle(self, query, history_str):
        # Pure LLM response
        template = """You are a helpful AI Study Mentor.
        
        Chat History:
        {history}
        
        User: {question}
        Answer:"""
        
        chain = (
            ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"question": query, "history": history_str})

# --- Registry (The Factory) ---
class TaskRegistry:
    def __init__(self, db_instance):
        self.handlers = {
            "qa": QATaskHandler(db_instance),
            "explanation": QATaskHandler(db_instance), # Re-use QA for now
            "comparison": QATaskHandler(db_instance),  # Re-use QA for now
            "summary": SummaryTaskHandler(db_instance),
            "default": ChatTaskHandler(db_instance),
            "follow_up": ChatTaskHandler(db_instance)
        }

    def get_handler(self, task_name):
        return self.handlers.get(task_name, self.handlers["default"])