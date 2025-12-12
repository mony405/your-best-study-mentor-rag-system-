from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from config import Config
from IPython.display import Markdown, display, update_display


# Base interface 
class BaseHandler(ABC):
    def __init__(self, db_instance):
        self.db = db_instance
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)

    @abstractmethod
    def handle(self, query: str, history_str: str) -> str:
        pass


class QATaskHandler(BaseHandler):
    def handle(self, query, history_str):
        retriever = self.db.get_db().as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
        
        template = """You are an expert AI Study Mentor. Your goal is to help students understand complex university topics.

        Instructions:
        1. **Format:** Write the response in **Markdown**.
        2. **Style:** Provide a clear, structured, friendly explanation with analogies.
        3. **Constraint:** Answer ONLY based on the context provided below. If the answer is not in the context, say "I couldn't find that in your lecture notes."
        4. **Structure:**
           - Start with a direct answer.
           - Use **Bold** for key terms.
           - Use bullet points for lists.
           - Use `## Headers` to separate sections.

        Context from Slides:
        {context}

        Chat History:
        {history}

        Student Question: {question}
        
        Answer:"""
        
        chain = (
            RunnablePassthrough.assign(context=lambda x: retriever.invoke(x["question"]))
            | ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke({"question": query, "history": history_str})


class SummaryTaskHandler(BaseHandler):
    def handle(self, query, history_str):
        retriever = self.db.get_db().as_retriever(search_kwargs={"k": Config.SUMMARY_K})
        
        template = """You are an expert AI Study Mentor. Summarize the provided content for a student.

        Instructions:
        1. **Format:** Write the response in **Markdown**.
        2. **Style:** Provide a clear, structured, friendly explanation with analogies where possible.
        3. **Structure:**
           - Use `## Title` for the main header.
           - Use bullet points for key takeaways.
           - Bold important terms.

        Context from Slides:
        {context}

        Request: {question}
        
        Summary:"""
        
        chain = (
            RunnablePassthrough.assign(context=lambda x: retriever.invoke(x["question"]))
            | ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"question": query})


class ChatTaskHandler(BaseHandler):
    def handle(self, query, history_str):
        template = """You are a helpful and enthusiastic AI Study Mentor.

        Instructions:
        1. **Format:** Write the response in **Markdown**.
        2. **Style:** Provide a clear, structured, friendly explanation with analogies.
        3. Guide the student back to studying or answer general questions helpfully.

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


class FollowupTaskHandler(BaseHandler):
    def handle(self, query, history_str):
        template = """You are a helpful AI assistant continuing a conversation.
        
        Instructions:
        1. **Format:** Write the response in **Markdown**.
        2. **Style:** Provide a clear, structured, friendly explanation with analogies.
        3. Use the conversation history to answer the user's new query naturally.
        
        Conversation History:
        {history}
        
        User's New Query: {question}
        
        Answer:"""
        
        chain = (
            ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"question": query, "history": history_str})
    

class ComparisonTaskHandler(BaseHandler):
    def handle(self, query, history_str):
        # We might need a bit more context for comparisons, so we use Config.RETRIEVER_K (e.g. 3 or 5)
        retriever = self.db.get_db().as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
        
        template = """You are an expert AI Study Mentor specialized in technical comparisons.
        
        Instructions:
        1. **Format:** Write the response strictly in **Markdown**.
        2. **Structure:** - Start with a high-level summary of the core difference.
           - **Create a Markdown Table** comparing the two concepts (Columns: Feature, Concept A, Concept B).
           - Follow with a "Key Distinctions" list.
        3. **Constraint:** Use ONLY the provided context.
        
        Context:
        {context}
        
        User Request: {question}
        
        Comparison:"""
        
        chain = (
            RunnablePassthrough.assign(context=lambda x: retriever.invoke(x["question"]))
            | ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"question": query})
    

# Registry (The Factory
class TaskRegistry:
    def __init__(self, db_instance):
        self.handlers = {
            "qa": QATaskHandler(db_instance),
            "explanation": QATaskHandler(db_instance), 
            "comparison": ComparisonTaskHandler(db_instance),  
            "summary": SummaryTaskHandler(db_instance),
            "default": ChatTaskHandler(db_instance),
            "follow_up": FollowupTaskHandler(db_instance)
        }

    def get_handler(self, task_name):
        return self.handlers.get(task_name, self.handlers["default"])