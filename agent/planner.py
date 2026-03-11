import os
import logging
from typing import List

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
except ImportError:
    pass

logger = logging.getLogger(__name__)

class TaskDecomposer:
    """
    Uses LangChain to conceptually break down complex user queries into 
    atomic sub-queries for broader retrieval coverage.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        try:
            if not self.openai_api_key:
                logger.warning("OPENAI_API_KEY missing. TaskDecomposer will fail.")
                self.llm = None
            else:
                self.llm = ChatOpenAI(temperature=0.0, model=model_name, api_key=self.openai_api_key) # type: ignore
        except Exception as e:
            logger.error(f"Failed to initialize LangChain ChatOpenAI: {e}")
            self.llm = None

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query analyzer for an enterprise search system.
Your task is to take a complex user query and decompose it into 2-3 atomic, standalone sub-queries.
If the query is already simple and atomic, just return the original query.
Output EXACTLY one sub-query per line. Do not number them or use bullet points. Do not add conversational text.

Example 1:
User: Compare the blackout periods for SVPs and regular employees
Output:
What is the blackout period for SVPs?
What is the blackout period for regular employees?

Example 2:
User: What is the PTO policy?
Output:
What is the PTO policy?"""),
            ("user", "{query}")
        ])
        
        if self.llm:
            self.chain = self.prompt | self.llm | StrOutputParser()

    def decompose(self, query: str) -> List[str]:
        """Decomposes a query into multiple sub-queries."""
        if not self.llm:
            return [query]
            
        try:
            logger.info(f"Decomposing query using LangChain: '{query}'")
            output = self.chain.invoke({"query": query})
            
            # Parse lines, stripping whitespace and removing empty lines
            sub_queries = [line.strip() for line in output.split("\n") if line.strip()] # type: ignore
            
            if not sub_queries:
                return [query]
                
            logger.info(f"Decomposed into {len(sub_queries)} queries: {sub_queries}")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]
