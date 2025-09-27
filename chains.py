"""
AI Research Agent Chains Module

This module implements a two-stage AI research system using LangChain and Google's Gemini model.
The system consists of a first responder that generates initial answers with search queries,
and a revisor that refines the answer using search results and adds proper citations.

Key Components:
- First Responder: Generates initial detailed answers (~250 words) with reflection and search queries
- Revisor: Refines answers using search results, adds citations, and maintains word limit
- Pydantic Tools: Structured output parsing for AnswerQuestion and ReviseAnswer schemas
- Dynamic Prompts: Time-aware prompts with reflection and critique capabilities

Workflow:
1. First responder generates initial answer with self-critique and search queries
2. Search queries are executed (handled by external tool execution module)
3. Revisor refines the answer using search results and adds proper citations

Dependencies:
- langchain: Core framework for prompt templates and message handling
- langchain_google_genai: Google Generative AI integration (Gemini model)
- langchain_core: Output parsers and core utilities
- schema: Custom Pydantic schemas (AnswerQuestion, ReviseAnswer)
- dotenv: Environment variable management

Author: Aadarsh Pandey
Date: 28 Sep 2025
"""

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers import PydanticToolsParser
from dotenv import load_dotenv

# Load environment variables from .env file (API keys, configuration, etc.)
load_dotenv()

# Initialize the Google Generative AI model
# Using Gemini 2.5 Flash for fast, high-quality responses
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Core Actor Prompt Template
# ==========================
# This template serves as the foundation for both responder and revisor chains
# It includes dynamic time injection and structured instruction formatting
actor_prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """You are expert AI Researcher. 
            Current Time: {time}
            
            1. {first_instructions}
            2. Reflect and critique your answer. Be severe to maximize improvements.
            3. After the reflection, **list 1-3 search queries seperately** for researching improvements.
            Do not include them inside the reflection.   
            """
        ),
        MessagesPlaceholder(variable_name="messages"),  # Placeholder for conversation history
        ("system", "Answer the user's question above using the required format")
    ],
).partial(
    time=lambda: datetime.datetime.now().isoformat()  # Dynamic time injection for context
)


# Revisor-Specific Instructions
# ============================
# Detailed instructions for the revision stage focusing on:
# - Integration of search results
# - Citation requirements
# - Word limit enforcement
# - Quality improvements based on critique
revisor_prompt = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not count towards the word limit). 
    In the form of:
        - [1] [https://example.com](https://example.com)
        - [2] [https://example2.com](https://example2.com)
    - You should use the previous crituque to remove the superfluous 
    information from your answer and make SURE it is not more than 250 words.
"""


# Chain Component Configuration
# =============================

# First Responder Prompt Configuration
# Specialized version of actor_prompt_template for initial response generation
# Focuses on providing detailed ~250 word answers with self-reflection
first_responder_prompt_template = actor_prompt_template.partial(
    first_instructions="Provide me a detailed ~250 words answer"
)

# Pydantic Tool Parsers
# These parsers ensure structured output from the LLM using predefined schemas
responder_parser = PydanticToolsParser(tools=[AnswerQuestion])  # Parses initial responses
revisor_parser = PydanticToolsParser(tools=[ReviseAnswer])      # Parses revision responses


# Chain Definitions
# =================

# First Responder Chain
# Handles initial query processing with structured output
# - Uses first_responder_prompt_template for consistent formatting
# - Binds AnswerQuestion tool with forced tool choice
# - Generates initial answer with reflection and search queries
first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    [AnswerQuestion], 
    tool_choice="AnswerQuestion"  # Forces the model to use AnswerQuestion schema
)

# Revisor Chain
# Handles answer refinement using search results and critique
# - Uses actor_prompt_template with revisor-specific instructions
# - Binds ReviseAnswer tool with forced tool choice
# - Integrates search results and adds proper citations
# - Enforces word limit and quality improvements
revisor_chain = actor_prompt_template.partial(
    first_instructions=revisor_prompt  # Inject revisor-specific instructions
) | llm.bind_tools(
    [ReviseAnswer], 
    tool_choice="ReviseAnswer"  # Forces the model to use ReviseAnswer schema
)