"""
Pydantic Schema Definitions for AI Research Agent System

This module defines the structured data models used by the AI research agent system
for processing questions, generating answers, and managing the revision workflow.
The schemas ensure consistent output formatting and enable structured interaction
between different components of the research pipeline.

Schema Hierarchy:
- Reflection: Base model for answer critique and improvement suggestions
- AnswerQuestion: Primary model for initial question answering with research queries
- ReviseAnswer: Extended model for answer revision with citation support

Key Features:
- Field validation and type safety through Pydantic
- Structured critique system for iterative improvement
- Search query generation for research enhancement
- Citation management for answer verification
- Inheritance-based design for workflow progression

Usage Context:
These schemas are used by LangChain tool parsers to structure AI model outputs,
ensuring consistent data formats across the first responder and revisor chains.

Author: Aadarsh Pandey
Date: 28 Sep 2025
"""

from pydantic import BaseModel, Field
from typing import List


class Reflection(BaseModel):
    """
    Model for structured self-critique and answer improvement analysis.
    
    This class provides a framework for AI agents to critically evaluate their
    initial responses by identifying missing information and unnecessary content.
    It supports the iterative improvement process by highlighting specific areas
    that need enhancement or reduction.
    
    Use Cases:
    - Self-assessment of answer completeness and accuracy
    - Identification of content gaps requiring additional research
    - Detection of irrelevant or excessive information
    - Structured feedback for answer refinement
    
    Attributes:
        missing (str): Critical analysis of information gaps, incomplete coverage,
                      or important points that should be included in the answer
        superfluous (str): Analysis of excessive, irrelevant, or redundant content
                          that should be removed or condensed for clarity
    """
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """
    Primary schema for initial question answering with research capabilities.
    
    This model structures the complete response workflow for the first responder
    agent, combining answer generation, self-reflection, and research query
    planning in a single structured output. It enables the system to provide
    immediate answers while identifying areas for improvement through search.
    
    Workflow Integration:
    1. Generate initial answer (250 words target)
    2. Perform self-reflection on answer quality
    3. Generate search queries for improvement research
    4. Pass to tool execution for query processing
    
    Attributes:
        answer (str): Detailed response to the user's question, targeting approximately
                     250 words with comprehensive coverage of key points
        search_queries (List[str]): 1-3 focused search queries designed to address
                                   identified gaps and improve answer quality
        reflection (Reflection): Structured self-critique identifying missing
                               information and superfluous content
    """
    answer: str = Field(description="~250 words detailed answer to the question.")
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements to address the critique of the current answer")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")


class ReviseAnswer(AnswerQuestion):
    """
    Extended schema for answer revision with citation and reference management.
    
    This model inherits all capabilities from AnswerQuestion while adding
    citation support for the revision workflow. It enables the revisor agent
    to refine answers using search results while maintaining verifiable
    references and adhering to quality standards.
    
    Inheritance Benefits:
    - Maintains all original fields (answer, search_queries, reflection)
    - Adds citation capabilities for enhanced credibility
    - Supports iterative improvement workflow
    - Enables tool choice differentiation in LangChain
    
    Revision Requirements:
    - Integrate search results from previous queries
    - Add numerical citations for verification
    - Include references section with URLs
    - Maintain 250-word limit excluding references
    - Apply reflection critique for quality improvement
    
    Attributes:
        references (List[str]): Structured list of citations supporting the revised
                               answer, typically formatted as numbered references
                               with URLs for verification and credibility
        
    Inherited Attributes:
        answer (str): Refined version of the original answer incorporating
                     search results and addressing identified gaps
        search_queries (List[str]): Additional queries if further research needed
        reflection (Reflection): Updated critique of the revised answer
    """
    references: List[str] = Field(description="Citations motivating your updated answer.")
