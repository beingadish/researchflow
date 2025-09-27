"""
Tool Execution Module for AI Agent Search Operations

This module handles the execution of search-based tool calls from AI agents.
It processes AnswerQuestion and ReviseAnswer tool calls, extracts search queries,
executes them using the Tavily search API, and returns formatted results.

Key Features:
- Integration with Tavily search API for web search capabilities
- Processing of multiple search queries from AI tool calls
- JSON-formatted result aggregation and return
- Support for AnswerQuestion and ReviseAnswer tool types

Dependencies:
- langchain_core: For message handling and AI integration
- langchain_community: For Tavily search tool access
- json: For result serialization

Author: Aadarsh Pandey
Date: 28 Sep 2025
"""

import json
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_community.tools import TavilySearchResults

# Initialize the Tavily search tool with result limit configuration
# TavilySearchResults provides web search capabilities through Tavily API
tavily_tool = TavilySearchResults(max_results=5)


def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    """
    Execute search queries from AI agent tool calls and return formatted results.
    
    This function processes the conversation state to find AI messages containing
    tool calls, extracts search queries from AnswerQuestion or ReviseAnswer tools,
    executes web searches using Tavily, and returns the results as ToolMessages.
    
    Args:
        state (List[BaseMessage]): Conversation history containing AI messages with tool calls
        
    Returns:
        List[BaseMessage]: List of ToolMessage objects containing search results,
                          empty list if no tool calls found
        
    Processing Flow:
        1. Extract the most recent AI message from conversation state
        2. Validate presence of tool calls in the AI message
        3. Process each AnswerQuestion/ReviseAnswer tool call
        4. Execute search queries using Tavily tool
        5. Aggregate results and create ToolMessage responses
        
    Supported Tool Types:
        - AnswerQuestion: Initial query processing with search
        - ReviseAnswer: Follow-up query processing for refinement
    """
    # Get the most recent AI message from the conversation state
    last_ai_message: AIMessage = state[-1]
    
    # Validate that the AI message contains tool calls
    # Early return if no tool calls are present
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []
    
    # Initialize container for processed tool messages
    tool_messages = []
    
    # Process each tool call in the AI message
    for tool_call in last_ai_message.tool_calls:
        # Filter for supported tool types (AnswerQuestion and ReviseAnswer)
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            # Extract tool call metadata
            call_id = tool_call["id"]  # Unique identifier for this tool call
            search_queries = tool_call["args"].get("search_queries", [])  # List of search queries
            
            # Execute web searches for each query using Tavily
            query_results = {}  # Dictionary to store query -> results mapping
            
            for query in search_queries:
                # Invoke Tavily search tool for current query
                result = tavily_tool.invoke(query)
                # Store results with query as key for easy reference
                query_results[query] = result
            
            # Create a ToolMessage containing all search results for this tool call
            # JSON serialization ensures consistent data format
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_results),  # Serialized search results
                    tool_call_id=call_id  # Links response to original tool call
                )
            )
    
    return tool_messages


# Example Usage and Testing
# =========================
# Commented example demonstrating typical usage pattern
# 
# Example conversation state with HumanMessage and AIMessage containing tool calls
# test_state = [
#     HumanMessage(
#         content="Write about how small business can leverage AI to grow"
#     ),
#     AIMessage(
#         content="", 
#         tool_calls=[
#             {
#                 "name": "AnswerQuestion",  # Tool type for initial response
#                 "args": {
#                     'answer': '',  # Will be populated by the AI agent
#                     'search_queries': [  # Queries to execute for research
#                             'AI tools for small business', 
#                             'AI in small business marketing', 
#                             'AI automation for small business'
#                     ], 
#                     'reflection': {  # Metadata for response quality assessment
#                         'missing': '', 
#                         'superfluous': ''
#                     }
#                 },
#                 "id": "call_KpYHichFFEmLitHFvFhKy1Ra",  # Unique tool call ID
#             }
#         ],
#     )
# ]

# Execute the tool processing function
# results = execute_tools(test_state)

# Display raw and parsed results for debugging
# print("Raw results:", results)
# if results:
#     parsed_content = json.loads(results[0].content)
#     print("Parsed content:", parsed_content)