"""
Multi-Agent LangGraph Application

This module implements a multi-agent system using LangGraph that processes user queries
through a three-stage pipeline: initial response, tool execution, and revision.
The system includes tool call limiting and conditional routing between agents.

Components:
- Responder: Generates initial response to user queries
- Tools: Executes tool calls based on responder output
- Revisor: Reviews and refines the response, with conditional routing

Author: Aadarsh Pandey
Date: 28 Sep 2025
"""

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage
from chains import first_responder_chain
from langgraph.graph import END, MessageGraph
from chains import first_responder_chain, revisor_chain
from execute_tools import execute_tools
from typing import List

# Load environment variables from .env file
load_dotenv()

# Initialize the message graph for multi-agent workflow
graph = MessageGraph()

# Node identifiers - used for routing and graph construction
RESPONDER = "responder"  # Initial response generation agent
REVISOR = "revisor"      # Response revision and refinement agent
TOOLS = "tools"          # Tool execution node

# Configuration: Maximum allowed tool calls to prevent infinite loops
MAX_TOOL_CALLS = 3


def event_loop(state: List[BaseMessage]) -> str:
    """
    Conditional routing function for the graph workflow.
    
    Determines the next node based on the current state, specifically
    counting tool message instances to prevent excessive tool usage.
    
    Args:
        state (List[BaseMessage]): Current conversation state containing all messages
        
    Returns:
        str: Next node to route to (either TOOLS for continued processing or END to terminate)
        
    Logic:
        - Counts ToolMessage instances in the current state
        - If tool calls exceed MAX_TOOL_CALLS, terminates the workflow
        - Otherwise, continues to TOOLS node for further processing
    """
    # Count how many tool messages have been processed so far
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    
    # Prevent infinite loops by limiting tool calls
    if count_tool_visits > MAX_TOOL_CALLS:
        return END
    return TOOLS


# Graph Construction
# ==================

# Adding nodes to the graph
# Each node represents a processing stage in the workflow
graph.add_node(RESPONDER, first_responder_chain)  # Initial query processing
graph.add_node(REVISOR, revisor_chain)            # Response revision
graph.add_node(TOOLS, execute_tools)              # Tool execution

# Adding fixed edges - these define the primary workflow path
graph.add_edge(RESPONDER, TOOLS)  # Responder -> Tools: Always execute tools after initial response
graph.add_edge(TOOLS, REVISOR)    # Tools -> Revisor: Always review after tool execution

# Adding conditional edges - these provide dynamic routing capability
# Revisor can route to multiple destinations based on event_loop logic
graph.add_conditional_edges(REVISOR, event_loop)

# Set the entry point - defines where the workflow begins
graph.set_entry_point(RESPONDER)

# Compile the graph into an executable application
app = graph.compile()

# Development/Debug utilities
# ===========================
# Generate visual representations of the graph structure
print(app.get_graph().draw_mermaid())  # Mermaid diagram for visualization
app.get_graph().print_ascii()          # ASCII art representation

# User Interaction
# ================
# Runtime query input from user
query = input("Query: ")

# Execute the compiled graph with user input
# The graph will process through responder -> tools -> revisor -> (conditional routing)
response = app.invoke(query)

# Extract the final answer from the response
# Assumes the last message contains tool calls with an answer argument
answer = response[-1].tool_calls[0]["args"]["answer"]

# Display formatted output
print("==" * 50)  # Visual separator
print(answer)

# =========================================== SAMPLE RESPONSE BELOW =========================================

# config:
#   flowchart:
#     curve: linear
# ---
# graph TD;
#         __start__([<p>__start__</p>]):::first
#         responder(responder)
#         revisor(revisor)
#         tools(tools)
#         __end__([<p>__end__</p>]):::last
#         __start__ --> responder;
#         responder --> tools;
#         tools --> revisor;
#         revisor -.-> responder;
#         revisor -.-> tools;
#         revisor -.-> __end__;
#         classDef default fill:#f2f0ff,line-height:1.2
#         classDef first fill-opacity:0
#         classDef last fill:#bfb6fc

#         +-----------+    
#         | __start__ |
#         +-----------+
#               *
#               *
#               *
#         +-----------+
#         | responder |
#         +-----------+
#          *         ..
#        **            .
#       *               ..
# +-------+               .
# | tools |             ..
# +-------+            .
#          *         ..
#           **     ..
#             *   .
#         +---------+
#         | revisor |
#         +---------+
#               .
#               .
#               .
#         +---------+
#         | __end__ |
#         +---------+
# Query: Write about the growth of Indian Footbal Team.
# ====================================================================================================
# The Indian football team's "golden era" (1950s-60s) included an Olympic semi-final in 1956 and two Asian Games gold medals in 1951 and 1962 [1]. After a prolonged decline, a resurgence began in the early 21st century, largely led by captain Sunil Chhetri.

# India's FIFA ranking improved, breaking into the top 100 in 2017, reaching 96th [2]. The team has consistently won the SAFF Championship and qualified for the AFC Asian Cup in 2011, 2019, and 2023 [3]. Notable recent successes also include winning the Intercontinental Cup in 2018 and 2023 [5].  

# The Indian Super League (ISL), launched in 2014, has been pivotal in professionalizing the sport, attracting foreign talent, and providing a competitive platform [4]. The All India Football Federation (AIFF) administers the ISL and, through its 'Vision 2047' strategic roadmap, aims for India to be a top 4 Asian footballing nation by 2047 [8]. This plan includes a grassroots project to reach 3.5 crore children in 100 villages and 10 tribal districts, with FIFA's Arsene Wenger assisting [10, 11]. Women's football has also seen significant growth, with a 138% surge in player registration in two years [6].

# Challenges persist, including ensuring consistent international exposure for players and addressing the lack of robust infrastructure, quality coaching, and structured youth development programs at the grassroots level [7, 9]. Beyond the ISL, the I-League operates as the second tier of the domestic league system [12]. Fan engagement efforts extend to rural outreach through mobile coaching units and community events [13].

# References:
# [1] https://www.olympics.com/en/news/history-of-indian-football
# [2] https://www.olympics.com/en/news/india-football-team-rankings-world-fifa-best-worst-position-points-table
# [3] https://en.wikipedia.org/wiki/India_national_football_team_records_and_statistics
# [4] https://turftown.in/blog/football-craze-in-india
# [5] https://en.wikipedia.org/wiki/India_national_football_team
# [6] https://www.the-aiff.com/index.php/article/aiff-records-historic-rise-in-womens-footballers
# [7] https://www.aljazeera.com/sports/2025/7/24/indian-football-hurt-scared-as-domestic-game-hits-fresh-low
# [8] https://www.sportbusiness.com/news/aiff-sets-out-vision-2047-long-term-plan-for-indian-football/
# [9] https://www.linkedin.com/pulse/unpacking-enigma-challenges-affecting-performance-team-chakraborty-pjc5c
# [10] https://www.espn.com/soccer/story/_/id/37635352/aiff-roadmap-decoding-vision-decide-fate-indian-football-future
# [11] https://www.olympics.com/en/news/arsene-wenger-fifa-task-force-indian-football
# [12] https://en.wikipedia.org/wiki/Indian_football_league_system
# [13] https://www.linkedin.com/pulse/united-game-growing-football-culture-india-jatin-tyagi--i0urc