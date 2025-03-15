# src/workflow.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any
from src.agents.orchestrator import OrchestratorAgent
from src.agents.analyst import AnalystAgent
from src.agents.postgresql_writer import PostgreSQLWriterAgent
from src.agents.postgresql_checker import PostgreSQLCheckerAgent
from src.agents.executor import ExecutorAgent

# Define State (Shared Information Passed Between Agents)
class AgentState:
    user_input: str
    next_agent: str
    response: Dict[str, Any]

# Initialize LangGraph Workflow
workflow = StateGraph(AgentState)

# Add Nodes (Agents)
workflow.add_node("orchestrator", OrchestratorAgent.run)
workflow.add_node("analyst", AnalystAgent.run)
workflow.add_node("postgresql_writer", PostgreSQLWriterAgent.run)
workflow.add_node("postgresql_checker", PostgreSQLCheckerAgent.run)
workflow.add_node("executor", ExecutorAgent.run)

# Define Routing Logic (Edges)
workflow.add_edge("orchestrator", "analyst", condition=lambda state: state.next_agent == "analyst")
workflow.add_edge("orchestrator", "postgresql_writer", condition=lambda state: state.next_agent == "postgresql_writer")
workflow.add_edge("orchestrator", "postgresql_checker", condition=lambda state: state.next_agent == "postgresql_checker")
workflow.add_edge("orchestrator", "executor", condition=lambda state: state.next_agent == "executor")
workflow.add_edge("orchestrator", END, condition=lambda state: state.next_agent == "complete")

# Handle Follow-Up Cases
def handle_follow_up(state: AgentState) -> AgentState:
    """If the orchestrator determines more info is needed, ask the user a follow-up question."""
    follow_up_question = state.response.get('follow_up_question', 'Could you please provide more details?')
    print(f"Follow-up needed: {follow_up_question}")
    user_reply = input("Your response: ")  # Capture user input dynamically
    state.user_input = user_reply
    return state

workflow.add_edge("orchestrator", handle_follow_up, condition=lambda state: state.next_agent == "follow_up")

# Set Entry Point
workflow.set_entry_point("orchestrator")

# Visualize the workflow using Mermaid syntax
mermaid_diagram = workflow.get_graph().draw_mermaid()
print(mermaid_diagram)