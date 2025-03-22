from langgraph.graph import StateGraph, START, END
from typing import Union

# Import agent logic from agents/
from src.agents.orchestrator_agent import route_request, OrchestratorResponse
from src.agents.postgresql_writer import generate_query
from src.agents.postgresql_checker import validate_query
from src.agents.executor_agent import execute_query
from src.agents.analyst_agent import analyze_request

# ───────────────────────────────────────────────────────────────
# Define the shared LangGraph state
# ───────────────────────────────────────────────────────────────

class WorkflowState:
    user_input: str
    decision: str
    follow_up_question: Union[str, None] = None
    sql_query: Union[str, None] = None
    query_results: Union[list, None] = None
    validated: Union[bool, None] = None
    analysis: Union[dict, None] = None
    executor_response: Union[dict, None] = None

# ───────────────────────────────────────────────────────────────
# Define LangGraph node functions
# ───────────────────────────────────────────────────────────────

def orchestrate(state: WorkflowState) -> WorkflowState:
    response: OrchestratorResponse = route_request(state.user_input)
    return {
        **state,
        "decision": response.decision,
        "follow_up_question": response.follow_up_question
    }

def handle_writer(state: WorkflowState) -> WorkflowState:
    result = generate_query({
        "user_request": state.user_input
    })
    return {**state, "sql_query": result.sql_query}

def handle_checker(state: WorkflowState) -> WorkflowState:
    result = validate_query({
        "user_request": state.user_input,
        "sql_query": state.sql_query
    })
    return {**state, "validated": result.is_valid}

def handle_executor(state: WorkflowState) -> WorkflowState:
    result = execute_query({"sql_query": state.sql_query})
    return {
        **state,
        "executor_response": result.dict(),
        "query_results": result.results
    }

def handle_analyst(state: WorkflowState) -> WorkflowState:
    result = analyze_request({
        "user_request": state.user_input,
        "query_results": state.query_results
    })
    return {**state, "analysis": result.dict()}

# ───────────────────────────────────────────────────────────────
# Build the stateful LangGraph
# ───────────────────────────────────────────────────────────────

workflow = StateGraph(WorkflowState)

# Register core nodes
workflow.add_node("orchestrator", orchestrate)
workflow.add_node("postgresql_writer", handle_writer)
workflow.add_node("postgresql_checker", handle_checker)
workflow.add_node("executor", handle_executor)
workflow.add_node("analyst", handle_analyst)

# Define the entry point of the workflow
workflow.add_edge(START, "orchestrator")

# Add routing logic based on orchestrator decision
workflow.add_conditional_edges(
    "orchestrator",
    lambda state: state["decision"],
    {
        "postgresql_writer": "postgresql_writer",
        "postgresql_checker": "postgresql_checker",
        "executor": "executor",
        "analyst": "analyst",
        "follow_up": "orchestrator",
        "complete": END  # End the workflow
    }
)

# Add edges returning to orchestrator after each subtask
workflow.add_edge("postgresql_writer", "orchestrator")
workflow.add_edge("postgresql_checker", "orchestrator")
workflow.add_edge("executor", "orchestrator")
workflow.add_edge("analyst", "orchestrator")

# ───────────────────────────────────────────────────────────────
# Compile the workflow app
# ───────────────────────────────────────────────────────────────

app = workflow.compile()
