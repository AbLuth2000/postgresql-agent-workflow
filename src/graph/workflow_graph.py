import pprint
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, List, Dict

# Import agent logic from agents/
from src.agents.orchestrator_agent import route_request, OrchestratorResponse
from src.agents.postgresql_writer import generate_query
from src.agents.postgresql_checker import validate_query
from src.agents.executor_agent import execute_query
from src.agents.analyst_agent import analyze_request, AnalystDependencies

# Max retries to prevent looping
MAX_RETRIES = 3

# ───────────────────────────────────────────────────────────────
# Define the shared LangGraph state
# ───────────────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    user_input: str
    decision: Optional[str] = None
    follow_up_question: Optional[str] = None
    sql_query: Optional[str] = None
    query_results: Optional[List[Dict]] = None
    validated: Optional[bool] = None
    analysis: Optional[Dict] = None
    executor_response: Optional[Dict] = None
    retry_count: Optional[int]
    message_history: Optional[List[Dict[str, str]]]
    in_analyst_mode: Optional[bool]

# ───────────────────────────────────────────────────────────────
# Define LangGraph node functions
# ───────────────────────────────────────────────────────────────

def orchestrate(state: WorkflowState) -> WorkflowState:
    retry_count = state.get("retry_count", 0)
    history = state.get("message_history", [])
    in_analyst_mode = state.get("in_analyst_mode", False)

    if retry_count >= MAX_RETRIES:
        print(f"Max retries ({MAX_RETRIES}) reached. Ending workflow.")
        return {
            **state,
            "decision": "complete",
            "follow_up_question": "We weren't able to process your request after several attempts. Please try again with more clarity."
        }

    # Append user input to history
    history.append({
        "role": "user",
        "content": state["user_input"]
    })

    # If already in analyst mode, instruct the orchestrator accordingly
    if in_analyst_mode:
        orchestrator_input = f"(You are currently in analyst mode)\nUser: {state['user_input']}"
    else:
        orchestrator_input = state["user_input"]

    response: OrchestratorResponse = route_request(orchestrator_input, history)

    # Append assistant response to history
    history.append({
        "role": "assistant",
        "content": response.follow_up_question or f"Decision: {response.decision}"
    })

    # Handle analyst mode toggle based on decision
    new_in_analyst_mode = in_analyst_mode
    if response.decision == "analyst":
        new_in_analyst_mode = True
    elif response.decision != "analyst":
        new_in_analyst_mode = False

    return {
        **state,
        "decision": response.decision,
        "follow_up_question": response.follow_up_question,
        "message_history": history,
        "retry_count": retry_count + 1,
        "in_analyst_mode": new_in_analyst_mode
    }

def handle_writer(state: WorkflowState) -> WorkflowState:
    result = generate_query({
        "user_request": state["user_input"]
    })
    return {**state, "sql_query": result.sql_query}

def handle_checker(state: WorkflowState) -> WorkflowState:
    result = validate_query({
        "user_request": state["user_input"],
        "sql_query": state["sql_query"]
    })
    return {**state, "validated": result.is_valid}

def handle_executor(state: WorkflowState) -> WorkflowState:
    result = execute_query({"sql_query": state["sql_query"]})
    return {
        **state,
        "executor_response": result.dict(),
        "query_results": result.results
    }

def handle_analyst(state: WorkflowState) -> WorkflowState:
    result = analyze_request(AnalystDependencies(
        user_request=state["user_input"]
    ))
    return {**state, "analysis": result.dict()}

# ───────────────────────────────────────────────────────────────
# Logging for testing visibility
# ───────────────────────────────────────────────────────────────

pp = pprint.PrettyPrinter(indent=2)

def log_node(name: str, fn):
    def wrapped(state: WorkflowState) -> WorkflowState:
        print(f"\n--- ENTERING NODE: {name.upper()} ---")
        print("Input:")
        pp.pprint({k: v for k, v in state.items() if v is not None})

        result = fn(state)

        print(f"\n--- EXITING NODE: {name.upper()} ---")
        print("Output:")
        pp.pprint({k: v for k, v in result.items() if v is not None})
        return result
    return wrapped

# ───────────────────────────────────────────────────────────────
# Build the stateful LangGraph
# ───────────────────────────────────────────────────────────────

workflow = StateGraph(WorkflowState)

# Register core nodes with logging
workflow.add_node("orchestrator", log_node("orchestrator", orchestrate))
workflow.add_node("postgresql_writer", log_node("postgresql_writer", handle_writer))
workflow.add_node("postgresql_checker", log_node("postgresql_checker", handle_checker))
workflow.add_node("executor", log_node("executor", handle_executor))
workflow.add_node("analyst", log_node("analyst", handle_analyst))

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
