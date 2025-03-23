from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.graph.workflow_graph import app as workflow_app, WorkflowState

# FastAPI app instance
fastapi_app = FastAPI(title="LangGraph Orchestrator API")

# Request model
class UserRequest(BaseModel):
    user_input: str
    session_id: str
    state: Optional[WorkflowState] = None

@fastapi_app.post("/chat")
def chat(request: UserRequest):
    session_id = request.session_id

    # Use passed state if provided (e.g., during follow-up)
    state = request.state or {
        "user_input": request.user_input,
        "retry_count": 0,
        "message_history": [],
        "in_analyst_mode": False
    }

    # Inject the new user input
    state["user_input"] = request.user_input
    state["awaiting_follow_up"] = False  # Reset in case previous run had follow-up

    try:
        # Run LangGraph
        result = workflow_app.invoke(state)

        # Determine what type of response to return
        status = (
            "awaiting_follow_up" if result.get("decision") == "follow_up"
            else "complete" if result.get("decision") == "complete"
            else "in_progress"
        )

        return {
            "session_id": session_id,
            "status": status,
            "decision": result.get("decision"),
            "follow_up_question": result.get("follow_up_question"),
            "sql_query": result.get("sql_query"),
            "query_results": result.get("query_results"),
            "analysis": result.get("analysis"),
            "executor_response": result.get("executor_response"),
            "state": result  # 👈 Return full state so user can continue
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
