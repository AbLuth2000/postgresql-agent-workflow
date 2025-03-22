from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph.workflow_graph import app

# FastAPI app instance
fastapi_app = FastAPI(title="LangGraph Orchestrator API")

# Request model
class UserRequest(BaseModel):
    user_input: str

@fastapi_app.post("/chat")
def chat(request: UserRequest):
    try:
        # Call LangGraph with just the user input
        state = app.invoke({
            "user_input": request.user_input
        })

        # Return the orchestrator decision + data (if any)
        return {
            "decision": state.get("decision"),
            "follow_up_question": state.get("follow_up_question"),
            "sql_query": state.get("sql_query"),
            "query_results": state.get("query_results"),
            "analysis": state.get("analysis"),
            "executor_response": state.get("executor_response")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
