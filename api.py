from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import run_agent


app = FastAPI(title="Simple LangGraph Agent")


class AgentRequest(BaseModel):
    query: str = Field(..., examples=["Who is the CEO of OpenAI?"])


class AgentResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/test")
def test() -> dict[str, str]:
    return {"status": "test ok"}

@app.post("/agent", response_model=AgentResponse)
def run_agent_endpoint(request: AgentRequest) -> AgentResponse:
    try:
        answer = run_agent(request.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgentResponse(answer=answer)
