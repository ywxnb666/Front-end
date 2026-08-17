from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AgentMode = Literal["ask", "agent"]


class AgentChatRequest(BaseModel):
    mode: AgentMode = "ask"
    session_id: str = ""
    history_view: bool = False
    messages: list[dict[str, Any]] = Field(default_factory=list)


class ParameterPatch(BaseModel):
    key: str
    value: str | int | float | bool


class PipelineRequest(BaseModel):
    steps: list[str] = Field(default_factory=list)
    full: bool = False


class ActionProposal(BaseModel):
    action_id: str
    action_type: Literal["pipeline", "remote_config"]
    title: str
    detail: str
    payload: dict[str, Any] = Field(default_factory=dict)
    expires_at: float


class ToolEvent(BaseModel):
    name: str
    status: Literal["started", "completed", "failed", "pending"]
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class AgentChatResponse(BaseModel):
    ok: bool
    mode: AgentMode
    session_id: str = ""
    message: str = ""
    reasoning: str = ""
    tool_events: list[ToolEvent] = Field(default_factory=list)
    pending_action: ActionProposal | None = None
    config: dict[str, Any] | None = None
    pipeline: dict[str, Any] | None = None
    error: str = ""
