from typing import Any, List, Optional

from groq import Groq
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import Extra, Field, root_validator

GROQ_MODEL = "openai/gpt-oss-20b"


def _to_groq_message(message: BaseMessage) -> dict:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, ChatMessage):
        role = message.role
    else:
        role = "user"
    return {"role": role, "content": message.content or ""}


class ChatGroq(SimpleChatModel):
    """LangChain 0.0.181 chat model backed by the Groq Python SDK."""

    groq_api_key: str
    model: str = GROQ_MODEL
    temperature: float = 0.0
    client: Any = Field(default=None, exclude=True)

    class Config:
        extra = Extra.ignore
        arbitrary_types_allowed = True

    @root_validator()
    def _build_client(cls, values):
        key = values.get("groq_api_key")
        if not key:
            raise ValueError("groq_api_key is required")
        values["client"] = Groq(api_key=key)
        return values

    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def _identifying_params(self):
        return {"model": self.model, "temperature": self.temperature}

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": [_to_groq_message(m) for m in messages],
            "temperature": self.temperature,
        }
        if stop:
            kwargs["stop"] = stop
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""
