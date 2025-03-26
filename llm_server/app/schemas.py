from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal

# Chat Completion Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

# Embeddings Models
class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    user: Optional[str] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]

# RAG Models
class Document(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class RAGRequest(BaseModel):
    query: str
    documents: Optional[List[Document]] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    stream: bool = False
    system_prompt: Optional[str] = None

class RAGResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    usage: UsageInfo
