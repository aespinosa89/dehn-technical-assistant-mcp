import os
import torch
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
from .config import settings
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    UsageInfo,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingData
)

# Global model instances
_llm = None
_embedding_model = None
_lock = asyncio.Lock()

async def load_model():
    """Load the language and embedding models"""
    global _llm, _embedding_model
    
    async with _lock:
        if _llm is None:
            # Set up GPU-specific parameters
            tensor_parallel_size = settings.TENSOR_PARALLEL_SIZE
            gpu_memory_utilization = settings.VLLM_GPU_MEMORY_UTILIZATION
            
            # Set up dtype
            if settings.DTYPE == "float":
                dtype = torch.float32
            elif settings.DTYPE == "half":
                dtype = torch.float16
            elif settings.DTYPE == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float16  # Default
                
            # Load the LLM with vLLM
            quantization = settings.QUANTIZATION
            _llm = LLMModel(
                model_path=settings.MODEL_PATH,
                tokenizer_path=settings.TOKENIZER_PATH or settings.MODEL_PATH,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                quantization=quantization,
                model_type=settings.MODEL_TYPE
            )
            
        if _embedding_model is None:
            # Load the embedding model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)

async def get_model():
    """Get or load the model"""
    if _llm is None:
        await load_model()
    return _llm

class LLMModel:
    """Wrapper around vLLM model to handle chat completions API"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: torch.dtype = torch.float16,
        quantization: Optional[str] = None,
        model_type: str = "gemma"
    ):
        # Load vLLM model
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            quantization=quantization,
            trust_remote_code=True,
        )
        
        self.model_name = os.path.basename(model_path)
        self.max_tokens = settings.MAX_TOKENS
        self.model_type = model_type
        
    def _create_prompt(self, messages: List[ChatMessage]) -> str:
        """Create prompt from messages based on model type"""
        prompt = ""
        
        # Extract system message if it exists
        system_content = ""
        for message in messages:
            if message.role.lower() == "system":
                system_content = message.content
                break
                
        if self.model_type == "gemma":
            # Add system prompt if it exists
            if system_content:
                prompt += settings.SYSTEM_PROMPT_TEMPLATE.format(system_prompt=system_content)
                
            # Process conversation history
            for message in messages:
                if message.role.lower() == "system":
                    continue  # Already handled above
                elif message.role.lower() == "user":
                    prompt += settings.USER_PROMPT_TEMPLATE.format(prompt=message.content)
                elif message.role.lower() == "assistant":
                    prompt += settings.ASSISTANT_PROMPT_TEMPLATE.format(assistant_response=message.content)
                    
            # Add final assistant turn start
            prompt += "<start_of_turn>assistant\n"
        
        else:  # Default for other models
            if system_content:
                prompt += f"System: {system_content}\n\n"
                
            for message in messages:
                if message.role.lower() == "system":
                    continue  # Already handled above
                elif message.role.lower() == "user":
                    prompt += f"User: {message.content}\n\n"
                elif message.role.lower() == "assistant":
                    prompt += f"Assistant: {message.content}\n\n"
                    
            prompt += "Assistant: "
            
        return prompt
        
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a completion for the provided chat messages"""
        prompt = self._create_prompt(request.messages)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or settings.MAX_TOKENS,
            stop=request.stop or None,
        )
        
        # Generate using vLLM
        outputs = self.llm.generate(prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Compute token counts (approximate)
        prompt_tokens = len(prompt) // 4  # Rough estimate
        completion_tokens = len(generated_text) // 4  # Rough estimate
        
        # Create response object
        response = ChatCompletionResponse(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            object="chat.completion",
            created=int(asyncio.get_event_loop().time()),
            model=self.model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        return response
        
    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Stream a completion for the provided prompt"""
        prompt = self._create_prompt(request.messages)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or settings.MAX_TOKENS,
            stop=request.stop or None,
        )
        
        # Stream output using vLLM
        stream_id = f"chatcmpl-{os.urandom(4).hex()}"
        creation_time = int(asyncio.get_event_loop().time())
        accumulated_text = ""
        chunk_id = 0
        
        for output in self.llm.generate_stream(prompt, sampling_params):
            new_text = output.outputs[0].text
            delta_text = new_text[len(accumulated_text):]
            accumulated_text = new_text
            
            if delta_text:
                chunk_id += 1
                chunk = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": creation_time,
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": delta_text
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {str(chunk).replace('\"', '\"')}\n\n"
                
        # Send the final chunk with finish_reason
        final_chunk = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": creation_time,
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {str(final_chunk).replace('\"', '\"')}\n\n"
        yield "data: [DONE]\n\n"

    async def get_embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Generate embeddings for the provided input"""
        global _embedding_model
        
        if _embedding_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
        
        # Process input
        input_texts = request.input if isinstance(request.input, list) else [request.input]
        
        # Generate embeddings
        embeddings = _embedding_model.encode(input_texts, convert_to_numpy=True).tolist()
        
        # Create response
        data = []
        for i, embedding in enumerate(embeddings):
            data.append(
                EmbeddingData(
                    index=i,
                    object="embedding",
                    embedding=embedding
                )
            )
        
        return EmbeddingsResponse(
            object="list",
            data=data,
            model=settings.EMBEDDING_MODEL,
            usage={
                "prompt_tokens": sum(len(text.split()) for text in input_texts),
                "total_tokens": sum(len(text.split()) for text in input_texts)
            }
        )
