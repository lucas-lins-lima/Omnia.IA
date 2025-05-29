"""
Schemas para validação de dados de texto na API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from enum import Enum

class ModelType(str, Enum):
    """Tipos de modelos suportados."""
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    FALCON = "falcon"
    
class ChatMessage(BaseModel):
    """Mensagem individual em uma conversa."""
    role: str = Field(..., description="Papel do remetente (usuário ou assistente)")
    content: str = Field(..., description="Conteúdo da mensagem")
    
    @validator('role')
    def validate_role(cls, v):
        """Valida que o papel é 'user' ou 'assistant'."""
        if v.lower() not in ['user', 'assistant', 'system']:
            raise ValueError("Role deve ser 'user', 'assistant' ou 'system'")
        return v.lower()
    
class TextGenerationRequest(BaseModel):
    """Requisição para geração de texto."""
    prompt: str = Field(..., description="Prompt para geração de texto")
    model_type: ModelType = Field(ModelType.LLAMA3, description="Tipo de modelo a ser usado")
    max_new_tokens: int = Field(512, description="Número máximo de tokens a serem gerados")
    temperature: float = Field(0.7, description="Temperatura para geração de texto (0.0 a 1.0)")
    top_p: float = Field(0.9, description="Parâmetro de amostragem nucleus (0.0 a 1.0)")
    top_k: int = Field(50, description="Número de tokens mais prováveis a considerar")
    repetition_penalty: float = Field(1.1, description="Penalidade para repetição de tokens")
    stream: bool = Field(False, description="Se True, retorna respostas incrementais")
    system_prompt: Optional[str] = Field(None, description="Instruções de sistema personalizadas")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """Valida que a temperatura está entre 0.0 e 1.0."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Temperature deve estar entre 0.0 e 1.0")
        return v
    
    @validator('top_p')
    def validate_top_p(cls, v):
        """Valida que top_p está entre 0.0 e 1.0."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Top_p deve estar entre 0.0 e 1.0")
        return v
    
    @validator('repetition_penalty')
    def validate_repetition_penalty(cls, v):
        """Valida que repetition_penalty é maior que 1.0."""
        if v < 1.0:
            raise ValueError("Repetition_penalty deve ser maior ou igual a 1.0")
        return v
    
class ChatCompletionRequest(BaseModel):
    """Requisição para completar uma conversa."""
    messages: List[ChatMessage] = Field(..., description="Histórico de mensagens da conversa")
    model_type: ModelType = Field(ModelType.LLAMA3, description="Tipo de modelo a ser usado")
    max_new_tokens: int = Field(512, description="Número máximo de tokens a serem gerados")
    temperature: float = Field(0.7, description="Temperatura para geração de texto (0.0 a 1.0)")
    top_p: float = Field(0.9, description="Parâmetro de amostragem nucleus (0.0 a 1.0)")
    top_k: int = Field(50, description="Número de tokens mais prováveis a considerar")
    repetition_penalty: float = Field(1.1, description="Penalidade para repetição de tokens")
    stream: bool = Field(False, description="Se True, retorna respostas incrementais")
    system_prompt: Optional[str] = Field(None, description="Instruções de sistema personalizadas")
    
    @validator('messages')
    def validate_messages(cls, v):
        """Valida que a lista de mensagens não está vazia e contém pelo menos uma mensagem do usuário."""
        if not v:
            raise ValueError("A lista de mensagens não pode estar vazia")
        
        has_user_message = False
        for msg in v:
            if msg.role == "user":
                has_user_message = True
                break
                
        if not has_user_message:
            raise ValueError("Deve haver pelo menos uma mensagem do usuário")
            
        return v
    
class EmbeddingRequest(BaseModel):
    """Requisição para geração de embeddings."""
    texts: Union[str, List[str]] = Field(..., description="Texto ou lista de textos para gerar embeddings")
    model_name: Optional[str] = Field(None, description="Nome do modelo de embedding a ser usado")
    normalize: bool = Field(True, description="Se True, normaliza os embeddings para norma unitária")
    
    @validator('texts')
    def validate_texts(cls, v):
        """Valida que a lista de textos não está vazia."""
        if isinstance(v, list) and not v:
            raise ValueError("A lista de textos não pode estar vazia")
        if isinstance(v, str) and not v.strip():
            raise ValueError("O texto não pode estar vazio")
        return v
    
class TextResponse(BaseModel):
    """Resposta de geração de texto."""
    text: str = Field(..., description="Texto gerado")
    
class EmbeddingResponse(BaseModel):
    """Resposta de geração de embeddings."""
    embeddings: List[List[float]] = Field(..., description="Lista de vetores de embedding")
    dimensions: int = Field(..., description="Número de dimensões dos embeddings")
    model_name: str = Field(..., description="Nome do modelo de embedding usado")
    
class ModelInfoResponse(BaseModel):
    """Informações sobre um modelo carregado."""
    status: str = Field(..., description="Status do modelo (carregado ou não)")
    model_name: str = Field(..., description="Nome do modelo")
    device: Optional[str] = Field(None, description="Dispositivo onde o modelo está carregado")
    quantized: Optional[bool] = Field(None, description="Se o modelo está quantizado")
    quantization: Optional[str] = Field(None, description="Tipo de quantização usada")
    memory_used_gb: Optional[float] = Field(None, description="Memória usada pelo modelo (GB)")
    parameters: Optional[float] = Field(None, description="Número de parâmetros do modelo (bilhões)")
