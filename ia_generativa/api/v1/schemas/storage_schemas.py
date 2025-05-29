"""
Schemas para validação de dados de persistência na API.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
from datetime import datetime

class ResultTypeEnum(str, Enum):
    """Tipos de resultados armazenados."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    WORKFLOW = "workflow"
    EMBEDDING = "embedding"
    OTHER = "other"
    
class StorageTypeEnum(str, Enum):
    """Tipos de armazenamento para diferentes dados."""
    DATABASE = "database"
    FILE = "file"
    S3 = "s3"
    COMPRESSED = "compressed"
    
class InteractionTypeEnum(str, Enum):
    """Tipos de interações de usuário."""
    VIEW = "view"
    LIKE = "like"
    DISLIKE = "dislike"
    SHARE = "share"
    GENERATE = "generate"
    FEEDBACK = "feedback"
    CHAT_MESSAGE = "chat_message"
    GENERATION_FEEDBACK = "generation_feedback"
    CONTENT_VIEW = "content_view"
    STYLE_SELECTION = "style_selection"
    CHAT_FEEDBACK = "chat_feedback"
    
class UserRequest(BaseModel):
    """Requisição para criar ou obter usuário."""
    username: str = Field(..., description="Nome de usuário")
    email: Optional[str] = Field(None, description="Email (opcional)")
    external_id: Optional[str] = Field(None, description="ID externo (opcional)")
    
    @validator('username')
    def username_not_empty(cls, v):
        """Valida que o nome de usuário não está vazio."""
        if not v.strip():
            raise ValueError("Nome de usuário não pode estar vazio")
        return v
    
class StoreResultRequest(BaseModel):
    """Requisição para armazenar um resultado."""
    user_id: int = Field(..., description="ID do usuário")
    result_type: ResultTypeEnum = Field(..., description="Tipo de resultado")
    content: Any = Field(..., description="Conteúdo a ser armazenado")
    storage_type: Optional[StorageTypeEnum] = Field(None, description="Tipo de armazenamento (opcional)")
    task_id: Optional[str] = Field(None, description="ID da tarefa associada (opcional)")
    workflow_id: Optional[str] = Field(None, description="ID do fluxo de trabalho associado (opcional)")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Metadados adicionais (opcional)")
    tags: Optional[List[str]] = Field([], description="Tags para categorização (opcional)")
    is_public: bool = Field(False, description="Se o resultado é público")
    ttl_days: Optional[int] = Field(None, description="Dias até expiração (opcional)")
    
class GetResultRequest(BaseModel):
    """Requisição para obter um resultado."""
    result_id: str = Field(..., description="ID do resultado")
    
class SearchResultsRequest(BaseModel):
    """Requisição para buscar resultados."""
    user_id: Optional[int] = Field(None, description="Filtrar por usuário (opcional)")
    result_type: Optional[ResultTypeEnum] = Field(None, description="Filtrar por tipo (opcional)")
    tags: Optional[List[str]] = Field(None, description="Filtrar por tags (opcional)")
    query: Optional[str] = Field(None, description="Busca textual (opcional)")
    include_public: bool = Field(False, description="Incluir resultados públicos")
    limit: int = Field(100, description="Limite de resultados")
    offset: int = Field(0, description="Deslocamento para paginação")
    
class DeleteResultRequest(BaseModel):
    """Requisição para excluir um resultado."""
    result_id: str = Field(..., description="ID do resultado")
    
class RecordInteractionRequest(BaseModel):
    """Requisição para registrar uma interação."""
    user_id: int = Field(..., description="ID do usuário")
    interaction_type: InteractionTypeEnum = Field(..., description="Tipo de interação")
    result_id: Optional[int] = Field(None, description="ID do resultado associado (opcional)")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Metadados adicionais (opcional)")
    
class GetUserInteractionsRequest(BaseModel):
    """Requisição para obter interações de um usuário."""
    user_id: int = Field(..., description="ID do usuário")
    interaction_type: Optional[InteractionTypeEnum] = Field(None, description="Filtrar por tipo (opcional)")
    limit: int = Field(100, description="Limite de resultados")
    offset: int = Field(0, description="Deslocamento para paginação")
    
class SetPreferenceRequest(BaseModel):
    """Requisição para definir uma preferência."""
    user_id: int = Field(..., description="ID do usuário")
    category: str = Field(..., description="Categoria da preferência")
    key: str = Field(..., description="Chave da preferência")
    value: Any = Field(..., description="Valor da preferência")
    
class GetPreferenceRequest(BaseModel):
    """Requisição para obter uma preferência."""
    user_id: int = Field(..., description="ID do usuário")
    category: str = Field(..., description="Categoria da preferência")
    key: str = Field(..., description="Chave da preferência")
    
class GetPreferencesByCategoryRequest(BaseModel):
    """Requisição para obter preferências por categoria."""
    user_id: int = Field(..., description="ID do usuário")
    category: str = Field(..., description="Categoria da preferência")
    
class DeletePreferenceRequest(BaseModel):
    """Requisição para excluir uma preferência."""
    user_id: int = Field(..., description="ID do usuário")
    category: str = Field(..., description="Categoria da preferência")
    key: str = Field(..., description="Chave da preferência")
    
class SetContextRequest(BaseModel):
    """Requisição para definir um contexto."""
    user_id: int = Field(..., description="ID do usuário")
    context_type: str = Field(..., description="Tipo de contexto")
    context_id: str = Field(..., description="ID do contexto")
    state: Dict[str, Any] = Field(..., description="Estado do contexto")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Metadados adicionais (opcional)")
    
class GetContextRequest(BaseModel):
    """Requisição para obter um contexto."""
    user_id: int = Field(..., description="ID do usuário")
    context_type: str = Field(..., description="Tipo de contexto")
    context_id: str = Field(..., description="ID do contexto")
    
class ListContextsRequest(BaseModel):
    """Requisição para listar contextos."""
    user_id: int = Field(..., description="ID do usuário")
    context_type: Optional[str] = Field(None, description="Filtrar por tipo (opcional)")
    limit: int = Field(10, description="Limite de resultados")
    
class BuildUserProfileRequest(BaseModel):
    """Requisição para construir perfil de usuário."""
    user_id: int = Field(..., description="ID do usuário")
    
class GetRelevantContextRequest(BaseModel):
    """Requisição para obter contexto relevante."""
    user_id: int = Field(..., description="ID do usuário")
    query: str = Field(..., description="Consulta ou prompt atual")
    max_results: int = Field(3, description="Número máximo de resultados")
    
class CreateSystemMessageRequest(BaseModel):
    """Requisição para criar mensagem de sistema com contexto."""
    user_id: int = Field(..., description="ID do usuário")
    include_preferences: bool = Field(True, description="Incluir preferências do usuário")
    include_history: bool = Field(True, description="Incluir resumo do histórico")
    include_profile: bool = Field(True, description="Incluir perfil do usuário")
    
class UserResponse(BaseModel):
    """Resposta com informações de usuário."""
    user_id: int = Field(..., description="ID do usuário")
    username: str = Field(..., description="Nome de usuário")
    email: Optional[str] = Field(None, description="Email")
    external_id: Optional[str] = Field(None, description="ID externo")
    created_at: str = Field(..., description="Data de criação")
    
class ResultMetadataResponse(BaseModel):
    """Resposta com metadados de um resultado."""
    result_id: str = Field(..., description="ID do resultado")
    user_id: int = Field(..., description="ID do usuário")
    result_type: str = Field(..., description="Tipo de resultado")
    storage_type: str = Field(..., description="Tipo de armazenamento")
    metadata: Dict[str, Any] = Field(..., description="Metadados")
    tags: List[str] = Field(..., description="Tags")
    is_public: bool = Field(..., description="Se é público")
    created_at: str = Field(..., description="Data de criação")
    updated_at: str = Field(..., description="Data de atualização")
    task_id: Optional[str] = Field(None, description="ID da tarefa")
    workflow_id: Optional[str] = Field(None, description="ID do fluxo de trabalho")
    size_bytes: Optional[int] = Field(None, description="Tamanho em bytes")
    preview: Optional[str] = Field(None, description="Preview do conteúdo")
    
class ResultResponse(BaseModel):
    """Resposta com resultado completo."""
    result_id: str = Field(..., description="ID do resultado")
    user_id: int = Field(..., description="ID do usuário")
    result_type: str = Field(..., description="Tipo de resultado")
    storage_type: str = Field(..., description="Tipo de armazenamento")
    content: Any = Field(..., description="Conteúdo")
    metadata: Dict[str, Any] = Field(..., description="Metadados")
    tags: List[str] = Field(..., description="Tags")
    is_public: bool = Field(..., description="Se é público")
    created_at: str = Field(..., description="Data de criação")
    updated_at: str = Field(..., description="Data de atualização")
    task_id: Optional[str] = Field(None, description="ID da tarefa")
    workflow_id: Optional[str] = Field(None, description="ID do fluxo de trabalho")
    
class SearchResultsResponse(BaseModel):
    """Resposta de busca de resultados."""
    results: List[ResultMetadataResponse] = Field(..., description="Lista de resultados")
    total: int = Field(..., description="Contagem total")
    limit: int = Field(..., description="Limite aplicado")
    offset: int = Field(..., description="Deslocamento aplicado")
    
class InteractionResponse(BaseModel):
    """Resposta com informações de interação."""
    id: int = Field(..., description="ID da interação")
    user_id: int = Field(..., description="ID do usuário")
    result_id: Optional[int] = Field(None, description="ID do resultado")
    interaction_type: str = Field(..., description="Tipo de interação")
    metadata: Dict[str, Any] = Field(..., description="Metadados")
    created_at: str = Field(..., description="Data de criação")
    
class PreferenceResponse(BaseModel):
    """Resposta com preferência."""
    user_id: int = Field(..., description="ID do usuário")
    category: str = Field(..., description="Categoria")
    key: str = Field(..., description="Chave")
    value: Any = Field(..., description="Valor")
    
class PreferencesCategoryResponse(BaseModel):
    """Resposta com preferências de uma categoria."""
    user_id: int = Field(..., description="ID do usuário")
    category: str = Field(..., description="Categoria")
    preferences: Dict[str, Any] = Field(..., description="Preferências")
    
class ContextResponse(BaseModel):
    """Resposta com contexto."""
    user_id: int = Field(..., description="ID do usuário")
    context_type: str = Field(..., description="Tipo de contexto")
    context_id: str = Field(..., description="ID do contexto")
    state: Dict[str, Any] = Field(..., description="Estado")
    metadata: Dict[str, Any] = Field(..., description="Metadados")
    version: int = Field(..., description="Versão")
    created_at: str = Field(..., description="Data de criação")
    updated_at: str = Field(..., description="Data de atualização")
    
class ContextListResponse(BaseModel):
    """Resposta com lista de contextos."""
    contexts: List[Dict[str, Any]] = Field(..., description="Lista de contextos")
    
class UserProfileResponse(BaseModel):
    """Resposta com perfil de usuário."""
    user_id: int = Field(..., description="ID do usuário")
    username: str = Field(..., description="Nome de usuário")
    preferences: Dict[str, Any] = Field(..., description="Preferências")
    recent_contexts: List[Dict[str, Any]] = Field(..., description="Contextos recentes")
    interaction_summary: Dict[str, Any] = Field(..., description="Resumo de interações")
    created_at: str = Field(..., description="Data de criação")
    profile_built_at: str = Field(..., description="Data de construção do perfil")
    
class SystemMessageResponse(BaseModel):
    """Resposta com mensagem de sistema."""
    role: str = Field(..., description="Papel da mensagem")
    content: str = Field(..., description="Conteúdo da mensagem")
