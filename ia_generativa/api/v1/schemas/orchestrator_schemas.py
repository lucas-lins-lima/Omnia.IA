"""
Schemas para validação de dados do orquestrador na API.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

class MediaTypeEnum(str, Enum):
    """Tipos de mídia suportados."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    SPREADSHEET = "spreadsheet"
    
class OperationStatusEnum(str, Enum):
    """Status possíveis para uma operação."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    
class WorkflowStatusEnum(str, Enum):
    """Status possíveis para um fluxo de trabalho."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    
class WorkflowOperation(BaseModel):
    """Definição de uma operação em um fluxo de trabalho."""
    media_type: MediaTypeEnum = Field(..., description="Tipo de mídia da operação")
    operation: str = Field(..., description="Nome da operação")
    input_from: Optional[str] = Field(None, description="ID da operação anterior para usar como entrada")
    parameters: Optional[Dict[str, Any]] = Field({}, description="Parâmetros para a operação")
    name: Optional[str] = Field(None, description="Nome da operação")
    description: Optional[str] = Field(None, description="Descrição da operação")
    
class CreateWorkflowRequest(BaseModel):
    """Requisição para criar um fluxo de trabalho."""
    name: str = Field(..., description="Nome do fluxo de trabalho")
    description: Optional[str] = Field(None, description="Descrição do fluxo")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Metadados adicionais")
    operations: Optional[List[WorkflowOperation]] = Field([], description="Lista de operações")
    
    @validator('name')
    def name_not_empty(cls, v):
        """Valida que o nome não está vazio."""
        if not v.strip():
            raise ValueError("Nome não pode estar vazio")
        return v
    
class AddOperationRequest(BaseModel):
    """Requisição para adicionar uma operação a um fluxo de trabalho."""
    workflow_id: str = Field(..., description="ID do fluxo de trabalho")
    media_type: MediaTypeEnum = Field(..., description="Tipo de mídia da operação")
    operation: str = Field(..., description="Nome da operação")
    input_from: Optional[str] = Field(None, description="ID da operação anterior para usar como entrada")
    parameters: Optional[Dict[str, Any]] = Field({}, description="Parâmetros para a operação")
    name: Optional[str] = Field(None, description="Nome da operação")
    description: Optional[str] = Field(None, description="Descrição da operação")
    
class ExecuteWorkflowRequest(BaseModel):
    """Requisição para executar um fluxo de trabalho."""
    workflow_id: str = Field(..., description="ID do fluxo de trabalho")
    initial_data: Optional[str] = Field(None, description="Dados iniciais para o fluxo (base64)")
    initial_media_type: Optional[MediaTypeEnum] = Field(None, description="Tipo de mídia dos dados iniciais")
    
    @validator('initial_media_type')
    def validate_media_type(cls, v, values):
        """Valida que o tipo de mídia é fornecido quando há dados iniciais."""
        if 'initial_data' in values and values['initial_data'] and v is None:
            raise ValueError("Tipo de mídia inicial deve ser especificado quando dados iniciais são fornecidos")
        return v
    
class CreateFromTemplateRequest(BaseModel):
    """Requisição para criar um fluxo de trabalho a partir de um template."""
    template_name: str = Field(..., description="Nome do template")
    name: Optional[str] = Field(None, description="Nome do fluxo de trabalho")
    description: Optional[str] = Field(None, description="Descrição do fluxo")
    parameters: Optional[Dict[str, Any]] = Field({}, description="Parâmetros para o template")
    
class WorkflowResponse(BaseModel):
    """Resposta contendo informações de um fluxo de trabalho."""
    workflow_id: str = Field(..., description="ID do fluxo de trabalho")
    name: str = Field(..., description="Nome do fluxo de trabalho")
    description: Optional[str] = Field(None, description="Descrição do fluxo")
    status: WorkflowStatusEnum = Field(..., description="Status do fluxo")
    created_at: float = Field(..., description="Timestamp de criação")
    updated_at: float = Field(..., description="Timestamp de última atualização")
    operations: List[Dict[str, Any]] = Field(..., description="Lista de operações")
    
class WorkflowStatusResponse(BaseModel):
    """Resposta contendo o status de um fluxo de trabalho."""
    workflow_id: str = Field(..., description="ID do fluxo de trabalho")
    name: str = Field(..., description="Nome do fluxo de trabalho")
    status: WorkflowStatusEnum = Field(..., description="Status do fluxo")
    created_at: float = Field(..., description="Timestamp de criação")
    updated_at: float = Field(..., description="Timestamp de última atualização")
    progress: float = Field(..., description="Progresso percentual")
    operations: Dict[str, int] = Field(..., description="Estatísticas de operações")
    error: Optional[str] = Field(None, description="Mensagem de erro, se houver")
    
class WorkflowResultsResponse(BaseModel):
    """Resposta contendo os resultados de um fluxo de trabalho."""
    workflow_id: str = Field(..., description="ID do fluxo de trabalho")
    status: WorkflowStatusEnum = Field(..., description="Status do fluxo")
    results: Dict[str, Any] = Field(..., description="Resultados por operação")
    operations: List[Dict[str, Any]] = Field(..., description="Detalhes das operações")
    error: Optional[str] = Field(None, description="Mensagem de erro, se houver")
    
class TemplateListResponse(BaseModel):
    """Resposta contendo a lista de templates disponíveis."""
    templates: List[str] = Field(..., description="Lista de templates disponíveis")
