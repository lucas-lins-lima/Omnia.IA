"""
Schemas para validação de dados de tarefas na API.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

class TaskStatusEnum(str, Enum):
    """Status possíveis para uma tarefa."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    REVOKED = "REVOKED"
    RETRY = "RETRY"
    
class TaskType(str, Enum):
    """Tipos de tarefas disponíveis."""
    TEXT_GENERATE = "text.generate"
    TEXT_CHAT = "text.chat"
    TEXT_EMBEDDINGS = "text.embeddings"
    TEXT_SUMMARIZE = "text.summarize"
    
    IMAGE_GENERATE = "image.generate"
    IMAGE_TRANSFORM = "image.transform"
    IMAGE_CAPTION = "image.caption"
    IMAGE_CLASSIFY = "image.classify"
    IMAGE_BATCH = "image.batch_process"
    
    AUDIO_TRANSCRIBE = "audio.transcribe"
    AUDIO_SYNTHESIZE = "audio.synthesize"
    AUDIO_GENERATE_MUSIC = "audio.generate_music"
    AUDIO_PROCESS = "audio.process"
    AUDIO_EXTRACT_FEATURES = "audio.extract_features"
    
    VIDEO_PROCESS = "video.process"
    VIDEO_ANALYZE = "video.analyze"
    VIDEO_SLIDESHOW = "video.create_slideshow"
    VIDEO_EXTRACT_AUDIO = "video.extract_audio"
    
    WORKFLOW_EXECUTE = "workflow.execute"
    WORKFLOW_CREATE_TEMPLATE = "workflow.create_from_template"
    WORKFLOW_BATCH = "workflow.batch_process"
    
class CreateTaskRequest(BaseModel):
    """Requisição para criar uma nova tarefa."""
    task_type: TaskType = Field(..., description="Tipo de tarefa")
    args: List[Any] = Field([], description="Argumentos posicionais para a tarefa")
    kwargs: Dict[str, Any] = Field({}, description="Argumentos nomeados para a tarefa")
    
class TaskStatusRequest(BaseModel):
    """Requisição para verificar o status de uma tarefa."""
    task_id: str = Field(..., description="ID da tarefa")
    
class TaskCancelRequest(BaseModel):
    """Requisição para cancelar uma tarefa."""
    task_id: str = Field(..., description="ID da tarefa")
    
class TaskResultRequest(BaseModel):
    """Requisição para obter o resultado de uma tarefa."""
    task_id: str = Field(..., description="ID da tarefa")
    include_result: bool = Field(True, description="Se True, inclui o resultado completo")
    
class TaskResponse(BaseModel):
    """Resposta de criação de tarefa."""
    task_id: str = Field(..., description="ID da tarefa")
    task_type: TaskType = Field(..., description="Tipo de tarefa")
    status: TaskStatusEnum = Field(TaskStatusEnum.PENDING, description="Status inicial da tarefa")
    
class TaskStatusResponse(BaseModel):
    """Resposta com o status de uma tarefa."""
    task_id: str = Field(..., description="ID da tarefa")
    status: TaskStatusEnum = Field(..., description="Status atual da tarefa")
    state: str = Field(..., description="Estado detalhado da tarefa")
    ready: bool = Field(..., description="Se True, a tarefa está concluída (sucesso ou falha)")
    success: Optional[bool] = Field(None, description="Se True, a tarefa foi concluída com sucesso")
    started_at: Optional[float] = Field(None, description="Timestamp de início")
    completed_at: Optional[float] = Field(None, description="Timestamp de conclusão")
    execution_time: Optional[float] = Field(None, description="Tempo de execução em segundos")
    error: Optional[str] = Field(None, description="Mensagem de erro, se houver")
    
class TaskResultResponse(BaseModel):
    """Resposta com o resultado de uma tarefa."""
    task_id: str = Field(..., description="ID da tarefa")
    status: TaskStatusEnum = Field(..., description="Status da tarefa")
    result: Optional[Any] = Field(None, description="Resultado da tarefa, se disponível")
    error: Optional[str] = Field(None, description="Mensagem de erro, se houver")
    result_size: Optional[str] = Field(None, description="Tamanho do resultado")
    execution_time: Optional[float] = Field(None, description="Tempo de execução em segundos")
    
class BatchTaskRequest(BaseModel):
    """Requisição para executar múltiplas tarefas em lote."""
    tasks: List[Dict[str, Any]] = Field(..., description="Lista de definições de tarefas")
    wait_for_results: bool = Field(False, description="Se True, espera pelos resultados")
    timeout: Optional[int] = Field(None, description="Tempo limite em segundos para espera")
    
    @validator('tasks')
    def validate_tasks(cls, v):
        """Valida que há pelo menos uma tarefa."""
        if not v:
            raise ValueError("É necessário fornecer pelo menos uma tarefa")
        return v
    
class BatchTaskResponse(BaseModel):
    """Resposta para execução em lote."""
    task_ids: List[str] = Field(..., description="Lista de IDs das tarefas criadas")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Resultados das tarefas, se disponíveis")
