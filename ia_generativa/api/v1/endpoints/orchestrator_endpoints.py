"""
Endpoints da API relacionados ao orquestrador.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
import time
import logging
from typing import List, Dict, Optional, Union, Any

from orchestrator.core import OrchestratorCore, MediaType
from orchestrator.workflow import WorkflowRegistry
from orchestrator.converters import *
from orchestrator.executors import *

from api.v1.schemas.orchestrator_schemas import (
    CreateWorkflowRequest,
    AddOperationRequest,
    ExecuteWorkflowRequest,
    CreateFromTemplateRequest,
    WorkflowResponse,
    WorkflowStatusResponse,
    WorkflowResultsResponse,
    TemplateListResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/orchestrator",
    tags=["orchestrator"],
    responses={404: {"description": "Not found"}},
)

# Instanciar orquestrador e registro de templates
orchestrator = OrchestratorCore()
workflow_registry = WorkflowRegistry(orchestrator)

# Registrar conversores
orchestrator.register_converter(MediaType.TEXT, MediaType.AUDIO, text_to_audio)
orchestrator.register_converter(MediaType.AUDIO, MediaType.TEXT, audio_to_text)
orchestrator.register_converter(MediaType.IMAGE, MediaType.TEXT, image_to_text)
orchestrator.register_converter(MediaType.TEXT, MediaType.IMAGE, text_to_image)
orchestrator.register_converter(MediaType.VIDEO, MediaType.AUDIO, video_to_audio)
orchestrator.register_converter(MediaType.VIDEO, MediaType.TEXT, video_to_text)
orchestrator.register_converter(MediaType.VIDEO, MediaType.IMAGE, video_to_images)
orchestrator.register_converter(MediaType.PDF, MediaType.TEXT, pdf_to_text)
orchestrator.register_converter(MediaType.SPREADSHEET, MediaType.TEXT, spreadsheet_to_text)

# Registrar executores
orchestrator.register_executor(MediaType.TEXT, "summarize", text_summarize)
orchestrator.register_executor(MediaType.TEXT, "translate", text_translate)
orchestrator.register_executor(MediaType.TEXT, "analyze", text_analyze)
orchestrator.register_executor(MediaType.IMAGE, "classify", image_classify)
orchestrator.register_executor(MediaType.IMAGE, "edit", image_edit)
orchestrator.register_executor(MediaType.AUDIO, "edit", audio_edit)
orchestrator.register_executor(MediaType.AUDIO, "analyze", audio_analyze)
orchestrator.register_executor(MediaType.VIDEO, "edit", video_edit)
orchestrator.register_executor(MediaType.VIDEO, "analyze", video_analyze)

@router.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(request: CreateWorkflowRequest):
    """
    Cria um novo fluxo de trabalho.
    """
    try:
        # Criar fluxo de trabalho
        workflow_id = orchestrator.create_workflow(
            name=request.name,
            description=request.description,
            metadata=request.metadata
        )
        
        # Adicionar operações, se fornecidas
        for op in request.operations:
            orchestrator.add_operation(
                workflow_id=workflow_id,
                media_type=MediaType(op.media_type),
                operation=op.operation,
                input_from=op.input_from,
                parameters=op.parameters,
                name=op.name,
                description=op.description
            )
            
        # Obter detalhes do fluxo de trabalho
        workflow = orchestrator.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "status": workflow["status"],
            "created_at": workflow["created_at"],
            "updated_at": workflow["updated_at"],
            "operations": workflow["operations"]
        }
        
    except Exception as e:
        logger.error(f"Erro ao criar fluxo de trabalho: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar fluxo de trabalho: {str(e)}")
        
@router.post("/workflows/operations", response_model=WorkflowResponse)
async def add_operation(request: AddOperationRequest):
    """
    Adiciona uma operação a um fluxo de trabalho existente.
    """
    try:
        # Adicionar operação
        operation_id = orchestrator.add_operation(
            workflow_id=request.workflow_id,
            media_type=MediaType(request.media_type),
            operation=request.operation,
            input_from=request.input_from,
            parameters=request.parameters,
            name=request.name,
            description=request.description
        )
        
        # Obter detalhes do fluxo de trabalho
        workflow = orchestrator.workflows[request.workflow_id]
        
        return {
            "workflow_id": request.workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "status": workflow["status"],
            "created_at": workflow["created_at"],
            "updated_at": workflow["updated_at"],
            "operations": workflow["operations"]
        }
        
    except ValueError as e:
        logger.error(f"Erro ao adicionar operação: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao adicionar operação: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar operação: {str(e)}")
        
@router.post("/workflows/execute", response_model=WorkflowResultsResponse)
async def execute_workflow(request: ExecuteWorkflowRequest, background_tasks: BackgroundTasks):
    """
    Executa um fluxo de trabalho.
    """
    try:
        # Verificar se o fluxo de trabalho existe
        if request.workflow_id not in orchestrator.workflows:
            raise HTTPException(status_code=404, detail=f"Fluxo de trabalho não encontrado: {request.workflow_id}")
            
        # Preparar dados iniciais
        initial_data = request.initial_data
        initial_media_type = MediaType(request.initial_media_type) if request.initial_media_type else None
        
        # Executar fluxo de trabalho
        result = await orchestrator.execute_workflow(
            workflow_id=request.workflow_id,
            initial_data=initial_data,
            initial_media_type=initial_media_type
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Erro ao executar fluxo de trabalho: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao executar fluxo de trabalho: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao executar fluxo de trabalho: {str(e)}")
        
@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str):
    """
    Obtém informações de um fluxo de trabalho.
    """
    try:
        # Verificar se o fluxo de trabalho existe
        if workflow_id not in orchestrator.workflows:
            raise HTTPException(status_code=404, detail=f"Fluxo de trabalho não encontrado: {workflow_id}")
            
        # Obter detalhes do fluxo de trabalho
        workflow = orchestrator.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "status": workflow["status"],
            "created_at": workflow["created_at"],
            "updated_at": workflow["updated_at"],
            "operations": workflow["operations"]
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter fluxo de trabalho: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter fluxo de trabalho: {str(e)}")
        
@router.get("/workflows/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    Obtém o status atual de um fluxo de trabalho.
    """
    try:
        # Obter status do fluxo de trabalho
        status = orchestrator.get_workflow_status(workflow_id)
        return status
        
    except ValueError as e:
        logger.error(f"Erro ao obter status do fluxo de trabalho: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao obter status do fluxo de trabalho: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter status: {str(e)}")
        
@router.get("/workflows/{workflow_id}/results", response_model=WorkflowResultsResponse)
async def get_workflow_results(workflow_id: str, include_data: bool = False):
    """
    Obtém os resultados de um fluxo de trabalho.
    """
    try:
        # Obter resultados do fluxo de trabalho
        results = orchestrator.get_workflow_results(workflow_id, include_data)
        return results
        
    except ValueError as e:
        logger.error(f"Erro ao obter resultados do fluxo de trabalho: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao obter resultados do fluxo de trabalho: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter resultados: {str(e)}")
        
@router.post("/workflows/{workflow_id}/cancel", response_model=WorkflowStatusResponse)
async def cancel_workflow(workflow_id: str):
    """
    Cancela a execução de um fluxo de trabalho.
    """
    try:
        # Cancelar fluxo de trabalho
        status = orchestrator.cancel_workflow(workflow_id)
        return status
        
    except ValueError as e:
        logger.error(f"Erro ao cancelar fluxo de trabalho: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao cancelar fluxo de trabalho: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao cancelar: {str(e)}")
        
@router.get("/templates", response_model=TemplateListResponse)
async def list_templates():
    """
    Lista os templates disponíveis.
    """
    try:
        # Listar templates
        templates = workflow_registry.list_templates()
        return {"templates": templates}
        
    except Exception as e:
        logger.error(f"Erro ao listar templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao listar templates: {str(e)}")
        
@router.post("/templates/create", response_model=WorkflowResponse)
async def create_from_template(request: CreateFromTemplateRequest):
    """
    Cria um fluxo de trabalho a partir de um template.
    """
    try:
        # Preparar parâmetros
        params = request.parameters or {}
        
        # Adicionar nome e descrição, se fornecidos
        if request.name:
            params["name"] = request.name
        if request.description:
            params["description"] = request.description
            
        # Criar fluxo de trabalho a partir do template
        workflow_id = workflow_registry.create_workflow_from_template(
            template_name=request.template_name,
            **params
        )
        
        # Obter detalhes do fluxo de trabalho
        workflow = orchestrator.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow["name"],
            "description": workflow["description"],
            "status": workflow["status"],
            "created_at": workflow["created_at"],
            "updated_at": workflow["updated_at"],
            "operations": workflow["operations"]
        }
        
    except ValueError as e:
        logger.error(f"Erro ao criar fluxo de trabalho a partir de template: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao criar fluxo de trabalho a partir de template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar fluxo de trabalho: {str(e)}")
