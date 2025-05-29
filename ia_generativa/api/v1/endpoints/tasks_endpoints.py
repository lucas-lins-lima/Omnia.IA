"""
Endpoints da API relacionados a gerenciamento de tarefas assíncronas.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
import time
import logging
from typing import List, Dict, Optional, Union, Any
from celery.result import AsyncResult
import asyncio

from tasks.core import celery_app, get_task_info, clean_task_result

from api.v1.schemas.tasks_schemas import (
    CreateTaskRequest,
    TaskStatusRequest,
    TaskCancelRequest,
    TaskResultRequest,
    TaskResponse,
    TaskStatusResponse,
    TaskResultResponse,
    BatchTaskRequest,
    BatchTaskResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    responses={404: {"description": "Not found"}},
)

@router.post("/create", response_model=TaskResponse)
async def create_task(request: CreateTaskRequest):
    """
    Cria uma nova tarefa assíncrona.
    """
    try:
        # Obter referência para a tarefa Celery
        task_func = celery_app.tasks.get(request.task_type)
        
        if task_func is None:
            raise HTTPException(status_code=400, detail=f"Tipo de tarefa não encontrado: {request.task_type}")
            
        # Criar tarefa
        task = task_func.apply_async(args=request.args, kwargs=request.kwargs)
        
        return {
            "task_id": task.id,
            "task_type": request.task_type,
            "status": task.status
        }
        
    except Exception as e:
        logger.error(f"Erro ao criar tarefa: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar tarefa: {str(e)}")
        
@router.post("/status", response_model=TaskStatusResponse)
async def get_task_status(request: TaskStatusRequest):
    """
    Verifica o status de uma tarefa.
    """
    try:
        # Obter informações da tarefa
        task_info = get_task_info(request.task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail=f"Tarefa não encontrada: {request.task_id}")
            
        return task_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao verificar status da tarefa: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao verificar status: {str(e)}")
        
@router.post("/result", response_model=TaskResultResponse)
async def get_task_result(request: TaskResultRequest):
    """
    Obtém o resultado de uma tarefa.
    """
    try:
        # Obter informações da tarefa
        task_info = get_task_info(request.task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail=f"Tarefa não encontrada: {request.task_id}")
            
        # Verificar se a tarefa está concluída
        if not task_info.get("ready"):
            return {
                "task_id": request.task_id,
                "status": task_info.get("status"),
                "error": "Tarefa ainda não concluída"
            }
            
        # Montar resposta
        response = {
            "task_id": request.task_id,
            "status": task_info.get("status"),
            "execution_time": task_info.get("execution_time")
        }
        
        # Incluir resultado ou erro
        if task_info.get("success"):
            if request.include_result:
                response["result"] = task_info.get("result")
            else:
                # Incluir apenas metadados
                if isinstance(task_info.get("result"), dict):
                    # Excluir dados grandes
                    metadata = {}
                    for k, v in task_info.get("result", {}).items():
                        if k not in ["audio", "video", "images", "embeddings"]:
                            metadata[k] = v
                    response["result"] = metadata
                else:
                    response["result_size"] = "Resultado disponível"
        else:
            response["error"] = task_info.get("error")
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter resultado da tarefa: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter resultado: {str(e)}")
        
@router.post("/cancel", response_model=TaskStatusResponse)
async def cancel_task(request: TaskCancelRequest):
    """
    Cancela uma tarefa em execução.
    """
    try:
        # Obter referência para a tarefa
        task = AsyncResult(request.task_id, app=celery_app)
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Tarefa não encontrada: {request.task_id}")
            
        # Cancelar tarefa
        task.revoke(terminate=True)
        
        # Obter status atualizado
        task_info = get_task_info(request.task_id)
        
        return task_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao cancelar tarefa: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao cancelar tarefa: {str(e)}")
        
@router.post("/clean", response_model=Dict[str, Any])
async def clean_task_results(request: TaskStatusRequest):
    """
    Remove os arquivos de resultado de uma tarefa.
    """
    try:
        # Limpar arquivos temporários
        success = clean_task_result(request.task_id)
        
        return {
            "task_id": request.task_id,
            "cleaned": success,
            "message": "Arquivos de resultado removidos com sucesso" if success else "Nenhum arquivo encontrado"
        }
        
    except Exception as e:
        logger.error(f"Erro ao limpar resultados da tarefa: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao limpar resultados: {str(e)}")
        
@router.post("/batch", response_model=BatchTaskResponse)
async def create_batch_tasks(request: BatchTaskRequest):
    """
    Cria múltiplas tarefas em lote.
    """
    try:
        task_ids = []
        results = None
        
        # Criar tarefas
        for task_def in request.tasks:
            # Extrair parâmetros
            task_type = task_def.get("task_type")
            args = task_def.get("args", [])
            kwargs = task_def.get("kwargs", {})
            
            # Obter referência para a tarefa Celery
            task_func = celery_app.tasks.get(task_type)
            
            if task_func is None:
                raise HTTPException(status_code=400, detail=f"Tipo de tarefa não encontrado: {task_type}")
                
            # Criar tarefa
            task = task_func.apply_async(args=args, kwargs=kwargs)
            task_ids.append(task.id)
            
        # Se deve esperar pelos resultados
        if request.wait_for_results:
            # Configurar timeout
            timeout = request.timeout or 60  # Default: 60 segundos
            
            # Esperar pelos resultados
            start_time = time.time()
            results = []
            
            while time.time() - start_time < timeout:
                # Verificar status de todas as tarefas
                all_ready = True
                
                for task_id in task_ids:
                    task_info = get_task_info(task_id)
                    if not task_info.get("ready", False):
                        all_ready = False
                        break
                        
                if all_ready:
                    # Todas as tarefas concluídas
                    results = [get_task_info(task_id) for task_id in task_ids]
                    break
                    
                # Aguardar um pouco antes de verificar novamente
                await asyncio.sleep(0.5)
                
            if not results:
                # Timeout atingido
                results = [get_task_info(task_id) for task_id in task_ids]
                
        return {
            "task_ids": task_ids,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao criar tarefas em lote: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar tarefas em lote: {str(e)}")
