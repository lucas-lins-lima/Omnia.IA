"""
Workflow Tasks - Tarefas Celery relacionadas a fluxos de trabalho.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import tempfile
import json

from tasks.core import celery_app, task_logger, store_large_result

from orchestrator.core import OrchestratorCore, MediaType
from orchestrator.workflow import WorkflowRegistry

logger = logging.getLogger(__name__)

# Instanciar orquestrador e registro de templates
orchestrator = OrchestratorCore()
workflow_registry = WorkflowRegistry(orchestrator)

# Registrar conversores e executores
# (Código omitido para brevidade - mesmo código do arquivo orchestrator_endpoints.py)

@celery_app.task(name="workflow.execute", bind=True)
@task_logger
async def execute_workflow(
    self,
    workflow_id: str,
    initial_data: Optional[str] = None,
    initial_media_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executa um fluxo de trabalho.
    
    Args:
        workflow_id: ID do fluxo de trabalho
        initial_data: Dados iniciais para o fluxo (opcional)
        initial_media_type: Tipo de mídia dos dados iniciais (opcional)
        
    Returns:
        Resultados do fluxo de trabalho
    """
    try:
        # Converter tipo de mídia para enum
        media_type_enum = None
        if initial_media_type:
            media_type_enum = MediaType(initial_media_type)
            
        # Executar fluxo de trabalho
        result = await orchestrator.execute_workflow(
            workflow_id=workflow_id,
            initial_data=initial_data,
            initial_media_type=media_type_enum
        )
        
        # Para fluxos de trabalho complexos, armazenar resultado em arquivo
        if 'results' in result and len(result['results']) > 5:
            stored_result = store_large_result(result, self.request.id)
            
            # Adicionar informações resumidas
            stored_result.update({
                "workflow_id": workflow_id,
                "status": result["status"],
                "operations_summary": {
                    op["id"]: {"status": op["status"], "media_type": op["media_type"]}
                    for op in result["operations"]
                }
            })
            
            return stored_result
        else:
            return result
            
    except Exception as e:
        logger.error(f"Erro ao executar fluxo de trabalho {workflow_id}: {str(e)}")
        raise

@celery_app.task(name="workflow.create_from_template", bind=True)
@task_logger
def create_workflow_from_template(
    self,
    template_name: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Cria um fluxo de trabalho a partir de um template.
    
    Args:
        template_name: Nome do template
        **kwargs: Parâmetros para o template
        
    Returns:
        Detalhes do fluxo de trabalho criado
    """
    try:
        # Criar fluxo de trabalho a partir do template
        workflow_id = workflow_registry.create_workflow_from_template(
            template_name=template_name,
            **kwargs
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
        logger.error(f"Erro ao criar fluxo de trabalho a partir de template {template_name}: {str(e)}")
        raise

@celery_app.task(name="workflow.batch_process", bind=True)
@task_logger
async def batch_process_workflows(
    self,
    workflow_ids: List[str],
    initial_data_list: Optional[List[str]] = None,
    initial_media_type: Optional[str] = None,
    parallel: bool = False
) -> Dict[str, Any]:
    """
    Executa múltiplos fluxos de trabalho em lote.
    
    Args:
        workflow_ids: Lista de IDs de fluxos de trabalho
        initial_data_list: Lista de dados iniciais para cada fluxo (opcional)
        initial_media_type: Tipo de mídia dos dados iniciais (opcional)
        parallel: Se True, executa fluxos em paralelo; caso contrário, em sequência
        
    Returns:
        Resultados dos fluxos de trabalho
    """
    try:
        results = []
        
        # Verificar parâmetros
        if initial_data_list and len(initial_data_list) != len(workflow_ids):
            raise ValueError("O número de dados iniciais deve corresponder ao número de fluxos de trabalho")
            
        # Converter tipo de mídia para enum
        media_type_enum = None
        if initial_media_type:
            media_type_enum = MediaType(initial_media_type)
            
        # Processar fluxos de trabalho
        if parallel:
            # Executar em paralelo
            import asyncio
            
            tasks = []
            for i, workflow_id in enumerate(workflow_ids):
                initial_data = initial_data_list[i] if initial_data_list else None
                tasks.append(orchestrator.execute_workflow(
                    workflow_id=workflow_id,
                    initial_data=initial_data,
                    initial_media_type=media_type_enum
                ))
                
            # Executar tarefas em paralelo
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Processar resultados
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Registrar erro
                    logger.error(f"Erro ao executar fluxo de trabalho {workflow_ids[i]}: {str(result)}")
                    results.append({
                        "workflow_id": workflow_ids[i],
                        "status": "ERROR",
                        "error": str(result)
                    })
                else:
                    results.append(result)
        else:
            # Executar em sequência
            for i, workflow_id in enumerate(workflow_ids):
                try:
                    initial_data = initial_data_list[i] if initial_data_list else None
                    result = await orchestrator.execute_workflow(
                        workflow_id=workflow_id,
                        initial_data=initial_data,
                        initial_media_type=media_type_enum
                    )
                    results.append(result)
                except Exception as e:
                    # Registrar erro, mas continuar com os próximos fluxos
                    logger.error(f"Erro ao executar fluxo de trabalho {workflow_id}: {str(e)}")
                    results.append({
                        "workflow_id": workflow_id,
                        "status": "ERROR",
                        "error": str(e)
                    })
                    
        # Armazenar resultados em arquivo para conjuntos grandes
        if len(workflow_ids) > 5:
            stored_result = store_large_result(
                {
                    "results": results,
                    "total_workflows": len(workflow_ids),
                    "successful": len([r for r in results if r.get("status") != "ERROR"])
                },
                self.request.id
            )
            
            # Adicionar informações resumidas
            stored_result.update({
                "total_workflows": len(workflow_ids),
                "successful": len([r for r in results if r.get("status") != "ERROR"]),
                "workflow_ids": workflow_ids
            })
            
            return stored_result
        else:
            return {
                "results": results,
                "total_workflows": len(workflow_ids),
                "successful": len([r for r in results if r.get("status") != "ERROR"])
            }
            
    except Exception as e:
        logger.error(f"Erro no processamento em lote de fluxos de trabalho: {str(e)}")
        raise
