"""
Core Orchestrator - Componente central que coordena operações entre diferentes modalidades.
"""

import os
import time
import logging
import json
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum
import tempfile
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class MediaType(str, Enum):
    """Tipos de mídia suportados pelo orquestrador."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    SPREADSHEET = "spreadsheet"
    
class OperationStatus(str, Enum):
    """Status possíveis para uma operação."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    
class WorkflowStatus(str, Enum):
    """Status possíveis para um fluxo de trabalho."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class OrchestratorCore:
    """
    Núcleo do orquestrador que gerencia e executa fluxos de trabalho entre diferentes modalidades.
    """
    
    def __init__(
        self,
        max_concurrent_operations: int = 3,
        temp_dir: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Inicializa o orquestrador.
        
        Args:
            max_concurrent_operations: Número máximo de operações concorrentes
            temp_dir: Diretório para arquivos temporários
            enable_caching: Se True, habilita cache de resultados intermediários
        """
        self.max_concurrent_operations = max_concurrent_operations
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.enable_caching = enable_caching
        
        # Diretório para armazenar resultados intermediários
        self.cache_dir = os.path.join(self.temp_dir, "omnia_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Armazenar workflows em execução e seus resultados
        self.workflows = {}
        self.results_cache = {}
        
        # Pool de executores para operações em background
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_operations)
        
        # Dicionário de conversores entre modalidades
        self.converters = {}
        
        # Dicionário de executores específicos por modalidade
        self.executors = {}
        
        logger.info(f"Orquestrador inicializado com {max_concurrent_operations} operações concorrentes")
        logger.info(f"Cache de resultados: {'habilitado' if enable_caching else 'desabilitado'}")
        
    def register_converter(
        self, 
        source_type: MediaType, 
        target_type: MediaType, 
        converter_func: Callable
    ):
        """
        Registra uma função de conversão entre tipos de mídia.
        
        Args:
            source_type: Tipo de mídia de origem
            target_type: Tipo de mídia de destino
            converter_func: Função que converte entre os tipos
        """
        key = f"{source_type.value}_to_{target_type.value}"
        self.converters[key] = converter_func
        logger.info(f"Conversor registrado: {key}")
        
    def register_executor(
        self, 
        media_type: MediaType, 
        operation: str, 
        executor_func: Callable
    ):
        """
        Registra uma função executora para uma operação específica.
        
        Args:
            media_type: Tipo de mídia
            operation: Nome da operação
            executor_func: Função que executa a operação
        """
        key = f"{media_type.value}_{operation}"
        self.executors[key] = executor_func
        logger.info(f"Executor registrado: {key}")
        
    def convert_media(
        self, 
        data: Any, 
        source_type: MediaType, 
        target_type: MediaType,
        **kwargs
    ) -> Any:
        """
        Converte dados de um tipo de mídia para outro.
        
        Args:
            data: Dados a serem convertidos
            source_type: Tipo de mídia de origem
            target_type: Tipo de mídia de destino
            **kwargs: Parâmetros adicionais para o conversor
            
        Returns:
            Dados convertidos
        """
        # Se os tipos são iguais, retornar os dados diretamente
        if source_type == target_type:
            return data
            
        # Obter a função de conversão
        key = f"{source_type.value}_to_{target_type.value}"
        
        if key not in self.converters:
            raise ValueError(f"Conversor não encontrado: {source_type} para {target_type}")
            
        converter_func = self.converters[key]
        
        try:
            logger.info(f"Convertendo de {source_type} para {target_type}")
            result = converter_func(data, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Erro ao converter de {source_type} para {target_type}: {str(e)}")
            raise
            
    def execute_operation(
        self, 
        media_type: MediaType, 
        operation: str, 
        data: Any,
        **kwargs
    ) -> Any:
        """
        Executa uma operação específica em dados.
        
        Args:
            media_type: Tipo de mídia
            operation: Nome da operação
            data: Dados para a operação
            **kwargs: Parâmetros adicionais para a operação
            
        Returns:
            Resultado da operação
        """
        # Obter a função executora
        key = f"{media_type.value}_{operation}"
        
        if key not in self.executors:
            raise ValueError(f"Executor não encontrado: {media_type} {operation}")
            
        executor_func = self.executors[key]
        
        try:
            logger.info(f"Executando operação {operation} em {media_type}")
            result = executor_func(data, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Erro ao executar {operation} em {media_type}: {str(e)}")
            raise
            
    def create_workflow(
        self, 
        name: str, 
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Cria um novo fluxo de trabalho.
        
        Args:
            name: Nome do fluxo de trabalho
            description: Descrição do fluxo (opcional)
            metadata: Metadados adicionais (opcional)
            
        Returns:
            ID do fluxo de trabalho
        """
        workflow_id = str(uuid.uuid4())
        
        workflow = {
            "id": workflow_id,
            "name": name,
            "description": description,
            "metadata": metadata or {},
            "created_at": time.time(),
            "updated_at": time.time(),
            "status": WorkflowStatus.PENDING,
            "operations": [],
            "results": {},
            "error": None
        }
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Fluxo de trabalho criado: {workflow_id} - {name}")
        
        return workflow_id
        
    def add_operation(
        self,
        workflow_id: str,
        media_type: MediaType,
        operation: str,
        input_from: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Adiciona uma operação a um fluxo de trabalho.
        
        Args:
            workflow_id: ID do fluxo de trabalho
            media_type: Tipo de mídia da operação
            operation: Nome da operação
            input_from: ID da operação anterior para usar como entrada (opcional)
            parameters: Parâmetros para a operação
            name: Nome da operação (opcional)
            description: Descrição da operação (opcional)
            
        Returns:
            ID da operação
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Fluxo de trabalho não encontrado: {workflow_id}")
            
        workflow = self.workflows[workflow_id]
        
        # Verificar se o fluxo de trabalho já foi iniciado
        if workflow["status"] != WorkflowStatus.PENDING:
            raise ValueError(f"Não é possível adicionar operações a um fluxo de trabalho em execução/concluído")
            
        # Gerar ID para a operação
        operation_id = f"op_{len(workflow['operations']) + 1}"
        
        # Nome padrão se não for fornecido
        if name is None:
            name = f"{media_type.value}_{operation}"
            
        # Criar objeto da operação
        op = {
            "id": operation_id,
            "name": name,
            "description": description,
            "media_type": media_type,
            "operation": operation,
            "input_from": input_from,
            "parameters": parameters or {},
            "status": OperationStatus.PENDING,
            "start_time": None,
            "end_time": None,
            "error": None
        }
        
        workflow["operations"].append(op)
        workflow["updated_at"] = time.time()
        
        logger.info(f"Operação adicionada ao fluxo de trabalho {workflow_id}: {operation_id} - {name}")
        
        return operation_id
        
    async def execute_workflow(
        self,
        workflow_id: str,
        initial_data: Optional[Any] = None,
        initial_media_type: Optional[MediaType] = None
    ) -> Dict[str, Any]:
        """
        Executa um fluxo de trabalho completo.
        
        Args:
            workflow_id: ID do fluxo de trabalho
            initial_data: Dados iniciais para o fluxo (opcional)
            initial_media_type: Tipo de mídia dos dados iniciais (opcional)
            
        Returns:
            Resultados do fluxo de trabalho
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Fluxo de trabalho não encontrado: {workflow_id}")
            
        workflow = self.workflows[workflow_id]
        
        # Verificar se já há operações
        if not workflow["operations"]:
            raise ValueError("O fluxo de trabalho não tem operações")
            
        # Atualizar status
        workflow["status"] = WorkflowStatus.RUNNING
        workflow["updated_at"] = time.time()
        
        logger.info(f"Iniciando execução do fluxo de trabalho: {workflow_id}")
        
        try:
            # Inicializar resultados
            if initial_data is not None:
                if initial_media_type is None:
                    raise ValueError("Tipo de mídia inicial deve ser especificado quando dados iniciais são fornecidos")
                    
                # Armazenar dados iniciais como "input"
                workflow["results"]["input"] = {
                    "data": initial_data,
                    "media_type": initial_media_type
                }
                
            # Executar operações em sequência
            for op in workflow["operations"]:
                # Atualizar status da operação
                op["status"] = OperationStatus.RUNNING
                op["start_time"] = time.time()
                workflow["updated_at"] = time.time()
                
                try:
                    # Obter entrada para a operação
                    input_data = None
                    input_media_type = None
                    
                    if op["input_from"] is not None:
                        # Usar resultado de operação anterior
                        if op["input_from"] not in workflow["results"]:
                            # Operação anterior não executada/falhou
                            op["status"] = OperationStatus.SKIPPED
                            op["error"] = f"Entrada não disponível: {op['input_from']}"
                            continue
                            
                        input_result = workflow["results"][op["input_from"]]
                        input_data = input_result["data"]
                        input_media_type = input_result["media_type"]
                    elif "input" in workflow["results"]:
                        # Usar dados iniciais
                        input_result = workflow["results"]["input"]
                        input_data = input_result["data"]
                        input_media_type = input_result["media_type"]
                    else:
                        # Sem entrada disponível
                        op["status"] = OperationStatus.SKIPPED
                        op["error"] = "Nenhuma entrada disponível e input_from não especificado"
                        continue
                        
                    # Verificar se é necessário converter
                    if input_media_type != op["media_type"]:
                        # Converter para o tipo de mídia da operação
                        input_data = self.convert_media(
                            input_data,
                            input_media_type,
                            op["media_type"],
                            **op["parameters"].get("conversion_params", {})
                        )
                        
                    # Executar a operação
                    result = self.execute_operation(
                        op["media_type"],
                        op["operation"],
                        input_data,
                        **op["parameters"].get("operation_params", {})
                    )
                    
                    # Armazenar resultado
                    workflow["results"][op["id"]] = {
                        "data": result,
                        "media_type": op["media_type"]
                    }
                    
                    # Atualizar status
                    op["status"] = OperationStatus.COMPLETED
                    op["end_time"] = time.time()
                    
                except Exception as e:
                    logger.error(f"Erro ao executar operação {op['id']}: {str(e)}")
                    op["status"] = OperationStatus.FAILED
                    op["error"] = str(e)
                    op["end_time"] = time.time()
                    
            # Verificar status final do workflow
            failed_ops = [op for op in workflow["operations"] if op["status"] == OperationStatus.FAILED]
            
            if failed_ops:
                workflow["status"] = WorkflowStatus.FAILED
                workflow["error"] = f"{len(failed_ops)} operações falharam"
            else:
                workflow["status"] = WorkflowStatus.COMPLETED
                
            workflow["updated_at"] = time.time()
            
            # Retornar resultados
            return {
                "workflow_id": workflow_id,
                "status": workflow["status"],
                "results": {k: v for k, v in workflow["results"].items() if k != "input"},
                "operations": [{
                    "id": op["id"],
                    "name": op["name"],
                    "status": op["status"],
                    "media_type": op["media_type"],
                    "operation": op["operation"],
                    "error": op["error"]
                } for op in workflow["operations"]]
            }
            
        except Exception as e:
            logger.error(f"Erro ao executar fluxo de trabalho {workflow_id}: {str(e)}")
            workflow["status"] = WorkflowStatus.FAILED
            workflow["error"] = str(e)
            workflow["updated_at"] = time.time()
            
            # Retornar erro
            return {
                "workflow_id": workflow_id,
                "status": workflow["status"],
                "error": str(e),
                "results": workflow["results"]
            }
            
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Obtém o status atual de um fluxo de trabalho.
        
        Args:
            workflow_id: ID do fluxo de trabalho
            
        Returns:
            Status do fluxo de trabalho
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Fluxo de trabalho não encontrado: {workflow_id}")
            
        workflow = self.workflows[workflow_id]
        
        # Calcular estatísticas
        total_ops = len(workflow["operations"])
        completed_ops = len([op for op in workflow["operations"] if op["status"] == OperationStatus.COMPLETED])
        failed_ops = len([op for op in workflow["operations"] if op["status"] == OperationStatus.FAILED])
        pending_ops = len([op for op in workflow["operations"] if op["status"] == OperationStatus.PENDING])
        running_ops = len([op for op in workflow["operations"] if op["status"] == OperationStatus.RUNNING])
        
        # Calcular progresso
        progress = (completed_ops + failed_ops) / max(1, total_ops) * 100
        
        return {
            "workflow_id": workflow_id,
            "name": workflow["name"],
            "status": workflow["status"],
            "created_at": workflow["created_at"],
            "updated_at": workflow["updated_at"],
            "progress": progress,
            "operations": {
                "total": total_ops,
                "completed": completed_ops,
                "failed": failed_ops,
                "pending": pending_ops,
                "running": running_ops
            },
            "error": workflow["error"]
        }
        
    def cancel_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Cancela a execução de um fluxo de trabalho.
        
        Args:
            workflow_id: ID do fluxo de trabalho
            
        Returns:
            Status atualizado do fluxo de trabalho
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Fluxo de trabalho não encontrado: {workflow_id}")
            
        workflow = self.workflows[workflow_id]
        
        # Verificar se pode ser cancelado
        if workflow["status"] not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
            return self.get_workflow_status(workflow_id)
            
        # Atualizar status
        workflow["status"] = WorkflowStatus.CANCELED
        workflow["updated_at"] = time.time()
        
        # Atualizar operações pendentes
        for op in workflow["operations"]:
            if op["status"] == OperationStatus.PENDING:
                op["status"] = OperationStatus.SKIPPED
                
        logger.info(f"Fluxo de trabalho cancelado: {workflow_id}")
        
        return self.get_workflow_status(workflow_id)
        
    def get_workflow_results(self, workflow_id: str, include_data: bool = False) -> Dict[str, Any]:
        """
        Obtém os resultados de um fluxo de trabalho.
        
        Args:
            workflow_id: ID do fluxo de trabalho
            include_data: Se True, inclui os dados dos resultados
            
        Returns:
            Resultados do fluxo de trabalho
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Fluxo de trabalho não encontrado: {workflow_id}")
            
        workflow = self.workflows[workflow_id]
        
        # Preparar resultados
        results = {}
        
        for op_id, result in workflow["results"].items():
            if op_id == "input":
                continue
                
            if include_data:
                results[op_id] = result
            else:
                # Incluir apenas metadados, não os dados completos
                results[op_id] = {
                    "media_type": result["media_type"],
                    "data_size": self._get_data_size(result["data"])
                }
                
        return {
            "workflow_id": workflow_id,
            "status": workflow["status"],
            "results": results,
            "operations": [{
                "id": op["id"],
                "name": op["name"],
                "status": op["status"],
                "media_type": op["media_type"],
                "operation": op["operation"],
                "start_time": op["start_time"],
                "end_time": op["end_time"],
                "error": op["error"]
            } for op in workflow["operations"]]
        }
        
    def _get_data_size(self, data: Any) -> str:
        """
        Estima o tamanho dos dados.
        
        Args:
            data: Dados para estimar o tamanho
            
        Returns:
            Tamanho estimado como string (ex: "1.5 MB")
        """
        try:
            if isinstance(data, str):
                size_bytes = len(data.encode('utf-8'))
            elif isinstance(data, bytes):
                size_bytes = len(data)
            elif isinstance(data, (list, dict)):
                # Converter para JSON e medir
                size_bytes = len(json.dumps(data).encode('utf-8'))
            else:
                # Fallback para representação em string
                size_bytes = len(str(data).encode('utf-8'))
                
            # Converter para unidade apropriada
            if size_bytes < 1024:
                return f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes/(1024*1024):.1f} MB"
            else:
                return f"{size_bytes/(1024*1024*1024):.1f} GB"
                
        except Exception:
            return "tamanho desconhecido"
