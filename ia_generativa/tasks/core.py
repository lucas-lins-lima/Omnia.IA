"""
Tasks Core - Configuração principal do Celery e funções utilitárias.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from celery import Celery
from celery.result import AsyncResult
import tempfile

# Configuração de logging
logger = logging.getLogger(__name__)

# Configuração do Celery
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
TASK_SERIALIZER = 'json'
RESULT_SERIALIZER = 'json'
ACCEPT_CONTENT = ['json']
TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# Criar aplicação Celery
celery_app = Celery(
    'omnia_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Configurar aplicação
celery_app.conf.update(
    task_serializer=TASK_SERIALIZER,
    result_serializer=RESULT_SERIALIZER,
    accept_content=ACCEPT_CONTENT,
    timezone=TIMEZONE,
    enable_utc=CELERY_ENABLE_UTC,
    task_track_started=True,
    task_time_limit=3600,  # 1 hora
    worker_max_tasks_per_child=200,
    worker_prefetch_multiplier=1
)

# Diretório para armazenar resultados temporários
TASK_RESULTS_DIR = os.environ.get('TASK_RESULTS_DIR', os.path.join(tempfile.gettempdir(), 'omnia_tasks'))
os.makedirs(TASK_RESULTS_DIR, exist_ok=True)

class TaskStatus:
    """Status possíveis para uma tarefa."""
    PENDING = 'PENDING'
    STARTED = 'STARTED'
    SUCCESS = 'SUCCESS'
    FAILURE = 'FAILURE'
    REVOKED = 'REVOKED'
    RETRY = 'RETRY'

def get_task_info(task_id: str) -> Dict[str, Any]:
    """
    Obtém informações sobre uma tarefa.
    
    Args:
        task_id: ID da tarefa
        
    Returns:
        Dicionário com informações da tarefa
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        task_info = {
            'task_id': task_id,
            'status': result.status,
            'state': result.state,
            'success': result.successful() if result.ready() else None,
            'ready': result.ready()
        }
        
        # Adicionar tempo de início, se disponível
        if hasattr(result, 'date_start') and result.date_start:
            task_info['started_at'] = result.date_start.timestamp()
            
        # Adicionar resultado, se disponível
        if result.ready():
            if result.successful():
                # Resultado bem-sucedido
                # Para resultados grandes, verificar se há um caminho de arquivo
                raw_result = result.result
                if isinstance(raw_result, dict) and 'result_file_path' in raw_result:
                    # O resultado está armazenado em um arquivo
                    file_path = raw_result['result_file_path']
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r') as f:
                                task_info['result'] = json.load(f)
                        except Exception as e:
                            task_info['result'] = f"Erro ao carregar resultado: {str(e)}"
                    else:
                        task_info['result'] = "Arquivo de resultado não encontrado"
                else:
                    # O resultado está diretamente disponível
                    task_info['result'] = raw_result
                
                task_info['completed_at'] = time.time()
                task_info['execution_time'] = task_info.get('completed_at', 0) - task_info.get('started_at', 0)
            else:
                # Tarefa falhou
                task_info['error'] = str(result.result)
                
                if hasattr(result, 'date_done') and result.date_done:
                    task_info['failed_at'] = result.date_done.timestamp()
                    task_info['execution_time'] = task_info.get('failed_at', 0) - task_info.get('started_at', 0)
                
        return task_info
        
    except Exception as e:
        logger.error(f"Erro ao obter informações da tarefa {task_id}: {str(e)}")
        return {
            'task_id': task_id,
            'status': 'ERROR',
            'error': f"Erro ao obter informações: {str(e)}"
        }

def store_large_result(result: Any, task_id: str) -> Dict[str, Any]:
    """
    Armazena resultados grandes em um arquivo e retorna um identificador.
    
    Args:
        result: Resultado a ser armazenado
        task_id: ID da tarefa
        
    Returns:
        Dicionário com caminho para o arquivo de resultado
    """
    try:
        # Criar caminho para o arquivo
        result_file = os.path.join(TASK_RESULTS_DIR, f"task_result_{task_id}.json")
        
        # Salvar resultado
        with open(result_file, 'w') as f:
            json.dump(result, f)
            
        # Retornar referência
        return {
            'result_file_path': result_file,
            'result_size': os.path.getsize(result_file)
        }
        
    except Exception as e:
        logger.error(f"Erro ao armazenar resultado da tarefa {task_id}: {str(e)}")
        return {
            'error': f"Erro ao armazenar resultado: {str(e)}",
            'result_summary': str(result)[:1000] + '...' if len(str(result)) > 1000 else str(result)
        }

def clean_task_result(task_id: str) -> bool:
    """
    Remove o arquivo de resultado de uma tarefa.
    
    Args:
        task_id: ID da tarefa
        
    Returns:
        True se o arquivo foi removido com sucesso, False caso contrário
    """
    try:
        result_file = os.path.join(TASK_RESULTS_DIR, f"task_result_{task_id}.json")
        
        if os.path.exists(result_file):
            os.remove(result_file)
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Erro ao remover arquivo de resultado da tarefa {task_id}: {str(e)}")
        return False

# Decorador para log de tarefas
def task_logger(func):
    """Decorador para adicionar logs a tarefas Celery."""
    def wrapper(*args, **kwargs):
        task_name = func.__name__
        logger.info(f"Iniciando tarefa {task_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"Tarefa {task_name} concluída em {elapsed_time:.2f} segundos")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Erro na tarefa {task_name} após {elapsed_time:.2f} segundos: {str(e)}")
            raise
            
    return wrapper

# Registrar tarefas descobertas automaticamente
celery_app.autodiscover_tasks(['tasks.text_tasks', 'tasks.image_tasks', 'tasks.audio_tasks', 'tasks.video_tasks', 'tasks.workflow_tasks'])
