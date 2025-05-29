"""
Celery Worker - Arquivo para iniciar os workers Celery.
"""

import os
import logging
from tasks.core import celery_app

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Iniciando worker Celery...")
    
    # Argumentos para o worker
    args = [
        'worker',
        '--loglevel=INFO',
        # Configurar quantos processos usar (padrão: número de CPUs)
        # '--concurrency=4',
        # Nome do worker
        # '-n=worker1@%h',
        # Prefetch multiplier (controla quantas tarefas um worker pega de uma vez)
        '--prefetch-multiplier=1',
        # Limitar o número de tarefas por processo filho
        '--max-tasks-per-child=1000',
    ]
    
    # Iniciar worker
    celery_app.worker_main(args)
