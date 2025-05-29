"""
Text Tasks - Tarefas Celery relacionadas a processamento de texto.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import tempfile
import json

from tasks.core import celery_app, task_logger, store_large_result

from models.text.llm_manager import LLMManager
from models.text.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

# Inicializar gerenciadores
llm_manager = LLMManager()
embedding_manager = EmbeddingManager()

@celery_app.task(name="text.generate", bind=True)
@task_logger
def generate_text(self, prompt: str, model_type: str = "llama3", **kwargs) -> Dict[str, Any]:
    """
    Gera texto com um modelo de linguagem.
    
    Args:
        prompt: Prompt para geração de texto
        model_type: Tipo de modelo a ser usado
        **kwargs: Parâmetros adicionais para geração
        
    Returns:
        Dicionário com o texto gerado e metadados
    """
    try:
        # Formatar o prompt de acordo com o modelo
        from models.text.prompt_templates import PromptManager
        prompt_manager = PromptManager()
        
        formatted_prompt = prompt_manager.format_prompt(
            model_type=model_type,
            user_message=prompt,
            system_prompt=kwargs.get("system_prompt")
        )
        
        # Gerar texto
        generated_text = llm_manager.generate_text(
            prompt=formatted_prompt,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 50),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1)
        )
        
        return {
            "text": generated_text,
            "prompt_length": len(prompt),
            "output_length": len(generated_text),
            "model": model_type
        }
        
    except Exception as e:
        logger.error(f"Erro ao gerar texto: {str(e)}")
        raise

@celery_app.task(name="text.chat", bind=True)
@task_logger
def chat_completion(self, messages: List[Dict[str, str]], model_type: str = "llama3", **kwargs) -> Dict[str, Any]:
    """
    Gera uma resposta para uma conversa.
    
    Args:
        messages: Lista de mensagens da conversa
        model_type: Tipo de modelo a ser usado
        **kwargs: Parâmetros adicionais para geração
        
    Returns:
        Dicionário com a resposta gerada e metadados
    """
    try:
        # Extrair a última mensagem do usuário
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if not user_messages:
            raise ValueError("Nenhuma mensagem do usuário encontrada")
            
        last_user_message = user_messages[-1]["content"]
        
        # Preparar o histórico de chat (excluindo a última mensagem do usuário)
        chat_history = []
        for msg in messages[:-1]:
            chat_history.append({"role": msg["role"], "content": msg["content"]})
            
        # Formatar o prompt
        from models.text.prompt_templates import PromptManager
        prompt_manager = PromptManager()
        
        formatted_prompt = prompt_manager.format_prompt(
            model_type=model_type,
            user_message=last_user_message,
            chat_history=chat_history,
            system_prompt=kwargs.get("system_prompt")
        )
        
        # Gerar resposta
        generated_text = llm_manager.generate_text(
            prompt=formatted_prompt,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 50),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1)
        )
        
        return {
            "text": generated_text,
            "model": model_type,
            "messages_count": len(messages)
        }
        
    except Exception as e:
        logger.error(f"Erro ao completar chat: {str(e)}")
        raise

@celery_app.task(name="text.embeddings", bind=True)
@task_logger
def generate_embeddings(self, texts: Union[str, List[str]], model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Gera embeddings para um texto ou lista de textos.
    
    Args:
        texts: Texto ou lista de textos
        model_name: Nome do modelo de embedding a ser usado
        
    Returns:
        Dicionário com os embeddings e metadados
    """
    try:
        # Normalizar entrada
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = texts
            
        # Verificar se é necessário usar um modelo específico
        if model_name and model_name != embedding_manager.model_name:
            # Criar gerenciador temporário
            temp_manager = EmbeddingManager(model_name_or_path=model_name)
            embeddings = temp_manager.get_embeddings(texts_list)
            model_used = model_name
        else:
            # Usar gerenciador global
            embeddings = embedding_manager.get_embeddings(texts_list)
            model_used = embedding_manager.model_name
            
        # Para resultados grandes, armazenar em arquivo
        if len(texts_list) > 10:
            result = store_large_result(
                {
                    "embeddings": embeddings.tolist(),
                    "dimensions": embeddings.shape[1],
                    "model": model_used,
                    "texts_count": len(texts_list)
                },
                self.request.id
            )
            
            # Adicionar informações resumidas
            result.update({
                "dimensions": embeddings.shape[1],
                "model": model_used,
                "texts_count": len(texts_list)
            })
            
            return result
        else:
            return {
                "embeddings": embeddings.tolist(),
                "dimensions": embeddings.shape[1],
                "model": model_used,
                "texts_count": len(texts_list)
            }
            
    except Exception as e:
        logger.error(f"Erro ao gerar embeddings: {str(e)}")
        raise

@celery_app.task(name="text.summarize", bind=True)
@task_logger
def summarize_text(self, text: str, max_length: int = 200, min_length: int = 50) -> Dict[str, Any]:
    """
    Gera um resumo de um texto.
    
    Args:
        text: Texto para resumir
        max_length: Comprimento máximo do resumo
        min_length: Comprimento mínimo do resumo
        
    Returns:
        Dicionário com o resumo e metadados
    """
    try:
        # Preparar prompt
        prompt = f"""Resuma o seguinte texto em no máximo {max_length} caracteres, 
        mantendo as informações mais importantes. O resumo deve ter pelo menos {min_length} caracteres.
        
        Texto:
        {text}
        
        Resumo:"""
        
        # Gerar resumo
        summary = llm_manager.generate_text(
            prompt=prompt,
            max_new_tokens=max_length,
            temperature=0.3,
            top_p=0.9
        )
        
        # Calcular taxa de compressão
        compression_ratio = len(summary) / len(text) * 100
        
        return {
            "summary": summary.strip(),
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": f"{compression_ratio:.1f}%"
        }
        
    except Exception as e:
        logger.error(f"Erro ao resumir texto: {str(e)}")
        raise
