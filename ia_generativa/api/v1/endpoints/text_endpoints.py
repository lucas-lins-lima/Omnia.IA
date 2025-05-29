"""
Endpoints da API relacionados a processamento de texto.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
import numpy as np
import json
import logging
from typing import List, Dict, Optional, Union, Any

from models.text.llm_manager import LLMManager
from models.text.embedding_manager import EmbeddingManager
from models.text.prompt_templates import PromptManager
from preprocessors.text.cleaner import normalize_text
from api.v1.schemas.text_schemas import (
    TextGenerationRequest, 
    ChatCompletionRequest,
    EmbeddingRequest,
    TextResponse,
    EmbeddingResponse,
    ModelInfoResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/text",
    tags=["text"],
    responses={404: {"description": "Not found"}},
)

# Instanciar gerenciadores
llm_manager = LLMManager()
embedding_manager = EmbeddingManager()
prompt_manager = PromptManager()

@router.post("/generate", response_model=TextResponse)
async def generate_text(request: TextGenerationRequest):
    """
    Gera texto a partir de um prompt.
    """
    try:
        logger.info(f"Recebida requisição para gerar texto: {len(request.prompt)} caracteres")
        
        # Normalizar prompt (remover tags HTML, espaços extras, etc.)
        normalized_prompt = normalize_text(
            request.prompt,
            lower_case=False,  # Não converter para minúsculas para preservar intenção do usuário
            remove_html=True,
            remove_urls_flag=False,  # Preservar URLs que podem ser relevantes
            remove_extra_spaces=True
        )
        
        # Formatar o prompt de acordo com o modelo
        formatted_prompt = prompt_manager.format_prompt(
            model_type=request.model_type,
            user_message=normalized_prompt,
            system_prompt=request.system_prompt
        )
        
        # Gerar texto
        if request.stream:
            # Para streaming, precisamos retornar um StreamingResponse
            streamer = llm_manager.generate_text(
                prompt=formatted_prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                stream=True
            )
            
            async def stream_generator():
                for token in streamer:
                    yield f"data: {json.dumps({'text': token})}\n\n"
                yield "data: [DONE]\n\n"
                
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        else:
            # Geração síncrona
            generated_text = llm_manager.generate_text(
                prompt=formatted_prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                stream=False
            )
            
            return TextResponse(text=generated_text)
            
    except Exception as e:
        logger.error(f"Erro ao gerar texto: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao gerar texto: {str(e)}")
        
@router.post("/chat", response_model=TextResponse)
async def chat_completion(request: ChatCompletionRequest):
    """
    Completa uma conversa a partir de um histórico de mensagens.
    """
    try:
        logger.info(f"Recebida requisição de chat com {len(request.messages)} mensagens")
        
        # Extrair a última mensagem do usuário
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="Nenhuma mensagem do usuário encontrada")
            
        last_user_message = user_messages[-1].content
        
        # Preparar o histórico de chat (excluindo a última mensagem do usuário)
        chat_history = []
        for msg in request.messages[:-1]:
            chat_history.append({"role": msg.role, "content": msg.content})
            
        # Normalizar a última mensagem do usuário
        normalized_message = normalize_text(
            last_user_message,
            lower_case=False,
            remove_html=True,
            remove_urls_flag=False,
            remove_extra_spaces=True
        )
        
        # Formatar o prompt completo com o histórico
        formatted_prompt = prompt_manager.format_prompt(
            model_type=request.model_type,
            user_message=normalized_message,
            chat_history=chat_history,
            system_prompt=request.system_prompt
        )
        
        # Gerar resposta
        if request.stream:
            # Para streaming, precisamos retornar um StreamingResponse
            streamer = llm_manager.generate_text(
                prompt=formatted_prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                stream=True
            )
            
            async def stream_generator():
                for token in streamer:
                    yield f"data: {json.dumps({'text': token})}\n\n"
                yield "data: [DONE]\n\n"
                
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        else:
            # Geração síncrona
            generated_text = llm_manager.generate_text(
                prompt=formatted_prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                stream=False
            )
            
            return TextResponse(text=generated_text)
            
    except Exception as e:
        logger.error(f"Erro ao completar chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao completar chat: {str(e)}")
        
@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Gera embeddings para um texto ou lista de textos.
    """
    try:
        if isinstance(request.texts, str):
            num_texts = 1
            texts = [request.texts]
        else:
            num_texts = len(request.texts)
            texts = request.texts
            
        logger.info(f"Recebida requisição para gerar embeddings para {num_texts} texto(s)")
        
        # Normalizar os textos
        normalized_texts = [
            normalize_text(
                text,
                lower_case=True,  # Para embeddings, geralmente é bom normalizar o caso
                remove_html=True,
                remove_urls_flag=False,
                remove_extra_spaces=True
            ) for text in texts
        ]
        
        # Se um modelo específico foi solicitado, criar um novo gerenciador
        if request.model_name and request.model_name != embedding_manager.model_name:
            temp_embedding_manager = EmbeddingManager(
                model_name_or_path=request.model_name,
                normalize_embeddings=request.normalize
            )
            embeddings = temp_embedding_manager.get_embeddings(normalized_texts)
            model_name = request.model_name
        else:
            # Usar o gerenciador global
            embeddings = embedding_manager.get_embeddings(
                normalized_texts,
                batch_size=32
            )
            model_name = embedding_manager.model_name
            
        # Converter para lista Python
        embeddings_list = embeddings.tolist()
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimensions=embeddings.shape[1],
            model_name=model_name
        )
        
    except Exception as e:
        logger.error(f"Erro ao gerar embeddings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao gerar embeddings: {str(e)}")
        
@router.get("/models/llm/info", response_model=ModelInfoResponse)
async def get_llm_model_info():
    """
    Retorna informações sobre o modelo LLM carregado.
    """
    try:
        model_info = llm_manager.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo LLM: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações do modelo: {str(e)}")
        
@router.get("/models/embedding/info", response_model=ModelInfoResponse)
async def get_embedding_model_info():
    """
    Retorna informações sobre o modelo de embedding carregado.
    """
    try:
        model_info = embedding_manager.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo de embedding: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações do modelo: {str(e)}")

@router.post("/models/llm/unload")
async def unload_llm_model():
    """
    Descarrega o modelo LLM da memória.
    """
    try:
        llm_manager.unload_model()
        return {"status": "success", "message": "Modelo LLM descarregado com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao descarregar modelo LLM: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao descarregar modelo: {str(e)}")
        
@router.post("/models/embedding/unload")
async def unload_embedding_model():
    """
    Descarrega o modelo de embedding da memória.
    """
    try:
        embedding_manager.unload_model()
        return {"status": "success", "message": "Modelo de embedding descarregado com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao descarregar modelo de embedding: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao descarregar modelo: {str(e)}")
