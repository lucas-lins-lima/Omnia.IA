"""
Endpoints da API relacionados a persistência e memória.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
import logging
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import json

from storage.repositories import UserRepository, ResultRepository, InteractionRepository, PreferenceRepository, UserContextRepository
from memory.user_context import UserContextManager
from memory.preference_learning import PreferenceLearner
from memory.retrieval import ContextRetriever

from api.v1.schemas.storage_schemas import (
    UserRequest,
    StoreResultRequest,
    GetResultRequest,
    SearchResultsRequest,
    DeleteResultRequest,
    RecordInteractionRequest,
    GetUserInteractionsRequest,
    SetPreferenceRequest,
    GetPreferenceRequest,
    GetPreferencesByCategoryRequest,
    DeletePreferenceRequest,
    SetContextRequest,
    GetContextRequest,
    ListContextsRequest,
    BuildUserProfileRequest,
    GetRelevantContextRequest,
    CreateSystemMessageRequest,
    UserResponse,
    ResultResponse,
    ResultMetadataResponse,
    SearchResultsResponse,
    InteractionResponse,
    PreferenceResponse,
    PreferencesCategoryResponse,
    ContextResponse,
    ContextListResponse,
    UserProfileResponse,
    SystemMessageResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/storage",
    tags=["storage"],
    responses={404: {"description": "Not found"}},
)

# Inicializar banco de dados
from storage.database import init_db
init_db()

@router.post("/users", response_model=UserResponse)
async def create_user(request: UserRequest):
    """
    Cria um novo usuário.
    """
    try:
        user = UserRepository.create_user(
            username=request.username,
            email=request.email,
            external_id=request.external_id
        )
        
        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "external_id": user.external_id,
            "created_at": user.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao criar usuário: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar usuário: {str(e)}")
        
@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """
    Obtém um usuário pelo ID.
    """
    try:
        user = UserRepository.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail=f"Usuário não encontrado: {user_id}")
            
        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "external_id": user.external_id,
            "created_at": user.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter usuário: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter usuário: {str(e)}")
        
@router.post("/results", response_model=ResultMetadataResponse)
async def store_result(request: StoreResultRequest):
    """
    Armazena um resultado.
    """
    try:
        result = ResultRepository.store_result(
            user_id=request.user_id,
            result_type=request.result_type,
            content=request.content,
            storage_type=request.storage_type,
            task_id=request.task_id,
            workflow_id=request.workflow_id,
            metadata=request.metadata,
            tags=request.tags,
            is_public=request.is_public,
            ttl_days=request.ttl_days
        )
        
        return {
            "result_id": str(result.result_id),
            "user_id": result.user_id,
            "result_type": result.result_type,
            "storage_type": result.storage_type,
            "metadata": result.metadata,
            "tags": result.tags,
            "is_public": result.is_public,
            "created_at": result.created_at.isoformat(),
            "updated_at": result.updated_at.isoformat(),
            "task_id": result.task_id,
            "workflow_id": result.workflow_id,
            "size_bytes": result.size_bytes
        }
        
    except Exception as e:
        logger.error(f"Erro ao armazenar resultado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao armazenar resultado: {str(e)}")
        
@router.post("/results/get", response_model=ResultResponse)
async def get_result(request: GetResultRequest):
    """
    Recupera um resultado pelo ID.
    """
    try:
        result = ResultRepository.get_result(request.result_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Resultado não encontrado: {request.result_id}")
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter resultado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter resultado: {str(e)}")
        
@router.post("/results/search", response_model=SearchResultsResponse)
async def search_results(request: SearchResultsRequest):
    """
    Busca resultados com filtros.
    """
    try:
        results, total_count = ResultRepository.search_results(
            user_id=request.user_id,
            result_type=request.result_type,
            tags=request.tags,
            query=request.query,
            include_public=request.include_public,
            limit=request.limit,
            offset=request.offset
        )
        
        return {
            "results": results,
            "total": total_count,
            "limit": request.limit,
            "offset": request.offset
        }
        
    except Exception as e:
        logger.error(f"Erro ao buscar resultados: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar resultados: {str(e)}")
        
@router.post("/results/delete", response_model=Dict[str, Any])
async def delete_result(request: DeleteResultRequest):
    """
    Marca um resultado como excluído.
    """
    try:
        success = ResultRepository.delete_result(request.result_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Resultado não encontrado: {request.result_id}")
            
        return {
            "result_id": request.result_id,
            "deleted": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao excluir resultado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao excluir resultado: {str(e)}")
        
@router.post("/interactions", response_model=InteractionResponse)
async def record_interaction(request: RecordInteractionRequest, background_tasks: BackgroundTasks):
    """
    Registra uma interação do usuário.
    """
    try:
        interaction = InteractionRepository.record_interaction(
            user_id=request.user_id,
            interaction_type=request.interaction_type,
            result_id=request.result_id,
            metadata=request.metadata
        )
        
        # Atualizar preferências em background
        background_tasks.add_task(
            PreferenceLearner.update_from_interaction,
            request.user_id,
            request.interaction_type,
            request.metadata or {}
        )
        
        return {
            "id": interaction.id,
            "user_id": interaction.user_id,
            "result_id": interaction.result_id,
            "interaction_type": interaction.interaction_type,
            "metadata": interaction.metadata,
            "created_at": interaction.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao registrar interação: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao registrar interação: {str(e)}")
        
@router.post("/interactions/user", response_model=List[InteractionResponse])
async def get_user_interactions(request: GetUserInteractionsRequest):
    """
    Obtém interações de um usuário.
    """
    try:
        interactions = InteractionRepository.get_user_interactions(
            user_id=request.user_id,
            interaction_type=request.interaction_type,
            limit=request.limit,
            offset=request.offset
        )
        
        return interactions
        
    except Exception as e:
        logger.error(f"Erro ao obter interações: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter interações: {str(e)}")
        
@router.post("/preferences", response_model=PreferenceResponse)
async def set_preference(request: SetPreferenceRequest):
    """
    Define uma preferência de usuário.
    """
    try:
        preference = PreferenceRepository.set_preference(
            user_id=request.user_id,
            category=request.category,
            key=request.key,
            value=request.value
        )
        
        return {
            "user_id": preference.user_id,
            "category": preference.category,
            "key": preference.key,
            "value": request.value  # Usar o valor da requisição diretamente
        }
        
    except Exception as e:
        logger.error(f"Erro ao definir preferência: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao definir preferência: {str(e)}")
        
@router.post("/preferences/get", response_model=PreferenceResponse)
async def get_preference(request: GetPreferenceRequest):
    """
    Obtém uma preferência de usuário.
    """
    try:
        value = PreferenceRepository.get_preference(
            user_id=request.user_id,
            category=request.category,
            key=request.key
        )
        
        if value is None:
            raise HTTPException(status_code=404, detail=f"Preferência não encontrada: {request.category}.{request.key}")
            
        return {
            "user_id": request.user_id,
            "category": request.category,
            "key": request.key,
            "value": value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter preferência: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter preferência: {str(e)}")
        
@router.post("/preferences/category", response_model=PreferencesCategoryResponse)
async def get_preferences_by_category(request: GetPreferencesByCategoryRequest):
    """
    Obtém todas as preferências de uma categoria.
    """
    try:
        preferences = PreferenceRepository.get_preferences_by_category(
            user_id=request.user_id,
            category=request.category
        )
        
        return {
            "user_id": request.user_id,
            "category": request.category,
            "preferences": preferences
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter preferências por categoria: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter preferências: {str(e)}")
        
@router.post("/preferences/delete", response_model=Dict[str, Any])
async def delete_preference(request: DeletePreferenceRequest):
    """
    Remove uma preferência.
    """
    try:
        success = PreferenceRepository.delete_preference(
            user_id=request.user_id,
            category=request.category,
            key=request.key
        )
        
        return {
            "user_id": request.user_id,
            "category": request.category,
            "key": request.key,
            "deleted": success
        }
        
    except Exception as e:
        logger.error(f"Erro ao excluir preferência: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao excluir preferência: {str(e)}")
        
@router.post("/context", response_model=ContextResponse)
async def set_context(request: SetContextRequest):
    """
    Define ou atualiza um contexto de usuário.
    """
    try:
        context = UserContextRepository.set_context(
            user_id=request.user_id,
            context_type=request.context_type,
            context_id=request.context_id,
            state=request.state,
            metadata=request.metadata
        )
        
        return {
            "user_id": context.user_id,
            "context_type": context.context_type,
            "context_id": context.context_id,
            "state": context.state,
            "metadata": context.metadata,
            "version": context.version,
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao definir contexto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao definir contexto: {str(e)}")
        
@router.post("/context/get", response_model=ContextResponse)
async def get_context(request: GetContextRequest):
    """
    Obtém um contexto de usuário.
    """
    try:
        context = UserContextRepository.get_context(
            user_id=request.user_id,
            context_type=request.context_type,
            context_id=request.context_id
        )
        
        if not context:
            raise HTTPException(status_code=404, detail=f"Contexto não encontrado")
            
        return context
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter contexto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter contexto: {str(e)}")
        
@router.post("/context/list", response_model=ContextListResponse)
async def list_contexts(request: ListContextsRequest):
    """
    Lista contextos de um usuário.
    """
    try:
        contexts = UserContextRepository.list_contexts(
            user_id=request.user_id,
            context_type=request.context_type,
            limit=request.limit
        )
        
        return {
            "contexts": contexts
        }
        
    except Exception as e:
        logger.error(f"Erro ao listar contextos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao listar contextos: {str(e)}")
        
@router.post("/user/profile", response_model=UserProfileResponse)
async def build_user_profile(request: BuildUserProfileRequest):
    """
    Constrói um perfil completo do usuário para personalização.
    """
    try:
        profile = UserContextManager.build_user_profile(request.user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail=f"Usuário não encontrado: {request.user_id}")
            
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao construir perfil de usuário: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao construir perfil: {str(e)}")
        
@router.post("/memory/relevant-context", response_model=Dict[str, str])
async def get_relevant_context(request: GetRelevantContextRequest):
    """
    Obtém contexto relevante para um prompt.
    """
    try:
        context = ContextRetriever.get_relevant_context_for_prompt(
            user_id=request.user_id,
            query=request.query,
            max_results=request.max_results
        )
        
        return {
            "context": context
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter contexto relevante: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter contexto relevante: {str(e)}")
        
@router.post("/memory/system-message", response_model=SystemMessageResponse)
async def create_system_message(request: CreateSystemMessageRequest):
    """
    Cria uma mensagem de sistema rica em contexto.
    """
    try:
        message = ContextRetriever.create_system_message_with_context(
            user_id=request.user_id,
            include_preferences=request.include_preferences,
            include_history=request.include_history,
            include_profile=request.include_profile
        )
        
        return message
        
    except Exception as e:
        logger.error(f"Erro ao criar mensagem de sistema: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar mensagem de sistema: {str(e)}")
