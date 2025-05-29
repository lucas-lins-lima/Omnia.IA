"""
Endpoints da API relacionados a busca semântica.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, File, UploadFile, Form
from fastapi.responses import Response
import time
import logging
import json
import base64
from typing import List, Dict, Optional, Union, Any
from PIL import Image
import io

from semantic_search.search.semantic_search import SemanticSearchEngine
from storage.repositories import UserRepository

from api.v1.schemas.semantic_search_schemas import (
    CreateIndexRequest,
    IndexTextRequest,
    IndexImageRequest,
    IndexRepositoryRequest,
    SearchTextRequest,
    SearchImageRequest,
    SearchImagesWithTextRequest,
    IndexListResponse,
    IndexResponse,
    IndexOperationResponse,
    IndexRepositoryResponse,
    SearchResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/semantic-search",
    tags=["semantic-search"],
    responses={404: {"description": "Not found"}},
)

# Inicializar motor de busca semântica
search_engine = SemanticSearchEngine()

@router.post("/indices", response_model=IndexResponse)
async def create_index(request: CreateIndexRequest):
    """
    Cria um novo índice para busca semântica.
    """
    try:
        # Criar índice
        index_id = search_engine.create_index_for_content_type(
            content_type=request.content_type,
            dimension=request.dimension,
            index_type=request.index_type,
            metric=request.metric
        )
        
        # Obter informações do índice
        index = search_engine.index_manager.get_index(index_id)
        info = index.get_index_info()
        
        return {
            "index_id": index_id,
            "dimension": info["dimension"],
            "index_type": info["index_type"],
            "metric": info["metric"],
            "size": info["size"],
            "content_type": request.content_type
        }
        
    except Exception as e:
        logger.error(f"Erro ao criar índice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar índice: {str(e)}")
        
@router.get("/indices", response_model=IndexListResponse)
async def list_indices():
    """
    Lista todos os índices disponíveis.
    """
    try:
        # Listar índices
        indices = search_engine.index_manager.list_indices()
        
        return {
            "indices": indices
        }
        
    except Exception as e:
        logger.error(f"Erro ao listar índices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao listar índices: {str(e)}")
        
@router.get("/indices/{index_id}", response_model=IndexResponse)
async def get_index(index_id: str):
    """
    Obtém informações de um índice específico.
    """
    try:
        # Verificar se o índice existe
        index = search_engine.index_manager.get_index(index_id)
        
        if index is None:
            raise HTTPException(status_code=404, detail=f"Índice não encontrado: {index_id}")
            
        # Obter informações
        info = index.get_index_info()
        
        # Determinar tipo de conteúdo
        content_type = None
        for ctype, idx in search_engine.content_type_indices.items():
            if idx == index_id:
                content_type = ctype
                break
                
        return {
            "index_id": index_id,
            "dimension": info["dimension"],
            "index_type": info["index_type"],
            "metric": info["metric"],
            "size": info["size"],
            "content_type": content_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter informações do índice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações: {str(e)}")
        
@router.post("/indices/text", response_model=IndexOperationResponse)
async def index_text(request: IndexTextRequest):
    """
    Indexa textos para busca semântica.
    """
    try:
        # Indexar textos
        item_ids = search_engine.index_text(
            texts=request.texts,
            metadata=request.metadata,
            index_id=request.index_id
        )
        
        # Determinar índice usado
        index_id = request.index_id or search_engine.get_index_for_content_type("text")
        
        return {
            "success": True,
            "index_id": index_id,
            "item_ids": item_ids,
            "count": len(item_ids)
        }
        
    except Exception as e:
        logger.error(f"Erro ao indexar textos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao indexar textos: {str(e)}")
        
@router.post("/indices/image", response_model=IndexOperationResponse)
async def index_image(request: IndexImageRequest):
    """
    Indexa imagens para busca semântica.
    """
    try:
        # Indexar imagens
        item_ids = search_engine.index_image(
            images=request.images,
            metadata=request.metadata,
            index_id=request.index_id
        )
        
        # Determinar índice usado
        index_id = request.index_id or search_engine.get_index_for_content_type("image")
        
        return {
            "success": True,
            "index_id": index_id,
            "item_ids": item_ids,
            "count": len(item_ids)
        }
        
    except Exception as e:
        logger.error(f"Erro ao indexar imagens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao indexar imagens: {str(e)}")
        
@router.post("/indices/repository", response_model=IndexRepositoryResponse)
async def index_repository(request: IndexRepositoryRequest, background_tasks: BackgroundTasks):
    """
    Indexa conteúdo do repositório para busca semântica.
    """
    try:
        # Validar usuário se fornecido
        if request.user_id is not None:
            user = UserRepository.get_user_by_id(request.user_id)
            if not user:
                raise HTTPException(status_code=404, detail=f"Usuário não encontrado: {request.user_id}")
                
        # Executar indexação em segundo plano para evitar timeout
        if request.limit > 100:
            # Executar em background
            background_tasks.add_task(
                search_engine.index_content_from_repository,
                user_id=request.user_id,
                result_types=request.result_types,
                limit=request.limit,
                reindex=request.reindex
            )
            
            return {
                "success": True,
                "counts": {},
                "total": 0,
                "message": f"Indexação iniciada em segundo plano com limite de {request.limit} itens por tipo"
            }
        else:
            # Executar de forma síncrona para resultados imediatos
            counts = search_engine.index_content_from_repository(
                user_id=request.user_id,
                result_types=request.result_types,
                limit=request.limit,
                reindex=request.reindex
            )
            
            return {
                "success": True,
                "counts": counts,
                "total": sum(counts.values())
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao indexar repositório: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao indexar repositório: {str(e)}")
        
@router.post("/search/text", response_model=SearchResponse)
async def search_text(request: SearchTextRequest):
    """
    Busca textos semanticamente semelhantes.
    """
    try:
        # Criar função de filtro se necessário
        filter_func = None
        if request.user_id is not None:
            filter_func = lambda _, meta: meta.get("user_id") == request.user_id
            
        # Buscar textos
        results = search_engine.search_text(
                        query=request.query,
            index_id=request.index_id,
            k=request.k,
            include_content=request.include_content,
            filter_func=filter_func
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Erro ao buscar textos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar textos: {str(e)}")
        
@router.post("/search/image", response_model=SearchResponse)
async def search_image(request: SearchImageRequest):
    """
    Busca imagens visualmente semelhantes.
    """
    try:
        # Criar função de filtro se necessário
        filter_func = None
        if request.user_id is not None:
            filter_func = lambda _, meta: meta.get("user_id") == request.user_id
            
        # Buscar imagens
        results = search_engine.search_image(
            query_image=request.image,
            index_id=request.index_id,
            k=request.k,
            include_content=request.include_content,
            filter_func=filter_func
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Erro ao buscar imagens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar imagens: {str(e)}")
        
@router.post("/search/images-with-text", response_model=SearchResponse)
async def search_images_with_text(request: SearchImagesWithTextRequest):
    """
    Busca imagens usando uma consulta de texto (CLIP).
    """
    try:
        # Criar função de filtro se necessário
        filter_func = None
        if request.user_id is not None:
            filter_func = lambda _, meta: meta.get("user_id") == request.user_id
            
        # Buscar imagens com texto
        results = search_engine.search_images_with_text(
            query=request.query,
            index_id=request.index_id,
            k=request.k,
            include_content=request.include_content,
            filter_func=filter_func
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Erro ao buscar imagens com texto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar imagens com texto: {str(e)}")
        
@router.delete("/indices/{index_id}", response_model=Dict[str, Any])
async def delete_index(index_id: str):
    """
    Remove um índice.
    """
    try:
        # Verificar se o índice existe
        index = search_engine.index_manager.get_index(index_id)
        
        if index is None:
            raise HTTPException(status_code=404, detail=f"Índice não encontrado: {index_id}")
            
        # Remover referência em content_type_indices
        for content_type, idx in list(search_engine.content_type_indices.items()):
            if idx == index_id:
                del search_engine.content_type_indices[content_type]
                
        # Remover índice
        success = search_engine.index_manager.delete_index(index_id)
        
        return {
            "success": success,
            "index_id": index_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao remover índice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao remover índice: {str(e)}")
        
@router.post("/indices/{index_id}/save", response_model=Dict[str, Any])
async def save_index(index_id: str):
    """
    Salva um índice em disco.
    """
    try:
        # Verificar se o índice existe
        index = search_engine.index_manager.get_index(index_id)
        
        if index is None:
            raise HTTPException(status_code=404, detail=f"Índice não encontrado: {index_id}")
            
        # Salvar índice
        file_path = search_engine.index_manager.save_index(index_id)
        
        return {
            "success": True,
            "index_id": index_id,
            "file_path": file_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao salvar índice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao salvar índice: {str(e)}")
        
@router.post("/upload-image", response_model=Dict[str, Any])
async def upload_image(file: UploadFile = File(...), index: bool = Form(False), index_id: Optional[str] = Form(None)):
    """
    Faz upload de uma imagem e opcionalmente a indexa.
    """
    try:
        # Ler conteúdo do arquivo
        contents = await file.read()
        
        # Verificar se é uma imagem válida
        try:
            img = Image.open(io.BytesIO(contents))
            width, height = img.size
        except Exception as img_error:
            raise HTTPException(status_code=400, detail=f"Arquivo não é uma imagem válida: {str(img_error)}")
            
        # Converter para base64
        base64_image = base64.b64encode(contents).decode('utf-8')
        base64_image = f"data:image/{file.content_type.split('/')[-1]};base64,{base64_image}"
        
        response = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(contents),
            "width": width,
            "height": height,
            "base64": base64_image
        }
        
        # Indexar se solicitado
        if index:
            # Adicionar metadados básicos
            metadata = {
                "filename": file.filename,
                "content_type": file.content_type,
                "width": width,
                "height": height,
                "uploaded_at": time.time()
            }
            
            # Indexar imagem
            item_ids = search_engine.index_image(
                images=base64_image,
                metadata=metadata,
                index_id=index_id
            )
            
            # Adicionar informações de indexação à resposta
            response["indexed"] = True
            response["item_id"] = item_ids[0] if item_ids else None
            response["index_id"] = index_id or search_engine.get_index_for_content_type("image")
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao processar upload de imagem: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar upload: {str(e)}")
