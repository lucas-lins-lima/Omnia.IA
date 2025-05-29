"""
Endpoints da API relacionados a processamento de imagem.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, File, UploadFile, Form
from fastapi.responses import Response
import time
import json
import logging
from typing import List, Dict, Optional, Union, Any
import os
import io
import base64
import random
from PIL import Image
import numpy as np

from models.image.diffusion_manager import DiffusionManager
from models.image.image_understanding import ImageUnderstandingManager
from preprocessors.image.image_processor import (
    load_image_from_base64, 
    normalize_image, 
    resize_image,
    convert_to_base64
)
from api.v1.schemas.image_schemas import (
    TextToImageRequest,
    ImageToImageRequest,
    InpaintingRequest,
    ImageCaptionRequest,
    ImageClassificationRequest,
    ImageSimilarityRequest,
    GeneratedImageResponse,
    ImageCaptionResponse,
    ImageClassificationResponse,
    ImageSimilarityResponse,
    ModelInfoResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/image",
    tags=["image"],
    responses={404: {"description": "Not found"}},
)

# Instanciar gerenciadores
diffusion_manager = DiffusionManager()
image_understanding_manager = ImageUnderstandingManager()

@router.post("/generate", response_model=GeneratedImageResponse)
async def generate_image(request: TextToImageRequest):
    """
    Gera imagens a partir de um prompt de texto.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para gerar imagem: {request.prompt[:50]}...")
        
        # Validar seed ou gerar uma nova
        seed = request.seed
        if seed is None:
            seed = random.randint(0, 2147483647)
            
        # Gerar imagens
        images = diffusion_manager.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.size.width,
            height=request.size.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            seed=seed
        )
        
        # Converter imagens para base64
        base64_images = [
            convert_to_base64(img, format=request.output_format.value.upper()) 
            for img in images
        ]
        
        generation_time = time.time() - start_time
        logger.info(f"Imagens geradas em {generation_time:.2f} segundos")
        
        return GeneratedImageResponse(
            images=base64_images,
            seed=seed,
            prompt=request.prompt,
            generation_time=generation_time,
            model=request.model.value
        )
        
    except Exception as e:
        logger.error(f"Erro ao gerar imagem: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao gerar imagem: {str(e)}")
        
@router.post("/transform", response_model=GeneratedImageResponse)
async def transform_image(request: ImageToImageRequest):
    """
    Transforma uma imagem existente com base em um prompt.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para transformar imagem: {request.prompt[:50]}...")
        
        # Carregar e normalizar a imagem de entrada
        input_image = load_image_from_base64(request.image)
        
        # Manter as dimensões originais
        orig_width, orig_height = input_image.size
        
        # Redimensionar para processamento se necessário (múltiplos de 8)
        width = (orig_width // 8) * 8
        height = (orig_height // 8) * 8
        if width != orig_width or height != orig_height:
            input_image = resize_image(input_image, width, height)
        
        # Validar seed ou gerar uma nova
        seed = request.seed
        if seed is None:
            seed = random.randint(0, 2147483647)
            
        # Transformar imagem
        images = diffusion_manager.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            seed=seed,
            input_image=input_image,
            strength=request.strength
        )
        
        # Redimensionar de volta para as dimensões originais, se solicitado
        if request.resize_to_original and (width != orig_width or height != orig_height):
            images = [resize_image(img, orig_width, orig_height) for img in images]
        
        # Converter imagens para base64
        base64_images = [
            convert_to_base64(img, format=request.output_format.value.upper()) 
            for img in images
        ]
        
        generation_time = time.time() - start_time
        logger.info(f"Imagem transformada em {generation_time:.2f} segundos")
        
        return GeneratedImageResponse(
            images=base64_images,
            seed=seed,
            prompt=request.prompt,
            generation_time=generation_time,
            model=request.model.value
        )
        
    except Exception as e:
        logger.error(f"Erro ao transformar imagem: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao transformar imagem: {str(e)}")
        
@router.post("/inpaint", response_model=GeneratedImageResponse)
async def inpaint_image(request: InpaintingRequest):
    """
    Realiza inpainting (preenchimento) em áreas específicas de uma imagem.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Recebida requisição para inpainting: {request.prompt[:50]}...")
        
        # Carregar imagem e máscara
        input_image = load_image_from_base64(request.image)
        mask_image = load_image_from_base64(request.mask)
        
        # Assegurar que a máscara tem o mesmo tamanho da imagem
        if input_image.size != mask_image.size:
            mask_image = mask_image.resize(input_image.size, Image.LANCZOS)
        
        # Validar seed ou gerar uma nova
        seed = request.seed
        if seed is None:
            seed = random.randint(0, 2147483647)
            
        # Realizar inpainting
        images = diffusion_manager.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            seed=seed,
            input_image=input_image,
            mask_image=mask_image
        )
        
        # Converter imagens para base64
        base64_images = [
            convert_to_base64(img, format=request.output_format.value.upper()) 
            for img in images
        ]
        
        generation_time = time.time() - start_time
        logger.info(f"Inpainting realizado em {generation_time:.2f} segundos")
        
        return GeneratedImageResponse(
            images=base64_images,
            seed=seed,
            prompt=request.prompt,
            generation_time=generation_time,
            model=request.model.value
        )
        
    except Exception as e:
        logger.error(f"Erro ao realizar inpainting: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao realizar inpainting: {str(e)}")
        
@router.post("/caption", response_model=ImageCaptionResponse)
async def caption_image(request: ImageCaptionRequest):
    """
    Gera uma legenda descritiva para uma imagem.
    """
    try:
        logger.info("Recebida requisição para gerar legenda para imagem")
        
        # Carregar imagem
        image = load_image_from_base64(request.image)
        
        # Normalizar imagem para o modelo
        normalized_image = normalize_image(
            image, 
            target_size=(384, 384),
            make_square=False,
            keep_aspect_ratio=True
        )
        
        # Gerar legenda
        caption = image_understanding_manager.generate_caption(
            normalized_image,
            max_length=request.max_length,
            conditional_prompt=request.conditional_prompt
        )
        
        return ImageCaptionResponse(caption=caption)
        
    except Exception as e:
        logger.error(f"Erro ao gerar legenda: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao gerar legenda: {str(e)}")
        
@router.post("/classify", response_model=ImageClassificationResponse)
async def classify_image(request: ImageClassificationRequest):
    """
    Classifica uma imagem entre várias categorias candidatas.
    """
    try:
        logger.info(f"Recebida requisição para classificar imagem entre {len(request.candidate_labels)} categorias")
        
        # Carregar imagem
        image = load_image_from_base64(request.image)
        
        # Normalizar imagem para o modelo
        normalized_image = normalize_image(
            image, 
            target_size=(336, 336),
            make_square=False,
            keep_aspect_ratio=True
        )
        
        # Classificar imagem
        classification_results = image_understanding_manager.classify_image(
            normalized_image,
            candidate_labels=request.candidate_labels
        )
        
        return ImageClassificationResponse(classifications=classification_results)
        
    except Exception as e:
        logger.error(f"Erro ao classificar imagem: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao classificar imagem: {str(e)}")
        
@router.post("/similarity", response_model=ImageSimilarityResponse)
async def calculate_image_text_similarity(request: ImageSimilarityRequest):
    """
    Calcula a similaridade entre uma imagem e várias strings de texto.
    """
    try:
        logger.info(f"Recebida requisição para calcular similaridade entre imagem e {len(request.texts)} textos")
        
        # Carregar imagem
        image = load_image_from_base64(request.image)
        
        # Normalizar imagem para o modelo
        normalized_image = normalize_image(
            image, 
            target_size=(336, 336),
            make_square=False,
            keep_aspect_ratio=True
        )
        
        # Calcular similaridades
        similarities = image_understanding_manager.calculate_image_text_similarity(
            normalized_image,
            texts=request.texts
        )
        
        # Criar dicionário de resultados
        result = {
            text: float(score) 
            for text, score in zip(request.texts, similarities)
        }
        
        return ImageSimilarityResponse(similarities=result)
        
    except Exception as e:
        logger.error(f"Erro ao calcular similaridade: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao calcular similaridade: {str(e)}")
        
@router.get("/models/diffusion/info", response_model=ModelInfoResponse)
async def get_diffusion_model_info():
    """
    Retorna informações sobre o modelo de difusão carregado.
    """
    try:
        model_info = diffusion_manager.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo de difusão: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações do modelo: {str(e)}")
        
@router.get("/models/understanding/info", response_model=ModelInfoResponse)
async def get_understanding_model_info():
    """
    Retorna informações sobre os modelos de entendimento de imagem carregados.
    """
    try:
        model_info = image_understanding_manager.get_model_info()
        return ModelInfoResponse(
            model_name=f"{model_info['caption_model_name']} / {model_info['clip_model_name']}",
            device=model_info['device'],
            dtype=model_info['dtype'],
            pipelines_loaded=model_info['models_loaded'],
            memory_used_gb=model_info['memory_used_gb']
        )
    except Exception as e:
        logger.error(f"Erro ao obter informações dos modelos de entendimento: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações dos modelos: {str(e)}")

@router.post("/models/unload")
async def unload_image_models():
    """
    Descarrega todos os modelos de imagem da memória.
    """
    try:
        diffusion_manager.unload_pipelines()
        image_understanding_manager.unload_models()
        return {"status": "success", "message": "Modelos de imagem descarregados com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao descarregar modelos de imagem: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao descarregar modelos: {str(e)}")
