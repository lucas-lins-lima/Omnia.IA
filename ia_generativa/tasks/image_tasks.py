"""
Image Tasks - Tarefas Celery relacionadas a processamento de imagem.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import tempfile
import json
import base64
from PIL import Image
import io
import numpy as np

from tasks.core import celery_app, task_logger, store_large_result

from models.image.diffusion_manager import DiffusionManager
from models.image.image_understanding import ImageUnderstandingManager
from preprocessors.image.image_processor import load_image_from_base64, convert_to_base64, normalize_image

logger = logging.getLogger(__name__)

# Inicializar gerenciadores
diffusion_manager = DiffusionManager()
image_understanding = ImageUnderstandingManager()

@celery_app.task(name="image.generate", bind=True)
@task_logger
def generate_image(
    self,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    num_images: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Gera imagens a partir de um prompt de texto.
    
    Args:
        prompt: Prompt descritivo para geração da imagem
        negative_prompt: Prompt negativo para guiar o que não deve ser gerado
        width: Largura da imagem em pixels
        height: Altura da imagem em pixels
        num_images: Número de imagens a serem geradas
        **kwargs: Parâmetros adicionais
        
    Returns:
        Dicionário com as imagens geradas e metadados
    """
    try:
        # Extrair parâmetros adicionais
        num_inference_steps = kwargs.get("num_inference_steps", 25)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        seed = kwargs.get("seed", None)
        output_format = kwargs.get("output_format", "PNG")
        
        # Gerar imagens
        images = diffusion_manager.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images=num_images,
            seed=seed
        )
        
        # Converter imagens para base64
        base64_images = [convert_to_base64(img, format=output_format) for img in images]
        
        # Para muitas imagens, armazenar em arquivo
        if num_images > 5:
            result = store_large_result(
                {
                    "images": base64_images,
                    "seed": seed,
                    "prompt": prompt,
                    "width": width,
                    "height": height
                },
                self.request.id
            )
            
            # Adicionar informações resumidas
            result.update({
                "image_count": len(base64_images),
                "seed": seed,
                "prompt": prompt
            })
            
            return result
        else:
            return {
                "images": base64_images,
                "seed": seed,
                "prompt": prompt,
                "width": width,
                "height": height
            }
            
    except Exception as e:
        logger.error(f"Erro ao gerar imagem: {str(e)}")
        raise

@celery_app.task(name="image.transform", bind=True)
@task_logger
def transform_image(
    self,
    image_data: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    strength: float = 0.8,
    **kwargs
) -> Dict[str, Any]:
    """
    Transforma uma imagem existente com base em um prompt.
    
    Args:
        image_data: Imagem base em formato base64
        prompt: Descrição textual da transformação desejada
        negative_prompt: Prompt negativo para guiar o que não deve ser gerado
        strength: Intensidade da transformação (0-1)
        **kwargs: Parâmetros adicionais
        
    Returns:
        Dicionário com as imagens transformadas e metadados
    """
    try:
        # Extrair parâmetros adicionais
        num_inference_steps = kwargs.get("num_inference_steps", 25)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        num_images = kwargs.get("num_images", 1)
        seed = kwargs.get("seed", None)
        output_format = kwargs.get("output_format", "PNG")
        
        # Carregar a imagem de entrada
        input_image = load_image_from_base64(image_data)
        
        # Transformar imagem
        images = diffusion_manager.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images=num_images,
            seed=seed,
            input_image=input_image,
            strength=strength
        )
        
        # Converter imagens para base64
        base64_images = [convert_to_base64(img, format=output_format) for img in images]
        
        return {
            "images": base64_images,
            "seed": seed,
            "prompt": prompt,
            "strength": strength
        }
        
    except Exception as e:
        logger.error(f"Erro ao transformar imagem: {str(e)}")
        raise

@celery_app.task(name="image.caption", bind=True)
@task_logger
def caption_image(
    self,
    image_data: str,
    max_length: int = 30,
    conditional_prompt: Optional[str] = None
) -> Dict[str, str]:
    """
    Gera uma legenda descritiva para uma imagem.
    
    Args:
        image_data: Imagem em formato base64
        max_length: Comprimento máximo da legenda em tokens
        conditional_prompt: Prompt condicional para guiar a geração
        
    Returns:
        Dicionário com a legenda gerada
    """
    try:
        # Carregar imagem
        image = load_image_from_base64(image_data)
        
        # Normalizar imagem para o modelo
        normalized_image = normalize_image(
            image, 
            target_size=(384, 384),
            make_square=False,
            keep_aspect_ratio=True
        )
        
        # Gerar legenda
        caption = image_understanding.generate_caption(
            normalized_image,
            max_length=max_length,
            conditional_prompt=conditional_prompt
        )
        
        return {"caption": caption}
        
    except Exception as e:
        logger.error(f"Erro ao gerar legenda: {str(e)}")
        raise

@celery_app.task(name="image.classify", bind=True)
@task_logger
def classify_image(
    self,
    image_data: str,
    candidate_labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Classifica uma imagem entre várias categorias candidatas.
    
    Args:
        image_data: Imagem em formato base64
        candidate_labels: Lista de categorias candidatas
        
    Returns:
        Dicionário com os scores de classificação
    """
    try:
        # Carregar imagem
        image = load_image_from_base64(image_data)
        
        # Normalizar imagem para o modelo
        normalized_image = normalize_image(
            image, 
            target_size=(336, 336),
            make_square=False,
            keep_aspect_ratio=True
        )
        
        # Classificar imagem
        classification_results = image_understanding.classify_image(
            normalized_image,
            candidate_labels=candidate_labels
        )
        
        return {"classifications": classification_results}
        
    except Exception as e:
        logger.error(f"Erro ao classificar imagem: {str(e)}")
        raise

@celery_app.task(name="image.batch_process", bind=True)
@task_logger
def batch_process_images(
    self,
    image_data_list: List[str],
    operation: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Processa um lote de imagens com a mesma operação.
    
    Args:
        image_data_list: Lista de imagens em formato base64
        operation: Operação a ser aplicada (caption, classify, transform)
        **kwargs: Parâmetros específicos para a operação
        
    Returns:
        Dicionário com os resultados para cada imagem
    """
    try:
        results = []
        
        # Processar cada imagem
        for i, image_data in enumerate(image_data_list):
            logger.info(f"Processando imagem {i+1}/{len(image_data_list)}")
            
            try:
                if operation == "caption":
                    # Gerar legenda
                    result = caption_image.apply(
                        args=[image_data],
                        kwargs={
                            "max_length": kwargs.get("max_length", 30),
                            "conditional_prompt": kwargs.get("conditional_prompt")
                        }
                    ).get()  # Executar de forma síncrona
                    
                    results.append(result)
                    
                elif operation == "classify":
                    # Classificar imagem
                    result = classify_image.apply(
                        args=[image_data, kwargs.get("candidate_labels", [])],
                        kwargs={}
                    ).get()  # Executar de forma síncrona
                    
                    results.append(result)
                    
                elif operation == "transform":
                    # Transformar imagem
                    result = transform_image.apply(
                        args=[
                            image_data,
                            kwargs.get("prompt", "Enhanced version"),
                        ],
                        kwargs={
                            "negative_prompt": kwargs.get("negative_prompt"),
                            "strength": kwargs.get("strength", 0.8),
                            "num_images": kwargs.get("num_images", 1),
                        }
                    ).get()  # Executar de forma síncrona
                    
                    results.append(result)
                    
                else:
                    results.append({
                        "error": f"Operação não suportada: {operation}"
                    })
                    
            except Exception as img_error:
                # Registrar erro para esta imagem específica, mas continuar com as outras
                logger.error(f"Erro ao processar imagem {i+1}: {str(img_error)}")
                results.append({
                    "error": str(img_error)
                })
                
        # Armazenar resultados em arquivo para conjuntos grandes
        if len(image_data_list) > 10:
            result = store_large_result(
                {
                    "results": results,
                    "operation": operation,
                    "total_images": len(image_data_list),
                    "successful": len([r for r in results if "error" not in r])
                },
                self.request.id
            )
            
            # Adicionar informações resumidas
            result.update({
                "total_images": len(image_data_list),
                "successful": len([r for r in results if "error" not in r]),
                "operation": operation
            })
            
            return result
        else:
            return {
                "results": results,
                "operation": operation,
                "total_images": len(image_data_list),
                "successful": len([r for r in results if "error" not in r])
            }
            
    except Exception as e:
        logger.error(f"Erro no processamento em lote: {str(e)}")
        raise
