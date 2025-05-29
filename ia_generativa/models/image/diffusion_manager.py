"""
Diffusion Manager - Responsável por carregar, gerenciar e fazer inferência com modelos de difusão.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler
)
from PIL import Image
import logging
import time
import gc

logger = logging.getLogger(__name__)

class DiffusionManager:
    """
    Gerencia o carregamento, execução e gerenciamento de modelos de difusão.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        dtype: Optional[torch.dtype] = None,
        enable_attention_slicing: bool = True,
        enable_xformers: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Inicializa o gerenciador de modelos de difusão.
        
        Args:
            model_name_or_path: ID do modelo no HuggingFace ou caminho local
            device: Dispositivo para execução (cuda, cpu, etc.)
            load_in_8bit: Se True, carrega o modelo quantizado em 8-bit
            dtype: Tipo de dados para o modelo (float16, float32)
            enable_attention_slicing: Se True, habilita fatiamento de atenção para economia de memória
            enable_xformers: Se True, habilita otimizações xFormers se disponíveis
            cache_dir: Diretório para cache de modelos
        """
        self.model_name = model_name_or_path
        
        # Determinar o dispositivo automaticamente se não for especificado
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Determinar o tipo de dados com base no dispositivo e preferências
        if dtype is None:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = dtype
            
        self.load_in_8bit = load_in_8bit
        self.enable_attention_slicing = enable_attention_slicing
        self.enable_xformers = enable_xformers
        
        # Configurar cache para os modelos
        self.cache_dir = cache_dir or os.path.join("data", "models_cache")
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        
        logger.info(f"Inicializando DiffusionManager com modelo: {self.model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Inicialmente não carregamos os pipelines, apenas quando necessário
        self.txt2img_pipeline = None
        self.img2img_pipeline = None
        self.inpaint_pipeline = None
        
    def load_txt2img_pipeline(self):
        """Carrega o pipeline de text-to-image na memória."""
        if self.txt2img_pipeline is not None:
            logger.info("Pipeline text-to-image já carregado, pulando.")
            return
        
        logger.info(f"Carregando pipeline text-to-image: {self.model_name}")
        
        try:
            # Configurar parâmetros para carregamento do modelo
            kwargs = {
                "torch_dtype": self.dtype,
                "cache_dir": self.cache_dir,
                "safety_checker": None,  # Desabilitar o safety checker para performance
                "requires_safety_checker": False
            }
            
            # Adicionar parâmetros condicionais
            if self.load_in_8bit:
                kwargs["load_in_8bit"] = True
            
            # Carregar o pipeline
            self.txt2img_pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                **kwargs
            )
            
            # Mover para o dispositivo adequado
            self.txt2img_pipeline = self.txt2img_pipeline.to(self.device)
            
            # Aplicar otimizações
            if self.enable_attention_slicing:
                self.txt2img_pipeline.enable_attention_slicing()
                
            if self.enable_xformers and self.device == "cuda":
                try:
                    self.txt2img_pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xFormers habilitado com sucesso.")
                except Exception as e:
                    logger.warning(f"Não foi possível habilitar xFormers: {str(e)}")
            
            # Configurar um scheduler mais rápido (DPMSolver++)
            self.txt2img_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.txt2img_pipeline.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True
            )
            
            logger.info("Pipeline text-to-image carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar pipeline text-to-image: {str(e)}")
            raise
            
    def load_img2img_pipeline(self):
        """Carrega o pipeline de image-to-image na memória."""
        if self.img2img_pipeline is not None:
            logger.info("Pipeline image-to-image já carregado, pulando.")
            return
        
        logger.info(f"Carregando pipeline image-to-image: {self.model_name}")
        
        try:
            # Se o pipeline txt2img já está carregado, reutilizar componentes
            if self.txt2img_pipeline is not None:
                logger.info("Reutilizando componentes do pipeline text-to-image")
                self.img2img_pipeline = StableDiffusionImg2ImgPipeline(**self.txt2img_pipeline.components)
                self.img2img_pipeline = self.img2img_pipeline.to(self.device)
            else:
                # Configurar parâmetros para carregamento do modelo
                kwargs = {
                    "torch_dtype": self.dtype,
                    "cache_dir": self.cache_dir,
                    "safety_checker": None,
                    "requires_safety_checker": False
                }
                
                # Adicionar parâmetros condicionais
                if self.load_in_8bit:
                    kwargs["load_in_8bit"] = True
                
                # Carregar o pipeline
                self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model_name,
                    **kwargs
                )
                
                # Mover para o dispositivo adequado
                self.img2img_pipeline = self.img2img_pipeline.to(self.device)
            
            # Aplicar otimizações
            if self.enable_attention_slicing:
                self.img2img_pipeline.enable_attention_slicing()
                
            if self.enable_xformers and self.device == "cuda":
                try:
                    self.img2img_pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Não foi possível habilitar xFormers: {str(e)}")
            
            # Configurar um scheduler mais rápido (DPMSolver++)
            self.img2img_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.img2img_pipeline.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True
            )
            
            logger.info("Pipeline image-to-image carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar pipeline image-to-image: {str(e)}")
            raise
            
    def load_inpaint_pipeline(self):
        """Carrega o pipeline de inpainting na memória."""
        if self.inpaint_pipeline is not None:
            logger.info("Pipeline inpainting já carregado, pulando.")
            return
        
        logger.info(f"Carregando pipeline inpainting: {self.model_name}")
        
        try:
            # Configurar parâmetros para carregamento do modelo
            kwargs = {
                "torch_dtype": self.dtype,
                "cache_dir": self.cache_dir,
                "safety_checker": None,
                "requires_safety_checker": False
            }
            
            # Adicionar parâmetros condicionais
            if self.load_in_8bit:
                kwargs["load_in_8bit"] = True
            
            # Carregar o pipeline
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_name,
                **kwargs
            )
            
            # Mover para o dispositivo adequado
            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
            
            # Aplicar otimizações
            if self.enable_attention_slicing:
                self.inpaint_pipeline.enable_attention_slicing()
                
            if self.enable_xformers and self.device == "cuda":
                try:
                    self.inpaint_pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Não foi possível habilitar xFormers: {str(e)}")
            
            # Configurar um scheduler mais rápido (DPMSolver++)
            self.inpaint_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.inpaint_pipeline.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True
            )
            
            logger.info("Pipeline inpainting carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar pipeline inpainting: {str(e)}")
            raise
            
    def unload_pipelines(self):
        """Libera todos os pipelines da memória."""
        if self.txt2img_pipeline is not None:
            del self.txt2img_pipeline
            self.txt2img_pipeline = None
            
        if self.img2img_pipeline is not None:
            del self.img2img_pipeline
            self.img2img_pipeline = None
            
        if self.inpaint_pipeline is not None:
            del self.inpaint_pipeline
            self.inpaint_pipeline = None
            
        # Limpar cache CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
        logger.info("Todos os pipelines foram descarregados da memória.")
            
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        input_image: Optional[Image.Image] = None,
        strength: float = 0.8,
        mask_image: Optional[Image.Image] = None,
    ) -> List[Image.Image]:
        """
        Gera imagens a partir de um prompt de texto.
        
        Args:
            prompt: Prompt descritivo para geração da imagem
            negative_prompt: Prompt negativo para guiar o que não deve ser gerado
            width: Largura da imagem em pixels (múltiplo de 8 recomendado)
            height: Altura da imagem em pixels (múltiplo de 8 recomendado)
            num_inference_steps: Número de passos de inferência
            guidance_scale: Escala de orientação para seguir o prompt (valores mais altos = mais fiéis)
            num_images: Número de imagens a serem geradas
            seed: Semente para geração determinística (opcional)
            input_image: Imagem de entrada para img2img ou inpainting
            strength: Intensidade da transformação para img2img (0-1)
            mask_image: Máscara para inpainting (áreas brancas serão preenchidas)
            
        Returns:
            Lista de imagens geradas
        """
        start_time = time.time()
        
        # Determinar qual pipeline usar
        if input_image is None and mask_image is None:
            # Text-to-Image
            if self.txt2img_pipeline is None:
                self.load_txt2img_pipeline()
                
            pipeline = self.txt2img_pipeline
            logger.info(f"Gerando imagem text-to-image, prompt: {prompt[:50]}...")
            
        elif input_image is not None and mask_image is not None:
            # Inpainting
            if self.inpaint_pipeline is None:
                self.load_inpaint_pipeline()
                
            pipeline = self.inpaint_pipeline
            logger.info(f"Gerando imagem com inpainting, prompt: {prompt[:50]}...")
            
        elif input_image is not None:
            # Image-to-Image
            if self.img2img_pipeline is None:
                self.load_img2img_pipeline()
                
            pipeline = self.img2img_pipeline
            logger.info(f"Gerando imagem image-to-image, prompt: {prompt[:50]}...")
            
        else:
            raise ValueError("Configuração inválida: mask_image precisa de input_image")
        
        # Definir gerador para seed fixa, se fornecida
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            logger.info(f"Usando seed fixa: {seed}")
            
        try:
            # Preparar argumentos comuns para todos os pipelines
            common_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images,
                "generator": generator,
            }
            
            # Executar pipeline específico com argumentos adequados
            if input_image is None and mask_image is None:
                # Text-to-Image
                result = pipeline(
                    **common_args,
                    width=width,
                    height=height,
                )
                
            elif input_image is not None and mask_image is not None:
                # Inpainting
                result = pipeline(
                    **common_args,
                    image=input_image,
                    mask_image=mask_image,
                )
                
            else:
                # Image-to-Image
                result = pipeline(
                    **common_args,
                    image=input_image,
                    strength=strength,
                )
                
            elapsed_time = time.time() - start_time
            logger.info(f"Imagem(ns) gerada(s) em {elapsed_time:.2f} segundos")
            
            # Obter as imagens geradas
            images = result.images
            return images
            
        except Exception as e:
            logger.error(f"Erro ao gerar imagem: {str(e)}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre os modelos carregados."""
        pipelines_loaded = []
        
        if self.txt2img_pipeline is not None:
            pipelines_loaded.append("text-to-image")
            
        if self.img2img_pipeline is not None:
            pipelines_loaded.append("image-to-image")
            
        if self.inpaint_pipeline is not None:
            pipelines_loaded.append("inpainting")
            
        memory_used = None
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "pipelines_loaded": pipelines_loaded,
            "memory_used_gb": memory_used,
            "optimizations": {
                "attention_slicing": self.enable_attention_slicing,
                "xformers": self.enable_xformers,
                "load_in_8bit": self.load_in_8bit
            }
        }
