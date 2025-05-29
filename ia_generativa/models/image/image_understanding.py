"""
Image Understanding - Responsável por carregar e utilizar modelos para captioning e análise de imagens.
"""

import os
import torch
from typing import Dict, List, Optional, Union, Any
from PIL import Image
import logging
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel
)

logger = logging.getLogger(__name__)

class ImageUnderstandingManager:
    """
    Gerencia o carregamento e execução de modelos para entendimento e captioning de imagens.
    """
    
    def __init__(
        self,
        caption_model_name: str = "Salesforce/blip-image-captioning-large",
        clip_model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Inicializa o gerenciador de modelos de entendimento de imagem.
        
        Args:
            caption_model_name: ID do modelo de captioning no HuggingFace
            clip_model_name: ID do modelo CLIP no HuggingFace
            device: Dispositivo para execução (cuda, cpu, etc.)
            dtype: Tipo de dados para o modelo (float16, float32)
            cache_dir: Diretório para cache de modelos
        """
        self.caption_model_name = caption_model_name
        self.clip_model_name = clip_model_name
        
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
            
        # Configurar cache para os modelos
        self.cache_dir = cache_dir or os.path.join("data", "models_cache")
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        
        logger.info(f"Inicializando ImageUnderstandingManager")
        logger.info(f"Modelo de captioning: {self.caption_model_name}")
        logger.info(f"Modelo CLIP: {self.clip_model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Inicialmente não carregamos os modelos, apenas quando necessário
        self.caption_processor = None
        self.caption_model = None
        self.clip_processor = None
        self.clip_model = None
        
    def load_caption_model(self):
        """Carrega o modelo de captioning na memória."""
        if self.caption_model is not None:
            logger.info("Modelo de captioning já carregado, pulando.")
            return
        
        logger.info(f"Carregando modelo de captioning: {self.caption_model_name}")
        
        try:
            # Carregar o processador
            self.caption_processor = BlipProcessor.from_pretrained(
                self.caption_model_name,
                cache_dir=self.cache_dir
            )
                        # Carregar o modelo
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                self.caption_model_name,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            logger.info("Modelo de captioning carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de captioning: {str(e)}")
            raise
            
    def load_clip_model(self):
        """Carrega o modelo CLIP na memória."""
        if self.clip_model is not None:
            logger.info("Modelo CLIP já carregado, pulando.")
            return
        
        logger.info(f"Carregando modelo CLIP: {self.clip_model_name}")
        
        try:
            # Carregar o processador
            self.clip_processor = CLIPProcessor.from_pretrained(
                self.clip_model_name,
                cache_dir=self.cache_dir
            )
            
            # Carregar o modelo
            self.clip_model = CLIPModel.from_pretrained(
                self.clip_model_name,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            logger.info("Modelo CLIP carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo CLIP: {str(e)}")
            raise
            
    def unload_models(self):
        """Libera os modelos da memória."""
        if self.caption_model is not None:
            del self.caption_processor
            del self.caption_model
            self.caption_processor = None
            self.caption_model = None
            
        if self.clip_model is not None:
            del self.clip_processor
            del self.clip_model
            self.clip_processor = None
            self.clip_model = None
            
        # Limpar cache CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        logger.info("Modelos de entendimento de imagem descarregados da memória.")
            
    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 30,
        num_beams: int = 4,
        min_length: Optional[int] = None,
        temperature: float = 1.0,
        conditional_prompt: Optional[str] = None
    ) -> str:
        """
        Gera uma legenda descritiva para uma imagem.
        
        Args:
            image: Imagem para gerar legenda
            max_length: Comprimento máximo da legenda em tokens
            num_beams: Número de feixes para busca em feixe
            min_length: Comprimento mínimo da legenda em tokens
            temperature: Temperatura para geração de texto
            conditional_prompt: Prompt condicional para guiar a geração (ex: "Uma foto de")
            
        Returns:
            Legenda gerada para a imagem
        """
        # Carregar o modelo se ainda não foi carregado
        if self.caption_model is None:
            self.load_caption_model()
            
        logger.info("Gerando legenda para imagem")
        
        try:
            # Preparar a imagem
            inputs = self.caption_processor(
                images=image, 
                text=conditional_prompt if conditional_prompt else "",
                return_tensors="pt"
            ).to(self.device)
            
            # Gerar legenda
            with torch.no_grad():
                outputs = self.caption_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    min_length=min_length,
                    temperature=temperature,
                    do_sample=temperature > 1.0,
                )
                
            # Decodificar a legenda
            caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Legenda gerada: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Erro ao gerar legenda: {str(e)}")
            raise
            
    def calculate_image_text_similarity(
        self,
        image: Image.Image,
        texts: List[str]
    ) -> List[float]:
        """
        Calcula a similaridade entre uma imagem e várias strings de texto usando CLIP.
        
        Args:
            image: Imagem para comparar
            texts: Lista de textos para comparar com a imagem
            
        Returns:
            Lista de scores de similaridade (0-1) entre a imagem e cada texto
        """
        # Carregar o modelo se ainda não foi carregado
        if self.clip_model is None:
            self.load_clip_model()
            
        logger.info(f"Calculando similaridade entre imagem e {len(texts)} textos")
        
        try:
            # Preparar inputs
            inputs = self.clip_processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Calcular similaridade
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                
            # Normalizar embeddings
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            
            # Calcular similaridade de cosseno
            similarity = torch.matmul(image_embeds, text_embeds.T).squeeze().cpu().numpy()
            
            # Converter para lista
            if isinstance(similarity, float):
                similarity = [float(similarity)]
            else:
                similarity = similarity.tolist()
                
            logger.info(f"Similaridades calculadas: {similarity}")
            return similarity
            
        except Exception as e:
            logger.error(f"Erro ao calcular similaridade: {str(e)}")
            raise
            
    def classify_image(
        self,
        image: Image.Image,
        candidate_labels: List[str]
    ) -> Dict[str, float]:
        """
        Classifica uma imagem entre várias categorias candidatas usando CLIP.
        
        Args:
            image: Imagem para classificar
            candidate_labels: Lista de categorias candidatas
            
        Returns:
            Dicionário mapeando categorias para scores de confiança
        """
        # Calcular similaridade com cada categoria
        similarity_scores = self.calculate_image_text_similarity(image, candidate_labels)
        
        # Criar dicionário de resultados
        results = {
            label: float(score) 
            for label, score in zip(candidate_labels, similarity_scores)
        }
        
        # Ordenar por score (maior para menor)
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return results
            
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre os modelos carregados."""
        models_loaded = []
        
        if self.caption_model is not None:
            models_loaded.append("caption (BLIP)")
            
        if self.clip_model is not None:
            models_loaded.append("CLIP")
            
        memory_used = None
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
        return {
            "caption_model_name": self.caption_model_name,
            "clip_model_name": self.clip_model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "models_loaded": models_loaded,
            "memory_used_gb": memory_used
        }
