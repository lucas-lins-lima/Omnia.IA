"""
Video Understanding - Responsável por carregar e utilizar modelos para análise e compreensão de vídeo.
"""

import os
import torch
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import logging
import time
from transformers import pipeline
from PIL import Image
import cv2

from preprocessors.video.video_loader import load_video

logger = logging.getLogger(__name__)

class VideoUnderstandingManager:
    """
    Gerencia o carregamento e execução de modelos para análise e compreensão de vídeo.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        frame_sample_rate: int = 10
    ):
        """
        Inicializa o gerenciador de análise de vídeo.
        
        Args:
            device: Dispositivo para execução (cuda, cpu, etc.)
            dtype: Tipo de dados para o modelo (float16, float32)
            cache_dir: Diretório para cache de modelos
            frame_sample_rate: Taxa de amostragem de frames para análise
        """
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
        
        self.frame_sample_rate = frame_sample_rate
        
        logger.info(f"Inicializando VideoUnderstandingManager")
        logger.info(f"Dispositivo: {self.device}")
        
        # Inicialmente não carregamos os modelos, apenas quando necessário
        self.video_classifier = None
        self.object_detector = None
        self.video_captioner = None
        
    def load_video_classifier(self, model_name: str = "microsoft/xclip-base-patch32"):
        """Carrega o modelo de classificação de vídeo."""
        if self.video_classifier is not None:
            logger.info("Modelo de classificação de vídeo já carregado.")
            return
            
        try:
            logger.info(f"Carregando modelo de classificação de vídeo: {model_name}")
            
            from transformers import AutoProcessor, AutoModelForVideoClassification
            
            self.video_classifier_processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            self.video_classifier = AutoModelForVideoClassification.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            logger.info("Modelo de classificação de vídeo carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de classificação de vídeo: {str(e)}")
            raise
            
    def load_object_detector(self, model_name: str = "facebook/detr-resnet-50"):
        """Carrega o modelo de detecção de objetos."""
        if self.object_detector is not None:
            logger.info("Modelo de detecção de objetos já carregado.")
            return
            
        try:
            logger.info(f"Carregando modelo de detecção de objetos: {model_name}")
            
            self.object_detector = pipeline(
                "object-detection",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            )
            
            logger.info("Modelo de detecção de objetos carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de detecção de objetos: {str(e)}")
            raise
            
    def load_video_captioner(self, model_name: str = "Salesforce/blip-image-captioning-large"):
        """Carrega o modelo para geração de legendas de vídeo."""
        if self.video_captioner is not None:
            logger.info("Modelo de geração de legendas já carregado.")
            return
            
        try:
            logger.info(f"Carregando modelo de geração de legendas: {model_name}")
            
            self.video_captioner = pipeline(
                "image-to-text",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            )
            
            logger.info("Modelo de geração de legendas carregado com sucesso.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de geração de legendas: {str(e)}")
            raise
            
    def unload_models(self):
        """Libera os modelos da memória."""
        if self.video_classifier is not None:
            del self.video_classifier
            del self.video_classifier_processor
            self.video_classifier = None
            self.video_classifier_processor = None
            
        if self.object_detector is not None:
            del self.object_detector
            self.object_detector = None
            
        if self.video_captioner is not None:
            del self.video_captioner
            self.video_captioner = None
            
        # Limpar cache CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        logger.info("Modelos de análise de vídeo descarregados da memória.")
            
    def classify_video(
        self,
        video_path: str,
        top_k: int = 5,
        frame_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Classifica um vídeo em categorias.
        
        Args:
            video_path: Caminho para o vídeo
            top_k: Número de categorias a retornar
            frame_interval: Intervalo entre frames (se None, usa frame_sample_rate)
            
        Returns:
            Dicionário com resultados da classificação
        """
        # Carregar o modelo se necessário
        if self.video_classifier is None:
            self.load_video_classifier()
            
        start_time = time.time()
        
        try:
            interval = frame_interval or self.frame_sample_rate
            logger.info(f"Classificando vídeo: {video_path}, intervalo: {interval}")
            
            # Carregar frames do vídeo
            frames, metadata = load_video(
                video_path,
                max_frames=None,
                frame_interval=interval
            )
            
            # Verificar se há frames suficientes
            if len(frames) < 8:  # Número mínimo arbitrário
                logger.warning(f"Poucos frames para classificação confiável: {len(frames)}")
                
            # Converter para formato esperado pelo modelo
            pixel_values = self.video_classifier_processor(
                [Image.fromarray(frame) for frame in frames],
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Realizar classificação
            with torch.no_grad():
                outputs = self.video_classifier(pixel_values)
                
            # Obter probabilidades
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Obter top_k classes
            values, indices = torch.topk(probs, top_k)
            
            # Mapear para nomes de classes
            labels = self.video_classifier.config.id2label
            
            results = [
                {
                    "label": labels[idx.item()],
                    "score": score.item()
                }
                for score, idx in zip(values[0], indices[0])
            ]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Vídeo classificado em {elapsed_time:.2f} segundos")
            
            return {
                "classifications": results,
                "duration": metadata["duration"],
                "frames_analyzed": len(frames),
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Erro ao classificar vídeo: {str(e)}")
            raise
            
    def detect_objects_in_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.3,
        frame_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detecta objetos em um vídeo.
        
        Args:
            video_path: Caminho para o vídeo
            confidence_threshold: Limiar de confiança para detecções
            frame_interval: Intervalo entre frames (se None, usa frame_sample_rate)
            
        Returns:
            Dicionário com resultados da detecção
        """
        # Carregar o modelo se necessário
        if self.object_detector is None:
            self.load_object_detector()
            
        start_time = time.time()
        
        try:
            interval = frame_interval or self.frame_sample_rate
            logger.info(f"Detectando objetos em vídeo: {video_path}, intervalo: {interval}")
            
            # Carregar frames do vídeo
            frames, metadata = load_video(
                video_path,
                max_frames=None,
                frame_interval=interval
            )
            
            # Resultado por frame
            frame_results = []
            
            # Contagem global de objetos
            object_counts = {}
            
            # Processar cada frame
            for i, frame in enumerate(frames):
                # Converter para formato PIL
                pil_image = Image.fromarray(frame)
                
                # Detectar objetos
                detections = self.object_detector(pil_image, threshold=confidence_threshold)
                
                # Filtrar resultados por confiança
                filtered_detections = [
                    d for d in detections 
                    if d["score"] >= confidence_threshold
                ]
                
                # Adicionar timestamp
                frame_time = i * interval / metadata["fps"]
                
                # Adicionar ao resultado
                frame_results.append({
                    "frame_index": i,
                    "time": frame_time,
                    "detections": filtered_detections
                })
                
                # Atualizar contagem global
                for d in filtered_detections:
                    label = d["label"]
                    if label in object_counts:
                        object_counts[label] += 1
                    else:
                        object_counts[label] = 1
                        
            # Ordenar contagem por frequência
            sorted_counts = sorted(
                object_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Converter para dicionário
            object_summary = [
                {"label": label, "count": count}
                for label, count in sorted_counts
            ]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Objetos detectados em {elapsed_time:.2f} segundos")
            
            return {
                "frame_results": frame_results,
                "object_summary": object_summary,
                "total_objects_detected": sum(object_counts.values()),
                "unique_objects": len(object_counts),
                "duration": metadata["duration"],
                "frames_analyzed": len(frames),
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Erro ao detectar objetos em vídeo: {str(e)}")
            raise
            
    def generate_video_caption(
        self,
        video_path: str,
        num_keyframes: int = 5,
        frame_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Gera uma descrição textual para um vídeo.
        
        Args:
            video_path: Caminho para o vídeo
            num_keyframes: Número de keyframes a utilizar
            frame_interval: Intervalo entre frames (se None, usa frame_sample_rate)
            
        Returns:
            Dicionário com a descrição gerada
        """
        # Carregar o modelo se necessário
        if self.video_captioner is None:
            self.load_video_captioner()
            
        start_time = time.time()
        
        try:
            interval = frame_interval or self.frame_sample_rate
            logger.info(f"Gerando legenda para vídeo: {video_path}")
            
            # Carregar frames do vídeo
            frames, metadata = load_video(
                video_path,
                max_frames=None,
                frame_interval=interval
            )
            
            # Selecionar keyframes
            if len(frames) <= num_keyframes:
                keyframes = frames
            else:
                # Distribuir uniformemente
                indices = np.linspace(0, len(frames) - 1, num_keyframes, dtype=int)
                keyframes = [frames[i] for i in indices]
                
            # Gerar legenda para cada keyframe
            keyframe_captions = []
            
            for i, frame in enumerate(keyframes):
                # Converter para formato PIL
                pil_image = Image.fromarray(frame)
                
                # Gerar legenda
                caption = self.video_captioner(pil_image)[0]["generated_text"]
                
                # Calcular timestamp
                if len(frames) <= num_keyframes:
                    frame_idx = i
                else:
                    frame_idx = indices[i]
                    
                frame_time = frame_idx * interval / metadata["fps"]
                
                keyframe_captions.append({
                    "time": frame_time,
                    "caption": caption
                })
                
            # Agregar legendas em uma descrição geral
            unique_phrases = set()
            for item in keyframe_captions:
                # Extrair frases únicas para evitar repetição
                unique_phrases.add(item["caption"].lower())
                
            # Criar descrição geral
            overall_description = ". ".join(unique_phrases)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Legenda gerada em {elapsed_time:.2f} segundos")
            
            return {
                "overall_description": overall_description,
                "keyframe_captions": keyframe_captions,
                "duration": metadata["duration"],
                "keyframes_analyzed": len(keyframes),
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar legenda para vídeo: {str(e)}")
            raise
            
    def analyze_video_content(
        self,
        video_path: str,
        analyze_objects: bool = True,
        classify_content: bool = True,
        generate_caption: bool = True,
        frame_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Realiza uma análise completa do conteúdo de um vídeo.
        
        Args:
            video_path: Caminho para o vídeo
            analyze_objects: Se True, detecta objetos
            classify_content: Se True, classifica o conteúdo
            generate_caption: Se True, gera descrição
            frame_interval: Intervalo entre frames
            
        Returns:
            Dicionário com resultados da análise
        """
        start_time = time.time()
        
        try:
            logger.info(f"Analisando conteúdo do vídeo: {video_path}")
            
            # Carregar frames do vídeo (apenas uma vez)
            interval = frame_interval or self.frame_sample_rate
            frames, metadata = load_video(
                video_path,
                max_frames=None,
                frame_interval=interval
            )
            
            results = {
                "video_info": {
                    "duration": metadata["duration"],
                    "fps": metadata["fps"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "total_frames": metadata["total_frames"]
                }
            }
            
            # Classificar conteúdo
            if classify_content:
                logger.info("Classificando conteúdo do vídeo...")
                
                if self.video_classifier is None:
                    self.load_video_classifier()
                    
                # Converter frames para formato esperado pelo modelo
                pixel_values = self.video_classifier_processor(
                    [Image.fromarray(frame) for frame in frames],
                    return_tensors="pt"
                ).pixel_values.to(self.device)
                
                # Realizar classificação
                with torch.no_grad():
                    outputs = self.video_classifier(pixel_values)
                    
                # Obter probabilidades
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                
                # Obter top-5 classes
                values, indices = torch.topk(probs, 5)
                
                # Mapear para nomes de classes
                labels = self.video_classifier.config.id2label
                
                classifications = [
                    {
                        "label": labels[idx.item()],
                        "score": score.item()
                    }
                    for score, idx in zip(values[0], indices[0])
                ]
                
                results["content_classification"] = classifications
                
            # Detectar objetos
            if analyze_objects:
                logger.info("Detectando objetos no vídeo...")
                
                if self.object_detector is None:
                    self.load_object_detector()
                    
                # Selecionar um subconjunto de frames para detecção
                # (usar todos pode ser muito lento)
                if len(frames) > 10:
                    indices = np.linspace(0, len(frames) - 1, 10, dtype=int)
                    object_frames = [frames[i] for i in indices]
                else:
                    object_frames = frames
                    
                # Contagem global de objetos
                object_counts = {}
                
                # Processar cada frame
                for frame in object_frames:
                    # Converter para formato PIL
                    pil_image = Image.fromarray(frame)
                    
                    # Detectar objetos
                    detections = self.object_detector(pil_image, threshold=0.3)
                    
                    # Atualizar contagem global
                    for d in detections:
                        label = d["label"]
                        if label in object_counts:
                            object_counts[label] += 1
                        else:
                            object_counts[label] = 1
                            
                # Ordenar contagem por frequência
                sorted_counts = sorted(
                    object_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Converter para dicionário
                object_summary = [
                    {"label": label, "count": count}
                    for label, count in sorted_counts
                ]
                
                results["object_detection"] = {
                    "object_summary": object_summary,
                    "total_objects_detected": sum(object_counts.values()),
                    "unique_objects": len(object_counts)
                }
                
            # Gerar descrição
            if generate_caption:
                logger.info("Gerando descrição para o vídeo...")
                
                if self.video_captioner is None:
                    self.load_video_captioner()
                    
                # Selecionar keyframes
                if len(frames) <= 5:
                    keyframes = frames
                else:
                    # Distribuir uniformemente
                    indices = np.linspace(0, len(frames) - 1, 5, dtype=int)
                    keyframes = [frames[i] for i in indices]
                    
                # Gerar legenda para cada keyframe
                captions = []
                
                for frame in keyframes:
                    # Converter para formato PIL
                    pil_image = Image.fromarray(frame)
                    
                    # Gerar legenda
                    caption = self.video_captioner(pil_image)[0]["generated_text"]
                    captions.append(caption)
                    
                # Agregar legendas em uma descrição geral
                unique_phrases = set()
                for caption in captions:
                    # Extrair frases únicas para evitar repetição
                    unique_phrases.add(caption.lower())
                    
                # Criar descrição geral
                overall_description = ". ".join(unique_phrases)
                
                results["video_description"] = overall_description
                
            elapsed_time = time.time() - start_time
            logger.info(f"Análise de vídeo concluída em {elapsed_time:.2f} segundos")
            
            results["processing_time"] = elapsed_time
            
            return results
            
        except Exception as e:
            logger.error(f"Erro ao analisar conteúdo do vídeo: {str(e)}")
            raise
