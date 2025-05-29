"""
Workflow - Definições e modelos para fluxos de trabalho multimodais.
"""

import os
import time
import logging
import json
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum
import tempfile

from orchestrator.core import MediaType, OrchestratorCore

logger = logging.getLogger(__name__)

class WorkflowTemplate:
    """Base para templates de fluxos de trabalho pré-definidos."""
    
    def __init__(self, orchestrator: OrchestratorCore):
        """
        Inicializa o template.
        
        Args:
            orchestrator: Instância do orquestrador
        """
        self.orchestrator = orchestrator
        
    def create(self, **kwargs) -> str:
        """
        Cria um fluxo de trabalho baseado no template.
        
        Args:
            **kwargs: Parâmetros específicos para o template
            
        Returns:
            ID do fluxo de trabalho criado
        """
        raise NotImplementedError("Subclasses devem implementar create()")
        
class VideoTranscriptionWorkflow(WorkflowTemplate):
    """
    Template para fluxo de trabalho de transcrição de vídeo.
    Fluxo: Vídeo -> Áudio -> Texto
    """
    
    def create(
        self,
        name: str = "Transcrição de Vídeo",
        description: Optional[str] = None,
        language: Optional[str] = None,
        summarize: bool = False,
        **kwargs
    ) -> str:
        """
        Cria um fluxo de trabalho de transcrição de vídeo.
        
        Args:
            name: Nome do fluxo de trabalho
            description: Descrição do fluxo
            language: Código do idioma para transcrição
            summarize: Se True, adiciona uma etapa de resumo
            **kwargs: Parâmetros adicionais
            
        Returns:
            ID do fluxo de trabalho
        """
        # Criar fluxo de trabalho
        workflow_id = self.orchestrator.create_workflow(
            name=name,
            description=description or "Extração de áudio de vídeo, seguida de transcrição de fala para texto",
            metadata={"type": "video_transcription", "language": language}
        )
        
        # Adicionar operações
        
        # 1. Extrair áudio do vídeo
        self.orchestrator.add_operation(
            workflow_id=workflow_id,
            media_type=MediaType.VIDEO,
            operation="to_audio",
            name="Extração de Áudio",
            parameters={
                "conversion_params": {
                    "output_format": "wav"
                }
            }
        )
        
        # 2. Transcrever áudio para texto
        transcription_op = self.orchestrator.add_operation(
            workflow_id=workflow_id,
            media_type=MediaType.AUDIO,
            operation="to_text",
            input_from="op_1",
            name="Transcrição",
            parameters={
                "conversion_params": {
                    "language": language,
                    "task": "transcribe"
                }
            }
        )
        
        # 3. Opcional: Resumir o texto transcrito
        if summarize:
            self.orchestrator.add_operation(
                workflow_id=workflow_id,
                media_type=MediaType.TEXT,
                operation="summarize",
                input_from=transcription_op,
                name="Resumo",
                parameters={
                    "operation_params": {
                        "max_length": kwargs.get("max_summary_length", 200),
                        "min_length": kwargs.get("min_summary_length", 50)
                    }
                }
            )
            
        return workflow_id
        
class ImageCaptioningWorkflow(WorkflowTemplate):
    """
    Template para fluxo de trabalho de legendagem de imagem.
    Fluxo: Imagem -> Texto
    """
    
    def create(
        self,
        name: str = "Legendagem de Imagem",
        description: Optional[str] = None,
        mode: str = "caption",
        translate_to: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Cria um fluxo de trabalho de legendagem de imagem.
        
        Args:
            name: Nome do fluxo de trabalho
            description: Descrição do fluxo
            mode: Modo de extração (caption ou ocr)
            translate_to: Código do idioma para tradução (opcional)
            **kwargs: Parâmetros adicionais
            
        Returns:
            ID do fluxo de trabalho
        """
        # Criar fluxo de trabalho
        workflow_id = self.orchestrator.create_workflow(
            name=name,
            description=description or f"Geração de descrição textual para imagem usando modo {mode}",
            metadata={"type": "image_captioning", "mode": mode}
        )
        
        # Adicionar operações
        
        # 1. Extrair texto da imagem
        caption_op = self.orchestrator.add_operation(
            workflow_id=workflow_id,
            media_type=MediaType.IMAGE,
            operation="to_text",
            name="Legendagem",
            parameters={
                "conversion_params": {
                    "mode": mode
                }
            }
        )
        
        # 2. Opcional: Traduzir o texto
        if translate_to:
            self.orchestrator.add_operation(
                workflow_id=workflow_id,
                media_type=MediaType.TEXT,
                operation="translate",
                input_from=caption_op,
                name="Tradução",
                parameters={
                    "operation_params": {
                        "target_language": translate_to
                    }
                }
            )
            
        return workflow_id
        
class TextToSpeechWorkflow(WorkflowTemplate):
    """
    Template para fluxo de trabalho de texto para fala.
    Fluxo: Texto -> Áudio
    """
    
    def create(
        self,
        name: str = "Texto para Fala",
        description: Optional[str] = None,
        voice_preset: str = "v2/en_speaker_6",
        translate_first: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Cria um fluxo de trabalho de texto para fala.
        
        Args:
            name: Nome do fluxo de trabalho
            description: Descrição do fluxo
            voice_preset: Preset de voz para síntese
            translate_first: Código do idioma para tradução prévia (opcional)
            **kwargs: Parâmetros adicionais
            
        Returns:
            ID do fluxo de trabalho
        """
        # Criar fluxo de trabalho
        workflow_id = self.orchestrator.create_workflow(
            name=name,
            description=description or "Conversão de texto para fala usando síntese de voz",
            metadata={"type": "text_to_speech", "voice": voice_preset}
        )
        
        # Adicionar operações
        
        last_op = None
        
        # 1. Opcional: Traduzir o texto primeiro
        if translate_first:
            last_op = self.orchestrator.add_operation(
                workflow_id=workflow_id,
                media_type=MediaType.TEXT,
                operation="translate",
                name="Tradução",
                parameters={
                    "operation_params": {
                        "target_language": translate_first
                    }
                }
            )
            
        # 2. Converter texto para fala
        self.orchestrator.add_operation(
            workflow_id=workflow_id,
            media_type=MediaType.TEXT,
            operation="to_audio",
            input_from=last_op,  # Pode ser None se não houver tradução
            name="Síntese de Fala",
            parameters={
                "conversion_params": {
                    "voice_preset": voice_preset,
                    "sampling_rate": kwargs.get("sampling_rate", 24000),
                    "output_format": kwargs.get("output_format", "wav")
                }
            }
        )
            
        return workflow_id
        
class VideoSummaryWorkflow(WorkflowTemplate):
    """
    Template para fluxo de trabalho de resumo de vídeo.
    Fluxo: Vídeo -> Texto (transcrição) -> Texto (resumo) -> Áudio (opcional)
    """
    
    def create(
        self,
        name: str = "Resumo de Vídeo",
        description: Optional[str] = None,
        language: Optional[str] = None,
        generate_audio: bool = False,
        voice_preset: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Cria um fluxo de trabalho de resumo de vídeo.
        
        Args:
            name: Nome do fluxo de trabalho
            description: Descrição do fluxo
            language: Código do idioma para transcrição
            generate_audio: Se True, converte o resumo em áudio
            voice_preset: Preset de voz para síntese (se generate_audio=True)
            **kwargs: Parâmetros adicionais
            
        Returns:
            ID do fluxo de trabalho
        """
        # Criar fluxo de trabalho
        workflow_id = self.orchestrator.create_workflow(
            name=name,
            description=description or "Transcrição de vídeo seguida de resumo textual",
            metadata={"type": "video_summary", "language": language}
        )
        
        # Adicionar operações
        
        # 1. Extrair áudio do vídeo
        self.orchestrator.add_operation(
            workflow_id=workflow_id,
            media_type=MediaType.VIDEO,
            operation="to_audio",
            name="Extração de Áudio",
            parameters={
                "conversion_params": {
                    "output_format": "wav"
                }
            }
        )
        
        # 2. Transcrever áudio para texto
        self.orchestrator.add_operation(
            workflow_id=workflow_id,
            media_type=MediaType.AUDIO,
            operation="to_text",
            input_from="op_1",
            name="Transcrição",
            parameters={
                "conversion_params": {
                    "language": language,
                    "task": "transcribe"
                }
            }
        )
        
        # 3. Resumir o texto transcrito
        summary_op = self.orchestrator.add_operation(
            workflow_id=workflow_id,
            media_type=MediaType.TEXT,
            operation="summarize",
            input_from="op_2",
            name="Resumo",
            parameters={
                "operation_params": {
                    "max_length": kwargs.get("max_summary_length", 300),
                    "min_length": kwargs.get("min_summary_length", 100)
                }
            }
        )
        
        # 4. Opcional: Converter resumo em áudio
        if generate_audio:
            self.orchestrator.add_operation(
                workflow_id=workflow_id,
                media_type=MediaType.TEXT,
                operation="to_audio",
                input_from=summary_op,
                name="Síntese de Fala do Resumo",
                parameters={
                    "conversion_params": {
                        "voice_preset": voice_preset or "v2/en_speaker_6",
                        "sampling_rate": kwargs.get("sampling_rate", 24000),
                        "output_format": kwargs.get("output_format", "wav")
                    }
                }
            )
            
        return workflow_id

class WorkflowRegistry:
    """
    Registro central de templates de fluxo de trabalho.
    """
    
    def __init__(self, orchestrator: OrchestratorCore):
        """
        Inicializa o registro.
        
        Args:
            orchestrator: Instância do orquestrador
        """
        self.orchestrator = orchestrator
        self.templates = {}
        
        # Registrar templates padrão
        self.register_default_templates()
        
    def register_template(self, name: str, template_class: type):
        """
        Registra um template de fluxo de trabalho.
        
        Args:
            name: Nome do template
            template_class: Classe do template
        """
        self.templates[name] = template_class(self.orchestrator)
        logger.info(f"Template de fluxo de trabalho registrado: {name}")
        
    def register_default_templates(self):
        """Registra os templates padrão."""
        self.register_template("video_transcription", VideoTranscriptionWorkflow)
        self.register_template("image_captioning", ImageCaptioningWorkflow)
        self.register_template("text_to_speech", TextToSpeechWorkflow)
        self.register_template("video_summary", VideoSummaryWorkflow)
        
    def get_template(self, name: str) -> WorkflowTemplate:
        """
        Obtém um template pelo nome.
        
        Args:
            name: Nome do template
            
        Returns:
            Template de fluxo de trabalho
        """
        if name not in self.templates:
            raise ValueError(f"Template não encontrado: {name}")
            
        return self.templates[name]
        
    def list_templates(self) -> List[str]:
        """
        Lista os templates disponíveis.
        
        Returns:
            Lista de nomes de templates
        """
        return list(self.templates.keys())
        
    def create_workflow_from_template(
        self,
        template_name: str,
        **kwargs
    ) -> str:
        """
        Cria um fluxo de trabalho a partir de um template.
        
        Args:
            template_name: Nome do template
            **kwargs: Parâmetros para o template
            
        Returns:
            ID do fluxo de trabalho criado
        """
        template = self.get_template(template_name)
        return template.create(**kwargs)
