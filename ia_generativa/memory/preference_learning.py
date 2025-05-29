"""
Preference Learning - Aprendizado de preferências do usuário.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import random

from storage.repositories import UserRepository, InteractionRepository, PreferenceRepository
from memory.user_context import UserContextManager

logger = logging.getLogger(__name__)

class PreferenceLearner:
    """Aprende e atualiza preferências do usuário com base em seu comportamento."""
    
    @staticmethod
    def update_from_interaction(
        user_id: int,
        interaction_type: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Atualiza preferências do usuário com base em uma interação.
        
        Args:
            user_id: ID do usuário
            interaction_type: Tipo de interação
            metadata: Metadados da interação
        """
        # Diferentes tipos de interação levam a diferentes atualizações de preferências
        if interaction_type == "generation_feedback":
            PreferenceLearner._process_generation_feedback(user_id, metadata)
        elif interaction_type == "content_view":
            PreferenceLearner._process_content_view(user_id, metadata)
        elif interaction_type == "style_selection":
            PreferenceLearner._process_style_selection(user_id, metadata)
        elif interaction_type == "chat_feedback":
            PreferenceLearner._process_chat_feedback(user_id, metadata)
            
    @staticmethod
    def _process_generation_feedback(user_id: int, metadata: Dict[str, Any]) -> None:
        """
        Processa feedback sobre uma geração para atualizar preferências.
        
        Args:
            user_id: ID do usuário
            metadata: Metadados do feedback
        """
        # Extrair dados relevantes
        rating = metadata.get("rating")
        result_type = metadata.get("result_type")
        model_used = metadata.get("model")
        parameters = metadata.get("parameters", {})
        
        if rating is None or result_type is None:
            return
            
        # Processar avaliação positiva
        if rating > 3:  # Em uma escala de 1-5
            # Incrementar contagem de preferência para este modelo
            PreferenceLearner._increment_model_preference(
                user_id, result_type, model_used
            )
            
            # Armazenar parâmetros favoritos
            if parameters and model_used:
                current_params = PreferenceRepository.get_preference(
                    user_id=user_id,
                    category=f"{result_type}_parameters",
                    key=model_used
                ) or {}
                
                # Atualizar com novos parâmetros bem avaliados
                for param_key, param_value in parameters.items():
                    if param_key in ["temperature", "top_p", "guidance_scale"]:
                        if param_key not in current_params:
                            current_params[param_key] = [param_value]
                        else:
                            current_params[param_key].append(param_value)
                            # Manter apenas os 5 valores mais recentes
                            current_params[param_key] = current_params[param_key][-5:]
                            
                PreferenceRepository.set_preference(
                    user_id=user_id,
                    category=f"{result_type}_parameters",
                    key=model_used,
                    value=current_params
                )
                
        # Processar avaliação negativa
        elif rating < 3:
            # Decrementar preferência por este modelo
            PreferenceLearner._decrement_model_preference(
                user_id, result_type, model_used
            )
            
    @staticmethod
    def _process_content_view(user_id: int, metadata: Dict[str, Any]) -> None:
        """
        Processa visualização de conteúdo para inferir interesses.
        
        Args:
            user_id: ID do usuário
            metadata: Metadados da visualização
        """
        content_type = metadata.get("content_type")
        tags = metadata.get("tags", [])
        duration = metadata.get("duration", 0)  # Tempo de visualização em segundos
        
        if not content_type or not tags:
            return
            
        # Se visualizou por tempo significativo, considerar como interesse
        if duration > 30:  # Mais de 30 segundos
            # Atualizar interesses do usuário
            current_interests = PreferenceRepository.get_preference(
                user_id=user_id,
                category="interests",
                key="topics"
            ) or []
            
            # Adicionar novos tags como interesses
            for tag in tags:
                if tag not in current_interests:
                    current_interests.append(tag)
                    
            PreferenceRepository.set_preference(
                user_id=user_id,
                category="interests",
                key="topics",
                value=current_interests
            )
            
    @staticmethod
    def _process_style_selection(user_id: int, metadata: Dict[str, Any]) -> None:
        """
        Processa seleção de estilo para aprendizado de preferências estéticas.
        
        Args:
            user_id: ID do usuário
            metadata: Metadados da seleção
        """
        style_type = metadata.get("style_type")  # ex: "image_style", "text_tone", etc.
        style_value = metadata.get("style_value")
        
        if not style_type or not style_value:
            return
            
        # Atualizar preferência de estilo
        PreferenceRepository.set_preference(
            user_id=user_id,
            category="style_preferences",
            key=style_type,
            value=style_value
        )
        
    @staticmethod
    def _process_chat_feedback(user_id: int, metadata: Dict[str, Any]) -> None:
        """
        Processa feedback de conversação para melhorar interações futuras.
        
        Args:
            user_id: ID do usuário
            metadata: Metadados do feedback
        """
        feedback_type = metadata.get("feedback_type")  # ex: "helpful", "not_helpful", "off_topic"
        chat_id = metadata.get("chat_id")
        message_id = metadata.get("message_id")
        
        if not feedback_type or not chat_id:
            return
            
        # Obter estatísticas atuais de feedback
        feedback_stats = PreferenceRepository.get_preference(
            user_id=user_id,
            category="chat_feedback",
            key="statistics"
        ) or {"helpful": 0, "not_helpful": 0, "off_topic": 0}
        
        # Atualizar contadores
        if feedback_type in feedback_stats:
            feedback_stats[feedback_type] += 1
        else:
            feedback_stats[feedback_type] = 1
            
        PreferenceRepository.set_preference(
            user_id=user_id,
            category="chat_feedback",
            key="statistics",
            value=feedback_stats
        )
        
        # Se feedback negativo, armazenar para análise e melhoria
        if feedback_type in ["not_helpful", "off_topic"]:
            negative_examples = PreferenceRepository.get_preference(
                user_id=user_id,
                category="chat_feedback",
                key="negative_examples"
            ) or []
            
            negative_examples.append({
                "timestamp": datetime.utcnow().isoformat(),
                "chat_id": chat_id,
                "message_id": message_id,
                "feedback_type": feedback_type
            })
            
            # Limitar a 20 exemplos mais recentes
            if len(negative_examples) > 20:
                negative_examples = negative_examples[-20:]
                
            PreferenceRepository.set_preference(
                user_id=user_id,
                category="chat_feedback",
                key="negative_examples",
                value=negative_examples
            )
            
    @staticmethod
    def _increment_model_preference(
        user_id: int,
        content_type: str,
        model_name: str
    ) -> None:
        """
        Incrementa a preferência por um modelo específico.
        
        Args:
            user_id: ID do usuário
            content_type: Tipo de conteúdo (image, text, audio, etc.)
            model_name: Nome do modelo
        """
        if not model_name:
            return
            
        preference_key = f"{content_type}_model_preference"
        current_prefs = PreferenceRepository.get_preference(
            user_id=user_id,
            category="models",
            key=preference_key
        ) or {}
        
        if model_name in current_prefs:
            current_prefs[model_name] += 1
        else:
            current_prefs[model_name] = 1
            
        PreferenceRepository.set_preference(
            user_id=user_id,
            category="models",
            key=preference_key,
            value=current_prefs
        )
        
    @staticmethod
    def _decrement_model_preference(
        user_id: int,
        content_type: str,
        model_name: str
    ) -> None:
        """
        Decrementa a preferência por um modelo específico.
        
        Args:
            user_id: ID do usuário
            content_type: Tipo de conteúdo (image, text, audio, etc.)
            model_name: Nome do modelo
        """
        if not model_name:
            return
            
        preference_key = f"{content_type}_model_preference"
        current_prefs = PreferenceRepository.get_preference(
            user_id=user_id,
            category="models",
            key=preference_key
        ) or {}
        
        if model_name in current_prefs:
            current_prefs[model_name] -= 1
            # Evitar valores negativos
            if current_prefs[model_name] < 0:
                current_prefs[model_name] = 0
        else:
            current_prefs[model_name] = 0
            
        PreferenceRepository.set_preference(
            user_id=user_id,
            category="models",
            key=preference_key,
            value=current_prefs
        )
        
    @staticmethod
    def get_preferred_model(
        user_id: int,
        content_type: str
    ) -> Optional[str]:
        """
        Obtém o modelo preferido do usuário para um tipo de conteúdo.
        
        Args:
            user_id: ID do usuário
            content_type: Tipo de conteúdo (image, text, audio, etc.)
            
        Returns:
            Nome do modelo preferido ou None
        """
        preference_key = f"{content_type}_model_preference"
        model_prefs = PreferenceRepository.get_preference(
            user_id=user_id,
            category="models",
            key=preference_key
        )
        
        if not model_prefs:
            return None
            
        # Encontrar o modelo com maior preferência
        max_pref = 0
        preferred_model = None
        
        for model, pref in model_prefs.items():
            if pref > max_pref:
                max_pref = pref
                preferred_model = model
                
        return preferred_model
        
    @staticmethod
    def get_optimal_parameters(
        user_id: int,
        content_type: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Obtém parâmetros otimizados com base em preferências anteriores.
        
        Args:
            user_id: ID do usuário
            content_type: Tipo de conteúdo (image, text, audio, etc.)
            model_name: Nome do modelo
            
        Returns:
            Dicionário com parâmetros otimizados
        """
        params = PreferenceRepository.get_preference(
            user_id=user_id,
            category=f"{content_type}_parameters",
            key=model_name
        ) or {}
        
        # Calcular valores médios ou mais frequentes
        optimal_params = {}
        
        for param_key, param_values in params.items():
            if not param_values:
                continue
                
            if param_key in ["temperature", "top_p", "guidance_scale"]:
                # Para parâmetros numéricos, usar média
                optimal_params[param_key] = sum(param_values) / len(param_values)
            else:
                # Para outros parâmetros, usar o mais frequente
                from collections import Counter
                counter = Counter(param_values)
                optimal_params[param_key] = counter.most_common(1)[0][0]
                
        return optimal_params
