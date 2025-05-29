"""
Retrieval - Recuperação de informações relevantes com base no contexto.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import json

from storage.repositories import UserRepository, ResultRepository, InteractionRepository
from memory.user_context import UserContextManager

logger = logging.getLogger(__name__)

class ContextRetriever:
    """Recupera informações relevantes com base no contexto atual."""
    
    @staticmethod
    def retrieve_relevant_results(
        user_id: int,
        query: str,
        result_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recupera resultados relevantes com base em uma consulta.
        
        Args:
            user_id: ID do usuário
            query: Texto da consulta
            result_type: Filtrar por tipo de resultado (opcional)
            limit: Limite de resultados
            
        Returns:
            Lista de resultados relevantes
        """
        # Este método pode ser melhorado com embeddings e busca semântica
        # Por enquanto, usamos uma busca simples por palavras-chave
        
        # Buscar resultados
        results, _ = ResultRepository.search_results(
            user_id=user_id,
            result_type=result_type,
            query=query,
            include_public=True,
            limit=limit
        )
        
        return results
        
    @staticmethod
    def retrieve_chat_history(
        user_id: int,
        chat_id: str,
        max_messages: int = 10,
        include_system: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Recupera o histórico de mensagens de um chat.
        
        Args:
            user_id: ID do usuário
            chat_id: ID do chat
            max_messages: Número máximo de mensagens
            include_system: Incluir mensagens do sistema
            
        Returns:
            Lista de mensagens
        """
        # Obter contexto do chat
        context = UserContextManager.get_chat_context(user_id, chat_id)
        
        if not context or "messages" not in context:
            return []
            
        # Filtrar mensagens
        messages = context["messages"]
        
        if not include_system:
            messages = [msg for msg in messages if msg.get("role") != "system"]
            
        # Retornar as últimas N mensagens
        return messages[-max_messages:] if messages else []
        
    @staticmethod
    def create_system_message_with_context(
        user_id: int,
        include_preferences: bool = True,
        include_history: bool = True,
        include_profile: bool = True
    ) -> Dict[str, str]:
        """
        Cria uma mensagem de sistema rica em contexto.
        
        Args:
            user_id: ID do usuário
            include_preferences: Incluir preferências do usuário
            include_history: Incluir resumo do histórico
            include_profile: Incluir perfil do usuário
            
        Returns:
            Mensagem de sistema com contexto
        """
        context_parts = ["You are a helpful assistant tailored to this user's needs."]
        
        # Adicionar preferências
        if include_preferences:
            prefs = UserContextManager.get_user_preferences(user_id)
            
            if prefs:
                context_parts.append("User preferences:")
                
                # Adicionar preferências de estilo
                if "style_preferences" in prefs:
                    style_prefs = prefs["style_preferences"]
                    style_text = []
                    
                    for key, value in style_prefs.items():
                        style_text.append(f"- {key}: {value}")
                        
                    if style_text:
                        context_parts.append("Style preferences:\n" + "\n".join(style_text))
                        
                # Adicionar interesses
                if "interests" in prefs and "topics" in prefs["interests"]:
                    topics = prefs["interests"]["topics"]
                    if topics:
                        context_parts.append(f"User is interested in: {', '.join(topics)}")
                        
        # Adicionar resumo do histórico
        if include_history:
            # Obter contextos recentes
            contexts = UserContextRepository.list_contexts(user_id, limit=3)
            
            if contexts:
                recent_context = []
                
                for ctx in contexts:
                    if ctx["context_type"] == "chat" and "metadata" in ctx:
                        metadata = ctx["metadata"]
                        if "topic" in metadata and metadata["topic"]:
                            recent_context.append(f"- Recent conversation about: {metadata['topic']}")
                            
                if recent_context:
                    context_parts.append("Recent interactions:\n" + "\n".join(recent_context))
                    
        # Adicionar perfil
        if include_profile:
            # Simplificado - em uma implementação real, seria mais detalhado
            profile_parts = []
            
            # Obter feedback e comportamento do usuário
            feedback_stats = PreferenceRepository.get_preference(
                user_id=user_id,
                category="chat_feedback",
                key="statistics"
            )
            
            if feedback_stats:
                helpful_ratio = feedback_stats.get("helpful", 0) / max(
                    sum(feedback_stats.values()), 1
                )
                
                if helpful_ratio > 0.8:
                    profile_parts.append("User typically finds detailed and thorough responses helpful.")
                elif helpful_ratio < 0.5:
                    profile_parts.append("User may prefer more concise, direct responses.")
                    
            if profile_parts:
                context_parts.append("User behavior insights:\n" + "\n".join(profile_parts))
                
        # Juntar tudo
        system_message = {
            "role": "system",
            "content": "\n\n".join(context_parts)
        }
        
        return system_message
        
    @staticmethod
    def get_relevant_context_for_prompt(
        user_id: int,
        query: str,
        max_results: int = 3
    ) -> str:
        """
        Obtém contexto relevante para aumentar um prompt.
        
        Args:
            user_id: ID do usuário
            query: Consulta ou prompt atual
            max_results: Número máximo de resultados a incluir
            
        Returns:
            Texto com contexto relevante
        """
        # Buscar resultados relevantes
        results = ContextRetriever.retrieve_relevant_results(
            user_id=user_id,
            query=query,
            limit=max_results
        )
        
        if not results:
            return ""
            
        # Construir contexto
        context_parts = ["Here is some relevant information that might help:"]
        
        for result in results:
            result_id = result.get("result_id")
            result_type = result.get("result_type")
            
            # Obter conteúdo completo
            full_result = ResultRepository.get_result(result_id)
            
            if not full_result or "content" not in full_result:
                continue
                
            content = full_result["content"]
            
            # Adicionar de acordo com o tipo
            if result_type == "text":
                # Limitar tamanho para evitar contexto muito grande
                if len(content) > 500:
                    content = content[:500] + "..."
                    
                context_parts.append(f"Previous information:\n{content}")
                
            elif result_type in ["image", "audio", "video"]:
                # Para mídia, incluir apenas metadados ou descrições
                metadata = full_result.get("metadata", {})
                description = metadata.get("description", "No description available.")
                
                context_parts.append(f"Related {result_type}: {description}")
                
        # Juntar tudo
        if len(context_parts) > 1:  # Se tiver mais que apenas o cabeçalho
            return "\n\n".join(context_parts)
        else:
            return ""
