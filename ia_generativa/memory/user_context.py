"""
User Context - Gerenciamento de contexto do usuário.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

from storage.repositories import UserRepository, UserContextRepository, InteractionRepository, PreferenceRepository

logger = logging.getLogger(__name__)

class UserContextManager:
    """Gerencia o contexto e memória do usuário."""
    
    @staticmethod
    def get_or_create_user(
        username: str,
        email: Optional[str] = None,
        external_id: Optional[str] = None
    ) -> int:
        """
        Obtém ou cria um usuário e retorna seu ID.
        
        Args:
            username: Nome do usuário
            email: Email (opcional)
            external_id: ID externo (opcional)
            
        Returns:
            ID do usuário
        """
        user, created = UserRepository.get_or_create_user(
            username=username,
            email=email,
            external_id=external_id
        )
        
        if created:
            logger.info(f"Novo usuário criado: {username} (ID: {user.id})")
        
        return user.id
        
    @staticmethod
    def get_chat_context(
        user_id: int,
        chat_id: str
    ) -> Dict[str, Any]:
        """
        Obtém o contexto de um chat.
        
        Args:
            user_id: ID do usuário
            chat_id: ID do chat
            
        Returns:
            Contexto do chat (incluindo histórico)
        """
        # Buscar contexto existente
        context = UserContextRepository.get_context(
            user_id=user_id,
            context_type="chat",
            context_id=chat_id
        )
        
        if context:
            return context["state"]
            
        # Criar novo contexto vazio
        default_context = {
            "messages": [],
            "summary": None,
            "topic": None,
            "created_at": datetime.utcnow().isoformat(),
            "last_message_at": None,
            "message_count": 0
        }
        
        UserContextRepository.set_context(
            user_id=user_id,
            context_type="chat",
            context_id=chat_id,
            state=default_context
        )
        
        return default_context
        
    @staticmethod
    def update_chat_context(
        user_id: int,
        chat_id: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adiciona uma mensagem ao contexto do chat.
        
        Args:
            user_id: ID do usuário
            chat_id: ID do chat
            message: Mensagem a ser adicionada
            
        Returns:
            Contexto atualizado
        """
        # Obter contexto atual
        context = UserContextManager.get_chat_context(user_id, chat_id)
        
        # Adicionar timestamp se não tiver
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
            
        # Adicionar mensagem
        context["messages"].append(message)
        context["last_message_at"] = message["timestamp"]
        context["message_count"] += 1
        
        # Atualizar contexto
        UserContextRepository.set_context(
            user_id=user_id,
            context_type="chat",
            context_id=chat_id,
            state=context
        )
        
        # Registrar interação
        InteractionRepository.record_interaction(
            user_id=user_id,
            interaction_type="chat_message",
            metadata={
                "chat_id": chat_id,
                "message_role": message.get("role", "unknown"),
                "length": len(message.get("content", ""))
            }
        )
        
        return context
        
    @staticmethod
    def summarize_chat(
        user_id: int,
        chat_id: str,
        max_messages: int = 10
    ) -> Optional[str]:
        """
        Gera um resumo do chat para contexto mais eficiente.
        
        Args:
            user_id: ID do usuário
            chat_id: ID do chat
            max_messages: Número máximo de mensagens a manter no contexto
            
        Returns:
            Resumo do chat ou None se não for necessário
        """
        # Obter contexto atual
        context = UserContextManager.get_chat_context(user_id, chat_id)
        
        # Verificar se é necessário resumir
        if len(context["messages"]) <= max_messages:
            return None
            
        # Mensagens para resumir (todas exceto as últimas max_messages)
        messages_to_summarize = context["messages"][:-max_messages]
        
        # Gerar resumo (aqui você pode usar o LLM para resumir)
        # Para este exemplo, usaremos uma abordagem simplificada
        from models.text.llm_manager import LLMManager
        llm = LLMManager()
        
        # Construir prompt para resumo
        prompt = "Resumo da conversa anterior:\n\n"
        for msg in messages_to_summarize:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n\n"
            
        prompt += "Por favor, faça um resumo conciso desta conversa, destacando os pontos principais discutidos."
        
        # Gerar resumo
        summary = llm.generate_text(
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.7
        )
        
        # Atualizar contexto
        context["summary"] = summary
        context["messages"] = context["messages"][-max_messages:]  # Manter apenas as últimas mensagens
        
        UserContextRepository.set_context(
            user_id=user_id,
            context_type="chat",
            context_id=chat_id,
            state=context
        )
        
        return summary
        
    @staticmethod
    def get_workflow_context(
        user_id: int,
        workflow_id: str
    ) -> Dict[str, Any]:
        """
        Obtém o contexto de um fluxo de trabalho.
        
        Args:
            user_id: ID do usuário
            workflow_id: ID do fluxo de trabalho
            
        Returns:
            Contexto do fluxo de trabalho
        """
        # Buscar contexto existente
        context = UserContextRepository.get_context(
            user_id=user_id,
            context_type="workflow",
            context_id=workflow_id
        )
        
        if context:
            return context["state"]
            
        # Criar novo contexto vazio
        default_context = {
            "operations": [],
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "last_updated_at": datetime.utcnow().isoformat(),
            "results": {}
        }
        
        UserContextRepository.set_context(
            user_id=user_id,
            context_type="workflow",
            context_id=workflow_id,
            state=default_context
        )
        
        return default_context
        
    @staticmethod
    def update_workflow_context(
        user_id: int,
        workflow_id: str,
        status: str,
        operation_id: Optional[str] = None,
        operation_status: Optional[str] = None,
        result: Optional[Any] = None
    ) -> Dict[str, Any]:
                """
        Atualiza o contexto de um fluxo de trabalho.
        
        Args:
            user_id: ID do usuário
            workflow_id: ID do fluxo de trabalho
            status: Status do fluxo de trabalho
            operation_id: ID da operação (opcional)
            operation_status: Status da operação (opcional)
            result: Resultado da operação (opcional)
            
        Returns:
            Contexto atualizado
        """
        # Obter contexto atual
        context = UserContextManager.get_workflow_context(user_id, workflow_id)
        
        # Atualizar status do fluxo
        context["status"] = status
        context["last_updated_at"] = datetime.utcnow().isoformat()
        
        # Atualizar operação específica, se fornecida
        if operation_id and operation_status:
            # Verificar se a operação já existe no contexto
            op_exists = False
            for op in context["operations"]:
                if op["id"] == operation_id:
                    op["status"] = operation_status
                    op["updated_at"] = datetime.utcnow().isoformat()
                    op_exists = True
                    break
                    
            # Se não existe, adicionar
            if not op_exists:
                context["operations"].append({
                    "id": operation_id,
                    "status": operation_status,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                })
                
        # Adicionar resultado, se fornecido
        if operation_id and result is not None:
            context["results"][operation_id] = {
                "timestamp": datetime.utcnow().isoformat(),
                "summary": str(result)[:100] + "..." if isinstance(result, str) and len(str(result)) > 100 else str(result)
            }
            
        # Atualizar contexto
        UserContextRepository.set_context(
            user_id=user_id,
            context_type="workflow",
            context_id=workflow_id,
            state=context
        )
        
        return context
        
    @staticmethod
    def get_user_preferences(
        user_id: int,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtém as preferências de um usuário.
        
        Args:
            user_id: ID do usuário
            category: Categoria específica (opcional)
            
        Returns:
            Dicionário de preferências
        """
        if category:
            return PreferenceRepository.get_preferences_by_category(user_id, category)
        else:
            # Buscar todas as categorias
            preferences = {}
            
            # Lista de categorias comuns
            categories = ["appearance", "notifications", "privacy", "language", "models", "generation"]
            
            for cat in categories:
                prefs = PreferenceRepository.get_preferences_by_category(user_id, cat)
                if prefs:
                    preferences[cat] = prefs
                    
            return preferences
            
    @staticmethod
    def get_recent_interactions(
        user_id: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtém as interações recentes de um usuário.
        
        Args:
            user_id: ID do usuário
            limit: Limite de resultados
            
        Returns:
            Lista de interações recentes
        """
        return InteractionRepository.get_user_interactions(
            user_id=user_id,
            limit=limit,
            offset=0
        )
        
    @staticmethod
    def build_user_profile(user_id: int) -> Dict[str, Any]:
        """
        Constrói um perfil completo do usuário para personalização.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Perfil do usuário
        """
        # Obter dados do usuário
        user = UserRepository.get_user_by_id(user_id)
        if not user:
            return {}
            
        # Obter preferências
        preferences = UserContextManager.get_user_preferences(user_id)
        
        # Obter contextos recentes
        contexts = UserContextRepository.list_contexts(user_id, limit=5)
        
        # Obter interações recentes
        interactions = UserContextManager.get_recent_interactions(user_id, limit=20)
        
        # Construir perfil
        profile = {
            "user_id": user_id,
            "username": user.username,
            "preferences": preferences,
            "recent_contexts": contexts,
            "interaction_summary": _summarize_interactions(interactions),
            "created_at": user.created_at.isoformat(),
            "profile_built_at": datetime.utcnow().isoformat()
        }
        
        return profile
        
def _summarize_interactions(interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sumariza interações do usuário para análise de padrões.
    
    Args:
        interactions: Lista de interações
        
    Returns:
        Resumo das interações
    """
    summary = {
        "total": len(interactions),
        "types": {},
        "recent_activity": []
    }
    
    # Contar por tipo
    for interaction in interactions:
        interaction_type = interaction.get("interaction_type", "unknown")
        
        if interaction_type in summary["types"]:
            summary["types"][interaction_type] += 1
        else:
            summary["types"][interaction_type] = 1
            
        # Adicionar às atividades recentes (simplificado)
        summary["recent_activity"].append({
            "type": interaction_type,
            "timestamp": interaction.get("created_at")
        })
        
    return summary
