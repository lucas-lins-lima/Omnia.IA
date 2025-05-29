"""
Prompt Templates - Modelos para estruturar prompts para diferentes modelos e tarefas.
"""

from typing import Dict, List, Optional, Union, Any
import json

class PromptTemplate:
    """Classe base para templates de prompt."""
    
    def format(self, **kwargs) -> str:
        """
        Formata o prompt com os parâmetros fornecidos.
        
        Args:
            **kwargs: Argumentos para preencher o template
            
        Returns:
            Prompt formatado
        """
        raise NotImplementedError("Subclasses devem implementar format()")
        
class Llama3PromptTemplate(PromptTemplate):
    """Template de prompt para modelos Llama 3."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Inicializa o template para Llama 3.
        
        Args:
            system_prompt: Instruções de sistema para o modelo
        """
        self.system_prompt = system_prompt or (
            "Você é Claude, um assistente de IA útil, inofensivo e honesto. Responda de forma "
            "concisa e precisa, sempre mantendo uma comunicação respeitosa e educada."
        )
    
    def format(
        self, 
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        include_system_prompt: bool = True
    ) -> str:
        """
        Formata o prompt para o formato esperado pelo Llama 3.
        
        Args:
            user_message: Mensagem do usuário
            chat_history: Histórico de conversas anteriores (opcional)
            include_system_prompt: Se True, inclui o prompt de sistema
            
        Returns:
            Prompt formatado
        """
        messages = []
        
        # Adicionar o prompt de sistema se solicitado
        if include_system_prompt:
            messages.append(f"<|system|>\n{self.system_prompt}\n</s>")
            
        # Adicionar histórico de chat, se fornecido
        if chat_history:
            for message in chat_history:
                role = message.get("role", "").lower()
                content = message.get("content", "")
                
                if role == "user":
                    messages.append(f"<|user|>\n{content}\n</s>")
                elif role == "assistant":
                    messages.append(f"<|assistant|>\n{content}\n</s>")
                    
        # Adicionar a mensagem atual do usuário
        messages.append(f"<|user|>\n{user_message}\n</s>")
        
        # Adicionar o início da resposta do assistente
        messages.append("<|assistant|>")
        
        # Juntar tudo
        return "\n".join(messages)
        
class MistralPromptTemplate(PromptTemplate):
    """Template de prompt para modelos Mistral."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Inicializa o template para Mistral.
        
        Args:
            system_prompt: Instruções de sistema para o modelo
        """
        self.system_prompt = system_prompt or (
            "Você é Claude, um assistente de IA útil, inofensivo e honesto. Responda de forma "
            "concisa e precisa, sempre mantendo uma comunicação respeitosa e educada."
        )
    
    def format(
        self, 
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        include_system_prompt: bool = True
    ) -> str:
        """
        Formata o prompt para o formato esperado pelo Mistral.
        
        Args:
            user_message: Mensagem do usuário
            chat_history: Histórico de conversas anteriores (opcional)
            include_system_prompt: Se True, inclui o prompt de sistema
            
        Returns:
            Prompt formatado
        """
        messages = []
        
        # Adicionar o prompt de sistema se solicitado
        if include_system_prompt:
            messages.append(f"<s>[INST] {self.system_prompt} [/INST]")
            
        # Adicionar histórico de chat, se fornecido
        if chat_history:
            for i, message in enumerate(chat_history):
                role = message.get("role", "").lower()
                content = message.get("content", "")
                
                if role == "user":
                    if i == 0 and include_system_prompt:
                        # Primeiro usuário após sistema
                        messages.append(f"\n\n[INST] {content} [/INST]")
                    else:
                        messages.append(f"<s>[INST] {content} [/INST]")
                elif role == "assistant":
                    messages.append(f" {content} </s>")
                    
        # Adicionar a mensagem atual do usuário
        if chat_history:
            messages.append(f"<s>[INST] {user_message} [/INST]")
        else:
            if include_system_prompt:
                messages.append(f"\n\n[INST] {user_message} [/INST]")
            else:
                messages.append(f"<s>[INST] {user_message} [/INST]")
        
        # Juntar tudo
        return "".join(messages)
        
class PromptManager:
    """Gerencia diferentes templates de prompt para vários modelos."""
    
    def __init__(self):
        """Inicializa o gerenciador de prompts."""
        self.templates = {
            "llama3": Llama3PromptTemplate(),
            "mistral": MistralPromptTemplate(),
            # Adicione mais templates conforme necessário
        }
        
    def get_template(self, model_type: str) -> PromptTemplate:
        """
        Retorna o template para um tipo de modelo específico.
        
        Args:
            model_type: Tipo do modelo (ex: "llama3", "mistral")
            
        Returns:
            Template de prompt para o modelo
        """
        model_type = model_type.lower()
        
        if model_type not in self.templates:
            raise ValueError(f"Template não encontrado para o modelo: {model_type}")
            
        return self.templates[model_type]
        
    def register_template(self, model_type: str, template: PromptTemplate):
        """
        Registra um novo template.
        
        Args:
            model_type: Tipo do modelo
            template: Template de prompt
        """
        self.templates[model_type.lower()] = template
        
    def format_prompt(
        self, 
        model_type: str,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Formata um prompt para um modelo específico.
        
        Args:
            model_type: Tipo do modelo
            user_message: Mensagem do usuário
            chat_history: Histórico da conversa
            system_prompt: Instruções de sistema (substituem o padrão)
            **kwargs: Argumentos adicionais para o template
            
        Returns:
            Prompt formatado
        """
        template = self.get_template(model_type)
        
        # Se um system_prompt foi fornecido, criar um novo template com ele
        if system_prompt is not None:
            if model_type.lower() == "llama3":
                template = Llama3PromptTemplate(system_prompt)
            elif model_type.lower() == "mistral":
                template = MistralPromptTemplate(system_prompt)
                
        return template.format(
            user_message=user_message,
            chat_history=chat_history,
            **kwargs
        )
