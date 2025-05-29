"""
Text Tokenizer - Funções para tokenização de texto.
"""

from typing import List, Dict, Any, Optional, Union
import re
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class TextTokenizer:
    """
    Classe para tokenização de texto utilizando diferentes métodos.
    """
    
    def __init__(
        self, 
        tokenizer_name: str = "meta-llama/Llama-3-8B-Instruct",
        use_fast: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Inicializa o tokenizador.
        
        Args:
            tokenizer_name: Nome do tokenizador no HuggingFace ou caminho local
            use_fast: Se True, usa a versão rápida do tokenizador (implementação Rust)
            cache_dir: Diretório para cache de modelos
        """
        self.tokenizer_name = tokenizer_name
        self.use_fast = use_fast
        self.cache_dir = cache_dir
        
        # O tokenizador é carregado sob demanda
        self._tokenizer = None
        
    @property
    def tokenizer(self):
        """Carrega o tokenizador caso ainda não tenha sido carregado."""
        if self._tokenizer is None:
            try:
                logger.info(f"Carregando tokenizador: {self.tokenizer_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name,
                    use_fast=self.use_fast,
                    cache_dir=self.cache_dir
                )
                logger.info("Tokenizador carregado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao carregar tokenizador: {str(e)}")
                raise
        return self._tokenizer
    
    def tokenize(
        self, 
        text: str,
        add_special_tokens: bool = False,
        return_tokens: bool = True,
        return_ids: bool = True,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tokeniza o texto usando o tokenizador carregado.
        
        Args:
            text: Texto a ser tokenizado
            add_special_tokens: Se True, adiciona tokens especiais como [CLS], [SEP]
            return_tokens: Se True, inclui a lista de tokens no resultado
            return_ids: Se True, inclui os IDs dos tokens no resultado
            return_attention_mask: Se True, inclui máscara de atenção no resultado
            return_token_type_ids: Se True, inclui IDs de tipo de token no resultado
            **kwargs: Argumentos adicionais para o tokenizador
            
        Returns:
            Dicionário com os resultados da tokenização
        """
        # Tokenizar o texto
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            **kwargs
        )
        
        result = {}
        
        # Incluir os IDs dos tokens
        if return_ids:
            result["input_ids"] = encoded["input_ids"]
            
        # Incluir a máscara de atenção
        if return_attention_mask and "attention_mask" in encoded:
            result["attention_mask"] = encoded["attention_mask"]
            
        # Incluir os IDs de tipo de token
        if return_token_type_ids and "token_type_ids" in encoded:
            result["token_type_ids"] = encoded["token_type_ids"]
            
        # Converter IDs para tokens se solicitado
        if return_tokens:
            if isinstance(encoded["input_ids"], list):
                result["tokens"] = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
            else:
                # Para tensores PyTorch ou arrays NumPy
                result["tokens"] = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"].tolist())
                
        # Adicionar contagem de tokens
        result["token_count"] = len(result.get("input_ids", []))
        
        return result
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estima a contagem de tokens para um texto.
        Útil para verificar se um texto não excede limites de contexto.
        
        Args:
            text: Texto para estimar a contagem de tokens
            
        Returns:
            Número estimado de tokens
        """
        return len(self.tokenizer.encode(text))
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """
        Converte IDs de tokens de volta para texto.
        
        Args:
            token_ids: Lista de IDs de tokens
            **kwargs: Argumentos adicionais para o decodificador
            
        Returns:
            Texto decodificado
        """
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def get_vocabulary_size(self) -> int:
        """Retorna o tamanho do vocabulário do tokenizador."""
        return len(self.tokenizer)
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o tokenizador."""
        return {
            "name": self.tokenizer_name,
            "vocabulary_size": self.get_vocabulary_size(),
            "model_max_length": getattr(self.tokenizer, "model_max_length", None),
            "is_fast": getattr(self.tokenizer, "is_fast", None),
            "special_tokens": {
                "pad_token": str(self.tokenizer.pad_token),
                "eos_token": str(self.tokenizer.eos_token),
                "bos_token": str(getattr(self.tokenizer, "bos_token", None)),
                "unk_token": str(self.tokenizer.unk_token),
                "mask_token": str(getattr(self.tokenizer, "mask_token", None)),
                "sep_token": str(getattr(self.tokenizer, "sep_token", None)),
                "cls_token": str(getattr(self.tokenizer, "cls_token", None))
            }
        }
