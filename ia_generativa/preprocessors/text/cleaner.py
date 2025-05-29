"""
Text Cleaner - Funções para limpeza e pré-processamento de texto.
"""

import re
import html
import unicodedata
from typing import List, Optional, Dict, Any

def remove_html_tags(text: str) -> str:
    """Remove tags HTML e entidades do texto."""
    # Primeiro, decodifica entidades HTML (&lt; -> <, &amp; -> &, etc.)
    text = html.unescape(text)
    
    # Remove tags HTML
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text

def remove_extra_whitespace(text: str) -> str:
    """Remove espaços em branco extras, incluindo quebras de linha."""
    # Substitui múltiplos espaços em branco por um único espaço
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_urls(text: str) -> str:
    """Remove URLs do texto."""
    # Padrão para capturar URLs comuns
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)

def remove_special_characters(
    text: str, 
    keep_punctuation: bool = True,
    keep_accents: bool = True
) -> str:
    """
    Remove caracteres especiais do texto.
    
    Args:
        text: Texto a ser processado
        keep_punctuation: Se True, mantém a pontuação básica
        keep_accents: Se True, mantém acentos e caracteres especiais de idiomas
        
    Returns:
        Texto limpo
    """
    # Lista de pontuações a manter
    punctuation = r'.,;:!?()[]{}"\'-'
    
    if keep_accents:
        # Remove apenas caracteres realmente especiais, mantendo letras acentuadas
        if keep_punctuation:
            pattern = r'[^\w\s' + punctuation + r']'
        else:
            pattern = r'[^\w\s]'
        return re.sub(pattern, '', text)
    else:
        # Normaliza texto (decompõe acentos e depois remove)
        text = unicodedata.normalize('NFKD', text)
        if keep_punctuation:
            pattern = r'[^\w\s' + punctuation + r']'
        else:
            pattern = r'[^\w\s]'
        return re.sub(pattern, '', text)

def normalize_text(
    text: str,
    lower_case: bool = True,
    remove_html: bool = True,
    remove_urls_flag: bool = True,
    remove_extra_spaces: bool = True,
    remove_special_chars: bool = False,
    keep_punctuation: bool = True,
    keep_accents: bool = True
) -> str:
    """
    Normaliza o texto aplicando várias transformações conforme especificado.
    
    Args:
        text: Texto a ser normalizado
        lower_case: Se True, converte para minúsculas
        remove_html: Se True, remove tags HTML
        remove_urls_flag: Se True, remove URLs
        remove_extra_spaces: Se True, remove espaços em branco extras
        remove_special_chars: Se True, remove caracteres especiais
        keep_punctuation: Se True e remove_special_chars é True, mantém pontuação
        keep_accents: Se True e remove_special_chars é True, mantém acentos
        
    Returns:
        Texto normalizado
    """
    if not text:
        return ""
    
    # Aplicar as transformações na ordem
    if remove_html:
        text = remove_html_tags(text)
        
    if remove_urls_flag:
        text = remove_urls(text)
        
    if remove_extra_spaces:
        text = remove_extra_whitespace(text)
        
    if remove_special_chars:
        text = remove_special_characters(
            text, 
            keep_punctuation=keep_punctuation,
            keep_accents=keep_accents
        )
        
    if lower_case:
        text = text.lower()
        
    return text.strip()
