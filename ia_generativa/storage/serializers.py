"""
Serializers - Utilitários para serialização e compressão de dados.
"""

import json
import pickle
import base64
import zlib
import lzma
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Compressão
def compress_data(data: Any, method: str = "zlib") -> bytes:
    """
    Comprime dados usando o método especificado.
    
    Args:
        data: Dados a serem comprimidos
        method: Método de compressão ('zlib' ou 'lzma')
        
    Returns:
        Dados comprimidos como bytes
    """
    # Serializar dados
    serialized = pickle.dumps(data)
    
    # Comprimir
    if method == "zlib":
        compressed = zlib.compress(serialized, level=9)  # Nível máximo de compressão
    elif method == "lzma":
        compressed = lzma.compress(serialized)
    else:
        logger.warning(f"Método de compressão não reconhecido: {method}, usando zlib")
        compressed = zlib.compress(serialized)
        
    return compressed

def decompress_data(compressed_data: bytes, method: str = "auto") -> Any:
    """
    Descomprime dados.
    
    Args:
        compressed_data: Dados comprimidos
        method: Método de compressão ('auto', 'zlib' ou 'lzma')
        
    Returns:
        Dados descomprimidos
    """
    # Determinar método automaticamente
    if method == "auto":
        # LZMA tem um cabeçalho específico
        if compressed_data.startswith(b"\xfd\x37\x7a\x58\x5a\x00"):
            method = "lzma"
        else:
            method = "zlib"
            
    # Descomprimir
    try:
        if method == "zlib":
            decompressed = zlib.decompress(compressed_data)
        elif method == "lzma":
            decompressed = lzma.decompress(compressed_data)
        else:
            logger.warning(f"Método de descompressão não reconhecido: {method}, tentando zlib")
            decompressed = zlib.decompress(compressed_data)
            
        # Deserializar
        return pickle.loads(decompressed)
    except Exception as e:
        logger.error(f"Erro ao descomprimir dados: {str(e)}")
        raise

# Serialização
def serialize_result(data: Any) -> Dict[str, Any]:
    """
    Serializa dados para armazenamento, tratando tipos especiais.
    
    Args:
        data: Dados a serem serializados
        
    Returns:
        Dados serializados em formato compatível com JSON
    """
    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    elif isinstance(data, (bytes, bytearray)):
        # Converter para base64
        return {
            "_type": "binary",
            "data": base64.b64encode(data).decode('utf-8')
        }
    elif isinstance(data, (list, tuple)):
        return [serialize_result(item) for item in data]
    elif isinstance(data, dict):
        return {k: serialize_result(v) for k, v in data.items()}
    elif hasattr(data, "__dict__"):
        # Objeto com atributos
        return {
            "_type": "object",
            "class": data.__class__.__name__,
            "data": serialize_result(data.__dict__)
        }
    else:
        # Fallback: pickle e base64
        try:
            pickled = pickle.dumps(data)
            return {
                "_type": "pickled",
                "data": base64.b64encode(pickled).decode('utf-8')
            }
        except:
            # Último recurso: converter para string
            return str(data)

def deserialize_result(data: Any) -> Any:
    """
    Deserializa dados de armazenamento, reconstruindo tipos especiais.
    
    Args:
        data: Dados serializados
        
    Returns:
        Dados deserializados
    """
    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    elif isinstance(data, list):
        return [deserialize_result(item) for item in data]
    elif isinstance(data, dict):
        if "_type" in data:
            if data["_type"] == "binary":
                # Converter de base64
                return base64.b64decode(data["data"])
            elif data["_type"] == "pickled":
                # Deserializar de pickle
                pickled = base64.b64decode(data["data"])
                return pickle.loads(pickled)
            elif data["_type"] == "object":
                # Reconstruir objeto simples
                obj_data = deserialize_result(data["data"])
                # Nota: apenas reconstruímos um dicionário aqui,
                # não o objeto original completo
                return obj_data
        return {k: deserialize_result(v) for k, v in data.items()}
    else:
        return data
