"""
LLM Manager - Responsável por carregar, gerenciar e fazer inferência com modelos de linguagem.
"""

import os
import torch
from typing import Dict, List, Optional, Union, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    TextIteratorStreamer
)
from threading import Thread
import logging

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Gerencia o carregamento, execução e gerenciamento de modelos de linguagem.
    """
    
    def __init__(
        self, 
        model_name_or_path: str = "meta-llama/Llama-3-8B-Instruct",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        use_flash_attention: bool = True
    ):
        """
        Inicializa o gerenciador de LLM.
        
        Args:
            model_name_or_path: ID do modelo no HuggingFace ou caminho local
            device: Dispositivo para execução (cuda, cpu, etc.)
            load_in_8bit: Se True, carrega o modelo quantizado em 8-bit
            load_in_4bit: Se True, carrega o modelo quantizado em 4-bit
            use_flash_attention: Se True, usa flash attention para maior eficiência
        """
        self.model_name = model_name_or_path
        
        # Determinar o dispositivo automaticamente se não for especificado
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attention = use_flash_attention
        
        logger.info(f"Inicializando LLMManager com modelo: {self.model_name}")
        logger.info(f"Dispositivo: {self.device}")
        
        # Configurar cache para os modelos
        os.environ["TRANSFORMERS_CACHE"] = os.path.join("data", "models_cache")
        
        # Inicialmente não carregamos o modelo, apenas quando necessário
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_model(self):
        """Carrega o modelo e tokenizador na memória."""
        if self.model is not None:
            logger.info("Modelo já carregado, pulando.")
            return
        
        logger.info(f"Carregando modelo: {self.model_name}")
        
        # Definir as configurações de quantização
        quantization_config = None
        if self.load_in_8bit or self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=self.load_in_8bit,
                    load_in_4bit=self.load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16
                )
                logger.info(f"Usando quantização: 8-bit={self.load_in_8bit}, 4-bit={self.load_in_4bit}")
            except ImportError:
                logger.warning("Biblioteca bitsandbytes não disponível, desativando quantização.")
                
        # Carregar o tokenizador
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("Tokenizador carregado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar tokenizador: {str(e)}")
            raise
            
        # Configurar atributos adicionais do tokenizador se necessário
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Carregar o modelo
        try:
            kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
            }
            
            if quantization_config:
                kwargs["quantization_config"] = quantization_config
                
            if self.use_flash_attention:
                kwargs["attn_implementation"] = "flash_attention_2"
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **kwargs
            )
            logger.info("Modelo carregado com sucesso.")
            
            # Criar pipeline de geração de texto
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            logger.info("Pipeline de geração de texto configurada.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
            
    def unload_model(self):
        """Libera o modelo da memória."""
        if self.model is not None:
            del self.model
            del self.pipeline
            self.model = None
            self.pipeline = None
            torch.cuda.empty_cache()
            logger.info("Modelo descarregado da memória.")
            
    def generate_text(
        self, 
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Gera texto a partir de um prompt.
        
        Args:
            prompt: O texto de entrada para gerar a continuação
            max_new_tokens: Número máximo de tokens a serem gerados
            temperature: Controla a aleatoriedade (mais alto = mais aleatório)
            top_p: Filtragem nucleus, controla a diversidade
            top_k: Mantém apenas os top_k tokens mais prováveis
            repetition_penalty: Penalidade para repetição de tokens
            stream: Se True, retorna um iterador que gera tokens incrementalmente
            
        Returns:
            O texto gerado ou um iterador, dependendo do parâmetro stream
        """
        # Carregar o modelo se ainda não foi carregado
        if self.model is None:
            self.load_model()
            
        logger.info(f"Gerando texto para prompt de {len(prompt)} caracteres")
        logger.debug(f"Parâmetros: max_new_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}")
        
        # Configurar parâmetros de geração
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs
        }
        
        # Gerar texto com streaming, se solicitado
        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                timeout=10.0
            )
            generation_config["streamer"] = streamer
            
            # Iniciar geração em uma thread separada
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            thread = Thread(
                target=self.model.generate,
                kwargs={
                    "inputs": inputs,
                    **generation_config
                }
            )
            thread.start()
            
            # Retornar o streamer para o cliente iterar
            return streamer
        
        # Geração síncrona (sem streaming)
        outputs = self.pipeline(
            prompt,
            **generation_config
        )
        
        # Extrair e retornar apenas o texto gerado, sem o prompt
        generated_text = outputs[0]["generated_text"]
        
        # Se o texto gerado inclui o prompt, remova-o
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
            
        return generated_text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo carregado."""
        if self.model is None:
            return {"status": "not_loaded", "model_name": self.model_name}
        
        memory_used = None
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "quantized": self.load_in_8bit or self.load_in_4bit,
            "quantization": "8-bit" if self.load_in_8bit else "4-bit" if self.load_in_4bit else None,
            "memory_used_gb": memory_used,
            "parameters": sum(p.numel() for p in self.model.parameters()) / 1e9  # Bilhões
        }
