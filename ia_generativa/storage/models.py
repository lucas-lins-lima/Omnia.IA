"""
Models - Definição dos modelos de dados para persistência.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey, LargeBinary, JSON, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from storage.database import Base, ResultType, StorageType

class User(Base):
    """Modelo para usuários."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(255), unique=True, index=True, nullable=True)
    username = Column(String(255), unique=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relações
    results = relationship("Result", back_populates="user")
    interactions = relationship("Interaction", back_populates="user")
    preferences = relationship("Preference", back_populates="user")
    contexts = relationship("UserContext", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.username}>"

class Result(Base):
    """Modelo para resultados de operações."""
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    task_id = Column(String(255), index=True, nullable=True)
    workflow_id = Column(String(255), index=True, nullable=True)
    
    result_type = Column(String(50), index=True)
    storage_type = Column(String(50), index=True)
    
    # Diferentes opções de armazenamento
    content_text = Column(Text, nullable=True)  # Para texto pequeno
    content_json = Column(JSONB, nullable=True)  # Para estruturas JSON (PostgreSQL)
    content_binary = Column(LargeBinary, nullable=True)  # Para dados binários pequenos
    content_file_path = Column(String(512), nullable=True)  # Para referência a arquivo
    content_url = Column(String(1024), nullable=True)  # Para referência a URL
    
    # Metadados
    metadata = Column(JSONB, nullable=True)  # Metadados específicos do resultado
    size_bytes = Column(Integer, nullable=True)  # Tamanho em bytes
    
    # Tags para busca e categorização
    tags = Column(JSONB, nullable=True)  # Lista de tags
    
    # Controle de acesso
    is_public = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime, nullable=True)  # Data de expiração opcional
    
    # Relações
    user = relationship("User", back_populates="results")
    interactions = relationship("Interaction", back_populates="result")
    
    __table_args__ = (
        Index('idx_result_type_user', 'result_type', 'user_id'),
        Index('idx_tags', 'tags', postgresql_using='gin'),
    )
    
    def __repr__(self):
        return f"<Result {self.result_id} ({self.result_type})>"

class Interaction(Base):
    """Modelo para interações do usuário com resultados."""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    result_id = Column(Integer, ForeignKey("results.id"), index=True, nullable=True)
    
    interaction_type = Column(String(50), index=True)  # view, like, dislike, share, etc.
    metadata = Column(JSONB, nullable=True)  # Metadados da interação
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    
    # Relações
    user = relationship("User", back_populates="interactions")
    result = relationship("Result", back_populates="interactions")
    
    __table_args__ = (
        Index('idx_interaction_type_user', 'interaction_type', 'user_id'),
    )
    
    def __repr__(self):
        return f"<Interaction {self.id} ({self.interaction_type})>"

class Preference(Base):
    """Modelo para preferências dos usuários."""
    __tablename__ = "preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    category = Column(String(255), index=True)  # Categoria da preferência
    key = Column(String(255), index=True)  # Chave da preferência
    value_text = Column(Text, nullable=True)  # Valor como texto
    value_number = Column(Float, nullable=True)  # Valor numérico
    value_json = Column(JSONB, nullable=True)  # Valor como JSON
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relações
    user = relationship("User", back_populates="preferences")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'category', 'key', name='uq_user_pref'),
    )
    
    def __repr__(self):
        return f"<Preference {self.category}.{self.key}>"

class UserContext(Base):
    """Modelo para contexto de usuário (histórico recente, estado de conversação, etc.)."""
    __tablename__ = "user_contexts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    context_type = Column(String(50), index=True)  # chat, workflow, session, etc.
    context_id = Column(String(255), index=True)  # ID externo do contexto
    
    state = Column(JSONB, nullable=False)  # Estado do contexto
    metadata = Column(JSONB, nullable=True)  # Metadados adicionais
    
    # Controle de versão
    version = Column(Integer, default=1)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_accessed = Column(DateTime, server_default=func.now())
    
    # Relações
    user = relationship("User", back_populates="contexts")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'context_type', 'context_id', name='uq_user_context'),
    )
    
    def __repr__(self):
        return f"<UserContext {self.context_type}.{self.context_id}>"
