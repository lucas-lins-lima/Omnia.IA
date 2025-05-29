"""
Database - Configuração e conexão com banco de dados.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
import json
import uuid
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey, LargeBinary, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import func

# Configuração de logging
logger = logging.getLogger(__name__)

# Obter configuração do banco de dados a partir de variáveis de ambiente
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASS", "postgres")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "omnia_db")
DB_ECHO = os.environ.get("DB_ECHO", "False").lower() == "true"

# Para SQLite (útil para desenvolvimento/testes)
USE_SQLITE = os.environ.get("USE_SQLITE", "False").lower() == "true"
SQLITE_PATH = os.environ.get("SQLITE_PATH", "sqlite:///data/omnia_db.sqlite")

# Configurar URL de conexão
if USE_SQLITE:
    SQLALCHEMY_DATABASE_URL = SQLITE_PATH
else:
    SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Criar engine do SQLAlchemy
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=DB_ECHO,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Reconectar após 30 minutos
)

# Criar sessão
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(SessionLocal)

# Base para declaração de modelos
Base = declarative_base()

@contextmanager
def get_db_session():
    """Contexto para gerenciar sessões do banco de dados."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Erro de banco de dados: {str(e)}")
        raise
    finally:
        session.close()

def init_db():
    """Inicializa o banco de dados criando todas as tabelas."""
    try:
        # Importar todos os modelos aqui
        from storage.models import User, Result, Interaction, Preference, UserContext
        
        # Criar tabelas
        Base.metadata.create_all(bind=engine)
        logger.info("Banco de dados inicializado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao inicializar banco de dados: {str(e)}")
        raise

def check_db_connection():
    """Verifica se a conexão com o banco de dados está funcionando."""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Erro ao conectar ao banco de dados: {str(e)}")
        return False

class ResultType(str, Enum):
    """Tipos de resultados armazenados."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    WORKFLOW = "workflow"
    EMBEDDING = "embedding"
    OTHER = "other"

class StorageType(str, Enum):
    """Tipos de armazenamento para diferentes dados."""
    DATABASE = "database"  # Armazenado diretamente no banco de dados
    FILE = "file"          # Armazenado como arquivo, com referência no banco
    S3 = "s3"              # Armazenado em bucket S3, com referência no banco
    COMPRESSED = "compressed"  # Armazenado comprimido no banco de dados
