"""
Repositories - Camada de acesso a dados.
"""

import os
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Generator
import json
import io
import base64
import time

from sqlalchemy import func, desc, asc, and_, or_, not_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from storage.database import get_db_session, ResultType, StorageType
from storage.models import User, Result, Interaction, Preference, UserContext
from storage.serializers import compress_data, decompress_data, serialize_result, deserialize_result

logger = logging.getLogger(__name__)

class UserRepository:
    """Repositório para operações com usuários."""
    
    @staticmethod
    def create_user(
        username: str,
        email: Optional[str] = None,
        external_id: Optional[str] = None
    ) -> User:
        """
        Cria um novo usuário.
        
        Args:
            username: Nome de usuário
            email: Email (opcional)
            external_id: ID externo para integração com sistemas de autenticação
            
        Returns:
            Objeto usuário criado
        """
        with get_db_session() as session:
            # Verificar se já existe
            existing_user = session.query(User).filter(
                or_(
                    User.username == username,
                    and_(User.email == email, email is not None),
                    and_(User.external_id == external_id, external_id is not None)
                )
            ).first()
            
            if existing_user:
                logger.info(f"Usuário já existe: {username}")
                return existing_user
                
            # Criar novo usuário
            user = User(
                username=username,
                email=email,
                external_id=external_id
            )
            
            session.add(user)
            session.flush()  # Para obter o ID gerado
            
            logger.info(f"Novo usuário criado: {username} (ID: {user.id})")
            
            return user
            
    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[User]:
        """
        Obtém um usuário pelo ID.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Objeto usuário ou None se não encontrado
        """
        with get_db_session() as session:
            return session.query(User).filter(User.id == user_id).first()
            
    @staticmethod
    def get_user_by_username(username: str) -> Optional[User]:
        """
        Obtém um usuário pelo nome de usuário.
        
        Args:
            username: Nome de usuário
            
        Returns:
            Objeto usuário ou None se não encontrado
        """
        with get_db_session() as session:
            return session.query(User).filter(User.username == username).first()
            
    @staticmethod
    def get_user_by_external_id(external_id: str) -> Optional[User]:
        """
        Obtém um usuário pelo ID externo.
        
        Args:
            external_id: ID externo
            
        Returns:
            Objeto usuário ou None se não encontrado
        """
        with get_db_session() as session:
            return session.query(User).filter(User.external_id == external_id).first()
            
    @staticmethod
    def get_or_create_user(
        username: str,
        email: Optional[str] = None,
        external_id: Optional[str] = None
    ) -> Tuple[User, bool]:
        """
        Obtém um usuário existente ou cria um novo.
        
        Args:
            username: Nome de usuário
            email: Email (opcional)
            external_id: ID externo (opcional)
            
        Returns:
            Tupla (usuário, criado) onde criado é True se foi criado um novo usuário
        """
        with get_db_session() as session:
            # Verificar se já existe
            existing_user = session.query(User).filter(
                or_(
                    User.username == username,
                    and_(User.email == email, email is not None),
                    and_(User.external_id == external_id, external_id is not None)
                )
            ).first()
            
            if existing_user:
                return existing_user, False
                
            # Criar novo usuário
            user = User(
                username=username,
                email=email,
                external_id=external_id
            )
            
            session.add(user)
            session.flush()
            
            logger.info(f"Novo usuário criado: {username} (ID: {user.id})")
            
            return user, True

class ResultRepository:
    """Repositório para operações com resultados."""
    
    @staticmethod
    def store_result(
        user_id: int,
        result_type: str,
        content: Any,
        storage_type: Optional[str] = None,
        task_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        is_public: bool = False,
        ttl_days: Optional[int] = None  # Time-to-live em dias
    ) -> Result:
        """
        Armazena um resultado.
        
        Args:
            user_id: ID do usuário
            result_type: Tipo de resultado (texto, imagem, etc.)
            content: Conteúdo a ser armazenado
            storage_type: Tipo de armazenamento (opcional, será determinado automaticamente)
            task_id: ID da tarefa associada (opcional)
            workflow_id: ID do fluxo de trabalho associado (opcional)
            metadata: Metadados adicionais (opcional)
            tags: Tags para categorização (opcional)
            is_public: Se o resultado é público
            ttl_days: Dias até expiração (opcional)
            
        Returns:
            Objeto resultado criado
        """
        with get_db_session() as session:
            # Verificar se o usuário existe
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError(f"Usuário não encontrado: {user_id}")
                
            # Determinar tipo de armazenamento se não especificado
            if storage_type is None:
                storage_type = ResultRepository._determine_storage_type(result_type, content)
                
            # Preparar resultado
            result = Result(
                result_id=uuid.uuid4(),
                user_id=user_id,
                task_id=task_id,
                workflow_id=workflow_id,
                result_type=result_type,
                storage_type=storage_type,
                metadata=metadata or {},
                tags=tags or [],
                is_public=is_public
            )
            
            # Configurar expiração se solicitado
            if ttl_days is not None:
                result.expires_at = datetime.utcnow() + timedelta(days=ttl_days)
                
            # Armazenar conteúdo de acordo com o tipo de armazenamento
            if storage_type == StorageType.DATABASE:
                # Armazenar diretamente no banco
                if result_type == ResultType.TEXT:
                    result.content_text = content
                    result.size_bytes = len(content.encode('utf-8'))
                else:
                    # Serializar para JSON
                    result.content_json = serialize_result(content)
                    result.size_bytes = len(json.dumps(result.content_json))
                    
            elif storage_type == StorageType.COMPRESSED:
                # Comprimir dados
                compressed_data = compress_data(content)
                result.content_binary = compressed_data
                result.size_bytes = len(compressed_data)
                
            elif storage_type == StorageType.FILE:
                # Salvar em arquivo e armazenar o caminho
                file_path = ResultRepository._save_to_file(content, result_type, result.result_id)
                result.content_file_path = file_path
                result.size_bytes = os.path.getsize(file_path)
                
            elif storage_type == StorageType.S3:
                # Implementar upload para S3
                # (código omitido para brevidade)
                raise NotImplementedError("Armazenamento S3 não implementado")
                
            # Salvar resultado
            session.add(result)
            session.flush()
            
            logger.info(f"Resultado armazenado: {result.result_id} ({result_type})")
            
            return result
            
    @staticmethod
    def _determine_storage_type(result_type: str, content: Any) -> str:
        """
        Determina o melhor tipo de armazenamento com base no tipo e conteúdo.
        
        Args:
            result_type: Tipo de resultado
            content: Conteúdo a ser armazenado
            
        Returns:
            Tipo de armazenamento recomendado
        """
        if result_type == ResultType.TEXT:
            # Texto pequeno pode ir direto no banco
            if len(content) < 10000:  # Menos de 10KB
                return StorageType.DATABASE
            else:
                # Texto grande vai comprimido
                return StorageType.COMPRESSED
                
        elif result_type == ResultType.EMBEDDING:
            # Embeddings são comprimidos
            return StorageType.COMPRESSED
            
        elif result_type in [ResultType.IMAGE, ResultType.AUDIO, ResultType.VIDEO]:
            # Arquivos binários grandes vão para armazenamento externo
            # Se for string base64, verificar tamanho
            if isinstance(content, str) and content.startswith(('data:', 'http')):
                if len(content) > 1000000:  # Mais de 1MB
                    return StorageType.FILE
                else:
                    return StorageType.COMPRESSED
            else:
                return StorageType.FILE
                
        elif result_type == ResultType.WORKFLOW:
            # Resultados de workflow geralmente são JSON
            return StorageType.DATABASE
            
        # Default para outros tipos
        return StorageType.DATABASE
        
    @staticmethod
    def _save_to_file(content: Any, result_type: str, result_id: uuid.UUID) -> str:
        """
        Salva conteúdo em um arquivo.
        
        Args:
            content: Conteúdo a ser salvo
            result_type: Tipo de resultado
            result_id: ID do resultado
            
        Returns:
            Caminho do arquivo salvo
        """
        # Criar diretório se não existir
        storage_dir = os.path.join("data", "storage", result_type)
        os.makedirs(storage_dir, exist_ok=True)
        
        # Gerar nome de arquivo
        file_name = f"{result_id}"
        
        # Determinar extensão
        if result_type == ResultType.IMAGE:
            ext = ".jpg"
            if isinstance(content, str) and "data:image/" in content:
                mime_type = content.split("data:image/")[1].split(";")[0]
                ext = f".{mime_type}"
        elif result_type == ResultType.AUDIO:
            ext = ".mp3"
        elif result_type == ResultType.VIDEO:
            ext = ".mp4"
        else:
            ext = ".dat"
            
        file_path = os.path.join(storage_dir, file_name + ext)
        
        # Salvar conteúdo
        if isinstance(content, str) and content.startswith('data:'):
            # Dado base64
            content_type, data = content.split(';base64,')
            binary_data = base64.b64decode(data)
            
            with open(file_path, 'wb') as f:
                f.write(binary_data)
        elif isinstance(content, (bytes, bytearray)):
            # Dados binários
            with open(file_path, 'wb') as f:
                f.write(content)
        else:
            # Outros tipos
            with open(file_path, 'wb') as f:
                f.write(serialize_result(content))
                
        return file_path
        
    @staticmethod
    def get_result(result_id: Union[str, uuid.UUID]) -> Optional[Dict[str, Any]]:
        """
        Recupera um resultado pelo ID.
        
        Args:
            result_id: ID do resultado
            
        Returns:
            Dicionário com o resultado e metadados, ou None se não encontrado
        """
        with get_db_session() as session:
            if isinstance(result_id, str):
                try:
                    result_id = uuid.UUID(result_id)
                except ValueError:
                    return None
                    
            # Buscar resultado
            result = session.query(Result).filter(
                Result.result_id == result_id,
                Result.is_deleted == False
            ).first()
            
            if not result:
                return None
                
            # Verificar expiração
            if result.expires_at and result.expires_at < datetime.utcnow():
                return None
                
            # Recuperar conteúdo
            content = ResultRepository._get_result_content(result)
            
            # Atualizar último acesso
            result.updated_at = func.now()
            
            # Montar resposta
            return {
                "result_id": str(result.result_id),
                "user_id": result.user_id,
                "result_type": result.result_type,
                "storage_type": result.storage_type,
                "content": content,
                "metadata": result.metadata,
                "tags": result.tags,
                "is_public": result.is_public,
                "created_at": result.created_at.isoformat(),
                "updated_at": result.updated_at.isoformat(),
                "task_id": result.task_id,
                "workflow_id": result.workflow_id
            }
            
    @staticmethod
    def _get_result_content(result: Result) -> Any:
        """
        Recupera o conteúdo de um resultado de acordo com o tipo de armazenamento.
        
        Args:
            result: Objeto resultado
            
        Returns:
            Conteúdo do resultado
        """
        if result.storage_type == StorageType.DATABASE:
            if result.content_text is not None:
                return result.content_text
            elif result.content_json is not None:
                return deserialize_result(result.content_json)
                
        elif result.storage_type == StorageType.COMPRESSED:
            if result.content_binary is not None:
                return decompress_data(result.content_binary)
                
        elif result.storage_type == StorageType.FILE:
            if result.content_file_path and os.path.exists(result.content_file_path):
                # Para imagens e outros binários, carregar como base64
                if result.result_type in [ResultType.IMAGE, ResultType.AUDIO, ResultType.VIDEO]:
                    with open(result.content_file_path, 'rb') as f:
                        binary_data = f.read()
                        
                    # Determinar MIME type
                    mime_map = {
                        ResultType.IMAGE: "image/jpeg",
                        ResultType.AUDIO: "audio/mpeg",
                        ResultType.VIDEO: "video/mp4"
                    }
                    mime_type = mime_map.get(result.result_type, "application/octet-stream")
                    
                    # Converter para base64
                    base64_data = base64.b64encode(binary_data).decode('utf-8')
                    return f"data:{mime_type};base64,{base64_data}"
                else:
                    # Para outros tipos, carregar como objeto
                    with open(result.content_file_path, 'rb') as f:
                        return deserialize_result(f.read())
                        
        elif result.storage_type == StorageType.S3:
            # Implementar download do S3
            # (código omitido para brevidade)
            raise NotImplementedError("Armazenamento S3 não implementado")
            
        # Fallback
        return None
        
    @staticmethod
    def search_results(
        user_id: Optional[int] = None,
        result_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        query: Optional[str] = None,
        include_public: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Busca resultados com filtros.
        
        Args:
            user_id: Filtrar por usuário (opcional)
            result_type: Filtrar por tipo (opcional)
            tags: Filtrar por tags (opcional)
            query: Busca textual (opcional)
            include_public: Incluir resultados públicos
            limit: Limite de resultados
            offset: Deslocamento para paginação
            
        Returns:
            Tupla (lista de resultados, contagem total)
        """
        with get_db_session() as session:
            # Construir query base
            base_query = session.query(Result).filter(Result.is_deleted == False)
            
            # Aplicar filtros
            if user_id is not None:
                base_query = base_query.filter(
                    or_(
                        Result.user_id == user_id,
                        and_(Result.is_public == True, include_public)
                    )
                )
            elif include_public:
                base_query = base_query.filter(Result.is_public == True)
                
            if result_type is not None:
                base_query = base_query.filter(Result.result_type == result_type)
                
            if tags is not None and tags:
                # Filtro para PostgreSQL usando JSONB
                for tag in tags:
                    base_query = base_query.filter(Result.tags.contains([tag]))
                    
            # Busca textual (simplificada)
            if query is not None and query.strip():
                base_query = base_query.filter(
                    or_(
                        Result.content_text.ilike(f"%{query}%"),
                        Result.metadata.cast(String).ilike(f"%{query}%")
                    )
                )
                
            # Ordenar por data de criação (mais recentes primeiro)
            base_query = base_query.order_by(desc(Result.created_at))
            
            # Contar total
            total_count = base_query.count()
            
            # Aplicar paginação
            results = base_query.limit(limit).offset(offset).all()
            
            # Converter para dicionários
            result_dicts = []
            for result in results:
                # Não incluir conteúdo completo, apenas metadados
                result_dict = {
                    "result_id": str(result.result_id),
                    "user_id": result.user_id,
                    "result_type": result.result_type,
                    "storage_type": result.storage_type,
                    "metadata": result.metadata,
                    "tags": result.tags,
                    "is_public": result.is_public,
                    "created_at": result.created_at.isoformat(),
                    "updated_at": result.updated_at.isoformat(),
                    "task_id": result.task_id,
                    "workflow_id": result.workflow_id,
                    "size_bytes": result.size_bytes
                }
                
                # Para texto, incluir preview
                if result.result_type == ResultType.TEXT and result.content_text:
                    preview_length = 100
                    if len(result.content_text) > preview_length:
                        result_dict["preview"] = result.content_text[:preview_length] + "..."
                    else:
                        result_dict["preview"] = result.content_text
                        
                result_dicts.append(result_dict)
                
            return result_dicts, total_count
            
    @staticmethod
    def delete_result(result_id: Union[str, uuid.UUID]) -> bool:
        """
        Marca um resultado como excluído.
        
        Args:
            result_id: ID do resultado
            
        Returns:
            True se excluído com sucesso, False caso contrário
        """
        with get_db_session() as session:
            if isinstance(result_id, str):
                try:
                    result_id = uuid.UUID(result_id)
                except ValueError:
                    return False
                    
            # Buscar resultado
            result = session.query(Result).filter(
                Result.result_id == result_id,
                Result.is_deleted == False
            ).first()
            
            if not result:
                return False
                
            # Marcar como excluído
            result.is_deleted = True
            
            # Se for arquivo, excluir o arquivo também
            if result.storage_type == StorageType.FILE and result.content_file_path:
                try:
                    if os.path.exists(result.content_file_path):
                        os.remove(result.content_file_path)
                except Exception as e:
                    logger.error(f"Erro ao excluir arquivo: {str(e)}")
                    
            return True

class InteractionRepository:
    """Repositório para operações com interações."""
    
    @staticmethod
    def record_interaction(
        user_id: int,
        interaction_type: str,
        result_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Interaction:
        """
        Registra uma interação do usuário.
        
        Args:
            user_id: ID do usuário
            interaction_type: Tipo de interação
            result_id: ID do resultado associado (opcional)
            metadata: Metadados adicionais (opcional)
            
        Returns:
            Objeto interação criado
        """
        with get_db_session() as session:
            # Criar interação
            interaction = Interaction(
                user_id=user_id,
                result_id=result_id,
                interaction_type=interaction_type,
                metadata=metadata or {}
            )
            
            session.add(interaction)
            session.flush()
            
            logger.info(f"Interação registrada: {interaction.id} ({interaction_type})")
            
            return interaction
            
    @staticmethod
    def get_user_interactions(
        user_id: int,
        interaction_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Obtém interações de um usuário.
        
        Args:
            user_id: ID do usuário
            interaction_type: Filtrar por tipo (opcional)
            limit: Limite de resultados
            offset: Deslocamento para paginação
            
        Returns:
            Lista de interações
        """
        with get_db_session() as session:
            # Construir query
            query = session.query(Interaction).filter(Interaction.user_id == user_id)
            
            if interaction_type:
                query = query.filter(Interaction.interaction_type == interaction_type)
                
            # Ordenar por data (mais recentes primeiro)
            query = query.order_by(desc(Interaction.created_at))
            
            # Aplicar paginação
            interactions = query.limit(limit).offset(offset).all()
            
            # Converter para dicionários
            return [
                {
                    "id": interaction.id,
                    "user_id": interaction.user_id,
                    "result_id": interaction.result_id,
                    "interaction_type": interaction.interaction_type,
                    "metadata": interaction.metadata,
                    "created_at": interaction.created_at.isoformat()
                }
                for interaction in interactions
            ]

class PreferenceRepository:
    """Repositório para operações com preferências."""
    
    @staticmethod
    def set_preference(
        user_id: int,
        category: str,
        key: str,
        value: Union[str, int, float, Dict, List]
    ) -> Preference:
        """
        Define uma preferência de usuário.
        
        Args:
            user_id: ID do usuário
            category: Categoria da preferência
            key: Chave da preferência
            value: Valor da preferência
            
        Returns:
            Objeto preferência criado ou atualizado
        """
        with get_db_session() as session:
            # Verificar se já existe
            preference = session.query(Preference).filter(
                Preference.user_id == user_id,
                Preference.category == category,
                Preference.key == key
            ).first()
            
            if preference:
                # Atualizar existente
                if isinstance(value, (str, bytes)):
                    preference.value_text = value
                    preference.value_number = None
                    preference.value_json = None
                elif isinstance(value, (int, float)):
                    preference.value_text = None
                    preference.value_number = value
                    preference.value_json = None
                else:
                    preference.value_text = None
                    preference.value_number = None
                    preference.value_json = value
            else:
                # Criar nova preferência
                preference = Preference(
                    user_id=user_id,
                    category=category,
                    key=key
                )
                
                if isinstance(value, (str, bytes)):
                    preference.value_text = value
                elif isinstance(value, (int, float)):
                    preference.value_number = value
                else:
                    preference.value_json = value
                    
                session.add(preference)
                
            session.flush()
            
            logger.info(f"Preferência definida: {category}.{key} para usuário {user_id}")
            
            return preference
            
    @staticmethod
    def get_preference(
        user_id: int,
        category: str,
        key: str
    ) -> Optional[Any]:
        """
        Obtém uma preferência de usuário.
        
        Args:
            user_id: ID do usuário
            category: Categoria da preferência
            key: Chave da preferência
            
        Returns:
            Valor da preferência ou None se não encontrada
        """
        with get_db_session() as session:
            preference = session.query(Preference).filter(
                Preference.user_id == user_id,
                Preference.category == category,
                Preference.key == key
            ).first()
            
            if not preference:
                return None
                
            # Retornar o valor não nulo
            if preference.value_text is not None:
                return preference.value_text
            elif preference.value_number is not None:
                return preference.value_number
            else:
                return preference.value_json
                
    @staticmethod
    def get_preferences_by_category(
        user_id: int,
        category: str
    ) -> Dict[str, Any]:
        """
        Obtém todas as preferências de uma categoria.
        
        Args:
            user_id: ID do usuário
            category: Categoria da preferência
            
        Returns:
            Dicionário de preferências
        """
        with get_db_session() as session:
            preferences = session.query(Preference).filter(
                Preference.user_id == user_id,
                Preference.category == category
            ).all()
            
            result = {}
            for pref in preferences:
                # Determinar o valor
                if pref.value_text is not None:
                    value = pref.value_text
                elif pref.value_number is not None:
                    value = pref.value_number
                else:
                    value = pref.value_json
                    
                result[pref.key] = value
                
            return result
            
    @staticmethod
    def delete_preference(
        user_id: int,
        category: str,
        key: str
    ) -> bool:
        """
        Remove uma preferência.
        
        Args:
            user_id: ID do usuário
            category: Categoria da preferência
            key: Chave da preferência
            
        Returns:
            True se removida com sucesso, False caso contrário
        """
        with get_db_session() as session:
            deleted = session.query(Preference).filter(
                Preference.user_id == user_id,
                Preference.category == category,
                Preference.key == key
            ).delete()
            
            return deleted > 0

class UserContextRepository:
    """Repositório para operações com contexto de usuário."""
    
    @staticmethod
    def set_context(
        user_id: int,
        context_type: str,
        context_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserContext:
        """
        Define ou atualiza um contexto de usuário.
        
        Args:
            user_id: ID do usuário
            context_type: Tipo de contexto
            context_id: ID do contexto
            state: Estado do contexto
            metadata: Metadados adicionais (opcional)
            
        Returns:
            Objeto contexto criado ou atualizado
        """
        with get_db_session() as session:
            # Verificar se já existe
            context = session.query(UserContext).filter(
                UserContext.user_id == user_id,
                UserContext.context_type == context_type,
                UserContext.context_id == context_id
            ).first()
            
            if context:
                # Atualizar existente
                context.state = state
                context.metadata = metadata or context.metadata
                context.version += 1
                context.updated_at = func.now()
                context.last_accessed = func.now()
            else:
                # Criar novo contexto
                context = UserContext(
                    user_id=user_id,
                    context_type=context_type,
                    context_id=context_id,
                    state=state,
                    metadata=metadata or {}
                )
                
                session.add(context)
                
            session.flush()
            
            logger.info(f"Contexto definido: {context_type}.{context_id} para usuário {user_id}")
            
            return context
            
    @staticmethod
    def get_context(
        user_id: int,
        context_type: str,
        context_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Obtém um contexto de usuário.
        
        Args:
            user_id: ID do usuário
            context_type: Tipo de contexto
            context_id: ID do contexto
            
        Returns:
            Dicionário com o contexto ou None se não encontrado
        """
        with get_db_session() as session:
            context = session.query(UserContext).filter(
                UserContext.user_id == user_id,
                UserContext.context_type == context_type,
                UserContext.context_id == context_id
            ).first()
            
            if not context:
                return None
                
            # Atualizar último acesso
            context.last_accessed = func.now()
            
            return {
                "user_id": context.user_id,
                "context_type": context.context_type,
                "context_id": context.context_id,
                "state": context.state,
                "metadata": context.metadata,
                "version": context.version,
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat()
            }
            
    @staticmethod
    def list_contexts(
        user_id: int,
        context_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Lista contextos de um usuário.
        
        Args:
            user_id: ID do usuário
            context_type: Filtrar por tipo (opcional)
            limit: Limite de resultados
            
        Returns:
            Lista de contextos
        """
        with get_db_session() as session:
            # Construir query
            query = session.query(UserContext).filter(UserContext.user_id == user_id)
            
            if context_type:
                query = query.filter(UserContext.context_type == context_type)
                
            # Ordenar por último acesso (mais recentes primeiro)
            query = query.order_by(desc(UserContext.last_accessed))
            
            # Limitar resultados
            contexts = query.limit(limit).all()
            
            # Atualizar último acesso de todos
            for context in contexts:
                context.last_accessed = func.now()
                
            # Converter para dicionários (sem incluir o state completo)
            return [
                {
                    "user_id": context.user_id,
                    "context_type": context.context_type,
                    "context_id": context.context_id,
                    "metadata": context.metadata,
                    "version": context.version,
                    "created_at": context.created_at.isoformat(),
                    "updated_at": context.updated_at.isoformat()
                }
                for context in contexts
            ]
