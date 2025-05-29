"""
Video Generator - Responsável por gerar e editar vídeos.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import tempfile
import time
import subprocess
from PIL import Image
from tqdm import tqdm

from preprocessors.video.video_loader import (
    load_video, 
    extract_audio_from_video,
    create_video_from_frames
)
from preprocessors.video.frame_processor import (
    resize_frame,
    apply_color_filter,
    add_text_to_frame
)

logger = logging.getLogger(__name__)

class VideoGenerator:
    """
    Gerencia a geração e edição de vídeos.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        temp_dir: Optional[str] = None
    ):
        """
        Inicializa o gerador de vídeo.
        
        Args:
            device: Dispositivo para execução (cuda, cpu, etc.)
            temp_dir: Diretório para arquivos temporários
        """
        # Determinar o dispositivo automaticamente se não for especificado
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        logger.info(f"Inicializando VideoGenerator")
        logger.info(f"Dispositivo: {self.device}")
        
    def create_slideshow(
        self,
        image_paths: List[str],
        output_path: Optional[str] = None,
        fps: float = 1.0,
        transition_frames: int = 30,
        transition_type: str = "fade",
        audio_path: Optional[str] = None,
        duration_per_image: Optional[float] = None,
        text_overlay: Optional[List[str]] = None
    ) -> str:
        """
        Cria um slideshow a partir de imagens.
        
        Args:
            image_paths: Lista de caminhos para as imagens
            output_path: Caminho para salvar o slideshow
            fps: Frames por segundo
            transition_frames: Número de frames para transição
            transition_type: Tipo de transição (fade, wipe)
            audio_path: Caminho para arquivo de áudio opcional
            duration_per_image: Duração em segundos para cada imagem
            text_overlay: Lista de textos para sobrepor em cada imagem
            
        Returns:
            Caminho do slideshow gerado
        """
        try:
            logger.info(f"Criando slideshow com {len(image_paths)} imagens")
            
            # Verificar se há imagens
            if not image_paths:
                raise ValueError("Nenhuma imagem fornecida")
                
            # Definir caminho de saída
            if output_path is None:
                output_dir = self.temp_dir
                output_path = os.path.join(output_dir, f"slideshow_{int(time.time())}.mp4")
                
            # Carregar imagens
            images = []
            target_size = None
            
            for path in tqdm(image_paths, desc="Carregando imagens"):
                img = cv2.imread(path)
                
                if img is None:
                    logger.warning(f"Não foi possível carregar a imagem: {path}")
                    continue
                    
                # Determinar tamanho padrão (usar o da primeira imagem)
                if target_size is None:
                    h, w = img.shape[:2]
                    # Garantir dimensões pares para codificação de vídeo
                    w = w - (w % 2)
                    h = h - (h % 2)
                    target_size = (w, h)
                    
                # Redimensionar para tamanho padrão
                img = cv2.resize(img, target_size)
                
                # Adicionar texto se fornecido
                if text_overlay and len(text_overlay) > len(images):
                    text = text_overlay[len(images)]
                    # Posição na parte inferior central
                    x = target_size[0] // 2 - 100
                    y = target_size[1] - 50
                    img = add_text_to_frame(
                        img, text, (x, y), font_size=1.0, 
                        background=True, bg_opacity=0.6
                    )
                    
                images.append(img)
                
            if not images:
                raise ValueError("Nenhuma imagem válida fornecida")
                
            # Determinar duração por imagem
            if duration_per_image is not None:
                frames_per_image = int(duration_per_image * fps)
            else:
                # Determinar duração com base na duração do áudio (se fornecido)
                if audio_path:
                    cap = cv2.VideoCapture(audio_path)
                    audio_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    # Calcular frames por imagem para distribuir uniformemente
                    total_frames = audio_duration * fps
                    frames_per_image = int(total_frames / len(images))
                else:
                    # Default: 5 segundos por imagem
                    frames_per_image = int(5 * fps)
                    
            # Garantir que temos frames suficientes para a transição
            if frames_per_image <= transition_frames:
                frames_per_image = transition_frames + 10
                
            logger.info(f"Usando {frames_per_image} frames por imagem")
            
            # Gerar todos os frames do slideshow
            all_frames = []
            
            for i in range(len(images)):
                current_img = images[i]
                next_img = images[(i + 1) % len(images)]
                
                # Adicionar frames sem transição
                for _ in range(frames_per_image - transition_frames):
                    all_frames.append(current_img.copy())
                    
                # Adicionar frames de transição
                if transition_type == "fade":
                    # Transição de fade
                    for j in range(transition_frames):
                        alpha = j / transition_frames
                        blended = cv2.addWeighted(current_img, 1 - alpha, next_img, alpha, 0)
                        all_frames.append(blended)
                elif transition_type == "wipe":
                    # Transição de wipe horizontal
                    h, w = current_img.shape[:2]
                    for j in range(transition_frames):
                        wipe_position = int((j / transition_frames) * w)
                        
                        # Criar frame com wipe
                        frame = current_img.copy()
                        frame[:, :wipe_position] = next_img[:, :wipe_position]
                        
                        all_frames.append(frame)
                else:
                    # Default para fade
                    for j in range(transition_frames):
                        alpha = j / transition_frames
                        blended = cv2.addWeighted(current_img, 1 - alpha, next_img, alpha, 0)
                        all_frames.append(blended)
                        
            # Remover a última transição (loop para a primeira imagem)
            all_frames = all_frames[:-transition_frames]
            
            # Salvar frames temporariamente
            temp_frames_dir = os.path.join(self.temp_dir, f"slideshow_frames_{int(time.time())}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            frame_paths = []
            for i, frame in enumerate(tqdm(all_frames, desc="Salvando frames")):
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
            # Criar vídeo a partir dos frames
            result_path = create_video_from_frames(
                frame_paths,
                output_path=output_path,
                fps=fps,
                audio_path=audio_path
            )
            
            # Limpar arquivos temporários
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
                    
            try:
                os.rmdir(temp_frames_dir)
            except:
                pass
                
            logger.info(f"Slideshow criado: {result_path}")
            
            return result_path
            
        except Exception as e:
            logger.error(f"Erro ao criar slideshow: {str(e)}")
            raise
            
    def add_text_to_video(
        self,
        video_path: str,
        text: str,
        position: str = "bottom",
        output_path: Optional[str] = None,
        font_size: float = 1.0,
        color: Tuple[int, int, int] = (255, 255, 255),
        background: bool = True,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> str:
        """
        Adiciona texto a um vídeo.
        
        Args:
            video_path: Caminho para o vídeo
            text: Texto a ser adicionado
            position: Posição do texto (top, center, bottom)
            output_path: Caminho para salvar o vídeo resultante
            font_size: Tamanho da fonte
            color: Cor do texto (BGR)
            background: Se True, adiciona fundo atrás do texto
            start_time: Tempo inicial para mostrar o texto (segundos)
            end_time: Tempo final para mostrar o texto (segundos)
            
        Returns:
            Caminho do vídeo com texto
        """
        try:
            logger.info(f"Adicionando texto ao vídeo: {video_path}")
            
            # Definir caminho de saída
            if output_path is None:
                output_dir = self.temp_dir
                output_path = os.path.join(output_dir, f"video_with_text_{int(time.time())}.mp4")
                
            # Carregar vídeo
            frames, metadata = load_video(
                video_path,
                max_frames=None,
                start_time=None,
                end_time=None
            )
            
            # Calcular posição do texto
            # Usar um frame para referência
            h, w = frames[0].shape[:2]
            
            if position == "top":
                pos_x = w // 2 - 100  # Aproximado, ajustar com base no texto
                pos_y = 50
            elif position == "center":
                pos_x = w // 2 - 100
                pos_y = h // 2
            else:  # bottom (default)
                pos_x = w // 2 - 100
                pos_y = h - 50
                
            # Calcular índices de frames para start/end
            fps = metadata["fps"]
            start_frame = 0
            end_frame = len(frames)
            
            if start_time is not None:
                start_frame = int(start_time * fps)
                
            if end_time is not None:
                end_frame = min(len(frames), int(end_time * fps))
                
            # Adicionar texto aos frames
            for i in range(len(frames)):
                if i >= start_frame and i < end_frame:
                    frames[i] = add_text_to_frame(
                        frames[i], text, (pos_x, pos_y), 
                        font_size=font_size, 
                        color=color, 
                        background=background
                    )
                    
            # Salvar frames temporariamente
            temp_frames_dir = os.path.join(self.temp_dir, f"text_frames_{int(time.time())}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            frame_paths = []
            for i, frame in enumerate(tqdm(frames, desc="Salvando frames")):
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
            # Extrair áudio original
            audio_path = extract_audio_from_video(video_path)
            
            # Criar vídeo a partir dos frames
            result_path = create_video_from_frames(
                frame_paths,
                output_path=output_path,
                fps=fps,
                audio_path=audio_path
            )
            
            # Limpar arquivos temporários
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
                    
            try:
                os.rmdir(temp_frames_dir)
            except:
                pass
                
            if audio_path and audio_path != output_path:
                try:
                    os.remove(audio_path)
                except:
                    pass
                    
            logger.info(f"Vídeo com texto salvo em: {result_path}")
            
            return result_path
            
        except Exception as e:
            logger.error(f"Erro ao adicionar texto ao vídeo: {str(e)}")
            raise
            
    def create_montage(
        self,
        video_paths: List[str],
        layout: str = "grid",
        grid_size: Optional[Tuple[int, int]] = None,
        output_path: Optional[str] = None,
        output_size: Optional[Tuple[int, int]] = None,
        include_audio: bool = True
    ) -> str:
        """
        Cria uma montagem de múltiplos vídeos.
        
        Args:
            video_paths: Lista de caminhos para os vídeos
            layout: Tipo de layout (grid, horizontal, vertical)
            grid_size: Tamanho da grade (rows, cols)
            output_path: Caminho para salvar a montagem
            output_size: Tamanho do vídeo de saída (width, height)
            include_audio: Se True, inclui áudio do primeiro vídeo
            
        Returns:
            Caminho da montagem gerada
        """
        try:
            logger.info(f"Criando montagem com {len(video_paths)} vídeos")
            
            # Verificar se há vídeos
            if not video_paths:
                raise ValueError("Nenhum vídeo fornecido")
                
            # Definir caminho de saída
            if output_path is None:
                output_dir = self.temp_dir
                output_path = os.path.join(output_dir, f"montage_{int(time.time())}.mp4")
                
            # Determinar layout da grade
            num_videos = len(video_paths)
            
            if layout == "grid":
                if grid_size:
                    rows, cols = grid_size
                else:
                    # Determinar automaticamente com base na raiz quadrada
                    cols = int(np.ceil(np.sqrt(num_videos)))
                    rows = int(np.ceil(num_videos / cols))
            elif layout == "horizontal":
                rows, cols = 1, num_videos
            elif layout == "vertical":
                rows, cols = num_videos, 1
            else:
                logger.warning(f"Layout desconhecido: {layout}, usando grid")
                cols = int(np.ceil(np.sqrt(num_videos)))
                rows = int(np.ceil(num_videos / cols))
                
            logger.info(f"Usando layout {rows}x{cols}")
            
            # Carregar vídeos e obter informações
            videos = []
            max_frames = 0
            common_fps = None
            
            for path in tqdm(video_paths, desc="Carregando vídeos"):
                # Ler informações do vídeo
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # Armazenar para uso posterior
                videos.append({
                    "path": path,
                    "fps": fps,
                    "total_frames": total_frames,
                    "width": width,
                    "height": height
                })
                
                # Atualizar máximo de frames
                max_frames = max(max_frames, total_frames)
                
                # Definir FPS comum (usar o primeiro vídeo como referência)
                if common_fps is None:
                    common_fps = fps
                    
            # Determinar tamanho do frame para cada vídeo na montagem
            if output_size:
                montage_width, montage_height = output_size
            else:
                # Usar tamanho do primeiro vídeo como referência
                single_width = videos[0]["width"]
                single_height = videos[0]["height"]
                
                # Calcular tamanho total
                montage_width = single_width * cols
                montage_height = single_height * rows
                
            # Calcular tamanho do frame individual na montagem
            cell_width = montage_width // cols
            cell_height = montage_height // rows
            
            # Criar frames da montagem
            montage_frames = []
            
            # Abrir todos os vídeos
            video_captures = [cv2.VideoCapture(v["path"]) for v in videos]
            
            # Processar frames
            for frame_idx in tqdm(range(max_frames), desc="Criando montagem"):
                # Criar frame da montagem
                montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)
                
                # Obter frame de cada vídeo
                for i, cap in enumerate(video_captures):
                    if i >= num_videos:
                        break
                        
                    # Verificar se ainda há frames
                    if frame_idx < videos[i]["total_frames"]:
                        # Ler frame
                        ret, frame = cap.read()
                        
                        if ret:
                            # Redimensionar para tamanho da célula
                            frame = cv2.resize(frame, (cell_width, cell_height))
                            
                            # Calcular posição na montagem
                            row = i // cols
                            col = i % cols
                            
                            # Inserir na montagem
                            y_start = row * cell_height
                            y_end = y_start + cell_height
                            x_start = col * cell_width
                            x_end = x_start + cell_width
                            
                            try:
                                montage[y_start:y_end, x_start:x_end] = frame
                            except:
                                # Em caso de problemas com dimensões
                                h, w = frame.shape[:2]
                                montage[y_start:y_start+h, x_start:x_start+w] = frame
                                
                # Adicionar à lista de frames
                montage_frames.append(montage)
                
            # Liberar recursos
            for cap in video_captures:
                cap.release()
                
            # Salvar frames temporariamente
            temp_frames_dir = os.path.join(self.temp_dir, f"montage_frames_{int(time.time())}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            frame_paths = []
            for i, frame in enumerate(tqdm(montage_frames, desc="Salvando frames")):
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
            # Extrair áudio do primeiro vídeo se solicitado
            audio_path = None
            if include_audio and video_paths:
                try:
                    audio_path = extract_audio_from_video(video_paths[0])
                except Exception as e:
                    logger.warning(f"Não foi possível extrair áudio: {str(e)}")
                    
            # Criar vídeo a partir dos frames
            result_path = create_video_from_frames(
                frame_paths,
                output_path=output_path,
                fps=common_fps,
                audio_path=audio_path
            )
            
            # Limpar arquivos temporários
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
                    
            try:
                os.rmdir(temp_frames_dir)
            except:
                pass
                
            if audio_path and audio_path != output_path:
                try:
                    os.remove(audio_path)
                except:
                    pass
                    
            logger.info(f"Montagem salva em: {result_path}")
            
            return result_path
            
        except Exception as e:
            logger.error(f"Erro ao criar montagem: {str(e)}")
            raise
