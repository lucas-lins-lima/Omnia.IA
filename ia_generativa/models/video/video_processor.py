"""
Video Processor - Responsável pelo processamento e manipulação de vídeos.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import tempfile
import time
import torch
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from preprocessors.video.video_loader import (
    load_video, 
    extract_audio_from_video, 
    create_video_from_frames
)
from preprocessors.video.frame_processor import (
    resize_frame,
    apply_color_filter,
    adjust_frame,
    add_text_to_frame,
    apply_effect_to_frame
)

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Gerencia o processamento e manipulação de vídeos.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        max_frames_in_memory: int = 300,
        num_workers: int = 4,
        temp_dir: Optional[str] = None
    ):
        """
        Inicializa o processador de vídeo.
        
        Args:
            device: Dispositivo para execução (cuda, cpu, etc.)
            max_frames_in_memory: Número máximo de frames para manter em memória
            num_workers: Número de workers para processamento paralelo
            temp_dir: Diretório para arquivos temporários
        """
        # Determinar o dispositivo automaticamente se não for especificado
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.max_frames_in_memory = max_frames_in_memory
        self.num_workers = num_workers
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        logger.info(f"Inicializando VideoProcessor")
        logger.info(f"Dispositivo: {self.device}")
        logger.info(f"Max frames em memória: {self.max_frames_in_memory}")
        logger.info(f"Número de workers: {self.num_workers}")
        
    def process_video(
        self,
        video_path: str,
        operations: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sample_rate: Optional[int] = None,
        preserve_audio: bool = True
    ) -> str:
        """
        Processa um vídeo aplicando uma lista de operações.
        
        Args:
            video_path: Caminho para o vídeo
            operations: Lista de operações a serem aplicadas
            output_path: Caminho para salvar o vídeo processado
            start_time: Tempo inicial para processamento (segundos)
            end_time: Tempo final para processamento (segundos)
            sample_rate: Taxa de amostragem (processar 1 a cada N frames)
            preserve_audio: Se True, preserva o áudio do vídeo original
            
        Returns:
            Caminho do vídeo processado
        """
        start_process_time = time.time()
        
        try:
            logger.info(f"Processando vídeo: {video_path}")
            logger.info(f"Operações: {operations}")
            
            # Carregar frames do vídeo
            frames, metadata = load_video(
                video_path,
                max_frames=None,
                frame_interval=sample_rate or 1,
                start_time=start_time,
                end_time=end_time
            )
            
            # Processar frames em lotes para economizar memória
            processed_frames = []
            
            # Processar frames em lotes
            for i in range(0, len(frames), self.max_frames_in_memory):
                batch = frames[i:i+self.max_frames_in_memory]
                
                # Processar batch
                batch_processed = self._process_frames_batch(batch, operations)
                
                # Adicionar ao resultado
                processed_frames.extend(batch_processed)
                
            logger.info(f"Processados {len(processed_frames)} frames")
            
            # Definir caminho de saída
            if output_path is None:
                output_dir = self.temp_dir
                output_path = os.path.join(output_dir, f"processed_video_{int(time.time())}.mp4")
                
            # Extrair áudio se necessário
            audio_path = None
            if preserve_audio:
                try:
                    audio_path = extract_audio_from_video(
                        video_path,
                        start_time=start_time,
                        end_time=end_time
                    )
                except Exception as e:
                    logger.warning(f"Não foi possível extrair áudio: {str(e)}")
                    
            # Salvar frames como vídeo
            temp_frames_dir = os.path.join(self.temp_dir, f"frames_{int(time.time())}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            frame_paths = []
            for i, frame in enumerate(processed_frames):
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
            # Criar vídeo a partir dos frames
            result_path = create_video_from_frames(
                frame_paths,
                output_path=output_path,
                fps=metadata["fps"],
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
                    
            elapsed_time = time.time() - start_process_time
            logger.info(f"Vídeo processado em {elapsed_time:.2f} segundos")
            logger.info(f"Vídeo salvo em: {result_path}")
            
            return result_path
            
        except Exception as e:
            logger.error(f"Erro ao processar vídeo: {str(e)}")
            raise
            
    def _process_frames_batch(
        self,
        frames: List[np.ndarray],
        operations: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """
        Processa um lote de frames.
        
        Args:
            frames: Lista de frames (arrays NumPy)
            operations: Lista de operações a serem aplicadas
            
        Returns:
            Lista de frames processados
        """
        # Fazer cópia para não modificar os originais
        result_frames = [frame.copy() for frame in frames]
        
        # Aplicar cada operação sequencialmente
        for operation in operations:
            op_type = operation.get("type", "")
            
            if op_type == "resize":
                width = operation.get("width")
                height = operation.get("height")
                scale = operation.get("scale")
                
                result_frames = [
                    resize_frame(frame, width, height, scale)
                    for frame in result_frames
                ]
                
            elif op_type == "filter":
                filter_type = operation.get("filter_type", "grayscale")
                intensity = operation.get("intensity", 1.0)
                
                result_frames = [
                    apply_color_filter(frame, filter_type, intensity)
                    for frame in result_frames
                ]
                
            elif op_type == "adjust":
                brightness = operation.get("brightness", 0.0)
                contrast = operation.get("contrast", 0.0)
                saturation = operation.get("saturation", 0.0)
                
                result_frames = [
                    adjust_frame(frame, brightness, contrast, saturation)
                    for frame in result_frames
                ]
                
            elif op_type == "text":
                text = operation.get("text", "")
                position = operation.get("position", (50, 50))
                font_size = operation.get("font_size", 1.0)
                color = operation.get("color", (255, 255, 255))
                thickness = operation.get("thickness", 2)
                background = operation.get("background", False)
                bg_color = operation.get("bg_color", (0, 0, 0))
                bg_opacity = operation.get("bg_opacity", 0.5)
                
                result_frames = [
                    add_text_to_frame(
                        frame, text, position, font_size, color, 
                        thickness, background, bg_color, bg_opacity
                    )
                    for frame in result_frames
                ]
                
            elif op_type == "effect":
                effect_type = operation.get("effect_type", "")
                params = operation.get("params", {})
                
                result_frames = [
                    apply_effect_to_frame(frame, effect_type, params)
                    for frame in result_frames
                ]
                
            else:
                logger.warning(f"Operação desconhecida: {op_type}")
                
        return result_frames
        
    def trim_video(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: Optional[str] = None
    ) -> str:
        """
        Recorta um vídeo entre dois pontos de tempo.
        
        Args:
            video_path: Caminho para o vídeo
            start_time: Tempo inicial (segundos)
            end_time: Tempo final (segundos)
            output_path: Caminho para salvar o vídeo recortado
            
        Returns:
            Caminho do vídeo recortado
        """
        try:
            logger.info(f"Recortando vídeo: {video_path}")
            logger.info(f"Intervalo: {start_time}s - {end_time}s")
            
            # Definir caminho de saída
            if output_path is None:
                output_dir = self.temp_dir
                output_path = os.path.join(output_dir, f"trimmed_video_{int(time.time())}.mp4")
                
            # Recortar vídeo com ffmpeg
            command = [
                "ffmpeg",
                "-i", video_path,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c:v", "libx264",
                "-c:a", "aac",
                output_path,
                "-y"
            ]
            
            # Executar comando
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"Vídeo recortado salvo em: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Erro ao recortar vídeo: {str(e)}")
            raise
            
    def concatenate_videos(
        self,
        video_paths: List[str],
        output_path: Optional[str] = None,
        crossfade_duration: Optional[float] = None
    ) -> str:
        """
        Concatena múltiplos vídeos.
        
        Args:
            video_paths: Lista de caminhos para os vídeos
            output_path: Caminho para salvar o vídeo concatenado
            crossfade_duration: Duração da transição em segundos (opcional)
            
        Returns:
            Caminho do vídeo concatenado
        """
        try:
            logger.info(f"Concatenando {len(video_paths)} vídeos")
            
            # Verificar se há vídeos
            if not video_paths:
                raise ValueError("Nenhum vídeo fornecido")
                
            # Definir caminho de saída
            if output_path is None:
                output_dir = self.temp_dir
                output_path = os.path.join(output_dir, f"concatenated_video_{int(time.time())}.mp4")
                
            # Criar arquivo de lista para ffmpeg
            list_file_path = os.path.join(self.temp_dir, f"concat_list_{int(time.time())}.txt")
            
            with open(list_file_path, "w") as f:
                for video_path in video_paths:
                    f.write(f"file '{os.path.abspath(video_path)}'\n")
                    
            # Concatenar vídeos com ffmpeg
            if crossfade_duration is None:
                # Concatenação simples
                command = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file_path,
                    "-c", "copy",
                    output_path,
                    "-y"
                ]
            else:
                # Concatenação com crossfade (mais complexo, requer recodificação)
                # Nota: Esta é uma implementação simplificada
                crossfade_frames = int(crossfade_duration * 25)  # Assumindo 25 FPS
                
                # Primeiro, concatenar sem transição
                temp_output = os.path.join(self.temp_dir, f"temp_concat_{int(time.time())}.mp4")
                command = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file_path,
                    "-c", "copy",
                    temp_output,
                    "-y"
                ]
                
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Aplicar crossfade
                command = [
                    "ffmpeg",
                    "-i", temp_output,
                    "-filter_complex", f"fade=t=in:st=0:d={crossfade_duration},fade=t=out:st=outpoint-{crossfade_duration}:d={crossfade_duration}",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    output_path,
                    "-y"
                ]
                
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Limpar arquivo temporário
                os.remove(temp_output)
            
            # Executar comando
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Limpar arquivo de lista
            os.remove(list_file_path)
            
            logger.info(f"Vídeos concatenados salvos em: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Erro ao concatenar vídeos: {str(e)}")
            raise
            
    def add_audio_to_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        replace_audio: bool = True
    ) -> str:
        """
        Adiciona ou mescla áudio a um vídeo.
        
        Args:
            video_path: Caminho para o vídeo
            audio_path: Caminho para o áudio
            output_path: Caminho para salvar o vídeo resultante
            replace_audio: Se True, substitui o áudio original; se False, mescla
            
        Returns:
            Caminho do vídeo com áudio adicionado
        """
        try:
            logger.info(f"Adicionando áudio ao vídeo: {video_path}")
            
            # Definir caminho de saída
            if output_path is None:
                output_dir = self.temp_dir
                output_path = os.path.join(output_dir, f"video_with_audio_{int(time.time())}.mp4")
                
            # Construir comando ffmpeg
            if replace_audio:
                # Substituir áudio original
                command = [
                    "ffmpeg",
                    "-i", video_path,
                    "-i", audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    output_path,
                    "-y"
                ]
            else:
                # Mesclar com áudio original
                command = [
                    "ffmpeg",
                    "-i", video_path,
                    "-i", audio_path,
                    "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first[aout]",
                    "-c:v", "copy",
                    "-map", "0:v:0",
                    "-map", "[aout]",
                    output_path,
                    "-y"
                ]
                
            # Executar comando
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"Vídeo com áudio salvo em: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Erro ao adicionar áudio ao vídeo: {str(e)}")
            raise
            
    def change_video_speed(
        self,
        video_path: str,
        speed_factor: float,
        output_path: Optional[str] = None,
        preserve_audio_pitch: bool = True
    ) -> str:
        """
        Altera a velocidade de um vídeo.
        
        Args:
            video_path: Caminho para o vídeo
            speed_factor: Fator de velocidade (>1 para acelerar, <1 para desacelerar)
            output_path: Caminho para salvar o vídeo resultante
            preserve_audio_pitch: Se True, preserva o tom do áudio
            
        Returns:
            Caminho do vídeo com velocidade alterada
        """
        try:
            logger.info(f"Alterando velocidade do vídeo: {video_path}")
            logger.info(f"Fator de velocidade: {speed_factor}")
            
            # Definir caminho de saída
            if output_path is None:
                output_dir = self.temp_dir
                output_path = os.path.join(output_dir, f"speed_video_{int(time.time())}.mp4")
                
            # Calcular filtro de tempo
            tempo = 1.0 / speed_factor
            
            # Construir comando ffmpeg
            if preserve_audio_pitch:
                # Preservar tom do áudio
                command = [
                    "ffmpeg",
                    "-i", video_path,
                    "-filter_complex", f"[0:v]setpts={tempo}*PTS[v];[0:a]atempo={speed_factor}[a]",
                    "-map", "[v]",
                    "-map", "[a]",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    output_path,
                    "-y"
                ]
            else:
                # Não preservar tom do áudio
                                command = [
                    "ffmpeg",
                    "-i", video_path,
                    "-filter_complex", f"[0:v]setpts={tempo}*PTS[v];[0:a]asetrate=r={44100*speed_factor},asetpts={tempo}*PTS[a]",
                    "-map", "[v]",
                    "-map", "[a]",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    output_path,
                    "-y"
                ]
                
            # Executar comando
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"Vídeo com velocidade alterada salvo em: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Erro ao alterar velocidade do vídeo: {str(e)}")
            raise
            
    def create_timelapse(
        self,
        video_path: str,
        speed_factor: float = 10.0,
        output_path: Optional[str] = None,
        frame_interval: int = 1
    ) -> str:
        """
        Cria um timelapse a partir de um vídeo.
        
        Args:
            video_path: Caminho para o vídeo
            speed_factor: Fator de aceleração do timelapse
            output_path: Caminho para salvar o timelapse
            frame_interval: Intervalo de frames a considerar
            
        Returns:
            Caminho do vídeo timelapse
        """
        try:
            logger.info(f"Criando timelapse do vídeo: {video_path}")
            logger.info(f"Fator de aceleração: {speed_factor}")
            
            # Definir caminho de saída
            if output_path is None:
                output_dir = self.temp_dir
                output_path = os.path.join(output_dir, f"timelapse_{int(time.time())}.mp4")
                
            # Extrair frames em intervalos
            frames, metadata = load_video(
                video_path,
                max_frames=None,
                frame_interval=frame_interval
            )
            
            # Salvar frames para montagem do vídeo
            temp_frames_dir = os.path.join(self.temp_dir, f"timelapse_frames_{int(time.time())}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            frame_paths = []
            for i, frame in enumerate(frames):
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
            # Calcular novo FPS para o timelapse
            new_fps = min(metadata["fps"] * speed_factor, 60.0)  # Limitar a 60 FPS
            
            # Criar vídeo a partir dos frames
            result_path = create_video_from_frames(
                frame_paths,
                output_path=output_path,
                fps=new_fps
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
                
            logger.info(f"Timelapse salvo em: {result_path}")
            
            return result_path
            
        except Exception as e:
            logger.error(f"Erro ao criar timelapse: {str(e)}")
            raise
            
    def extract_scene_transitions(
        self,
        video_path: str,
        threshold: float = 30.0,
        min_scene_length: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Detecta transições entre cenas em um vídeo.
        
        Args:
            video_path: Caminho para o vídeo
            threshold: Limiar de diferença para detectar transição
            min_scene_length: Duração mínima de uma cena em segundos
            
        Returns:
            Lista de dicionários com informações sobre as cenas
        """
        try:
            logger.info(f"Detectando transições de cena no vídeo: {video_path}")
            
            # Carregar vídeo
            frames, metadata = load_video(video_path, frame_interval=2)
            
            # Calcular diferenças entre frames consecutivos
            scene_transitions = []
            prev_frame_gray = None
            fps = metadata["fps"]
            min_frames = int(min_scene_length * fps / 2)  # Considerando o frame_interval=2
            
            for i, frame in enumerate(frames):
                # Converter para escala de cinza
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame_gray is not None:
                    # Calcular diferença absoluta
                    diff = cv2.absdiff(frame_gray, prev_frame_gray)
                    
                    # Calcular média da diferença
                    mean_diff = np.mean(diff)
                    
                    # Verificar se excede o limiar
                    if mean_diff > threshold:
                        # Calcular tempo da transição
                        time_sec = i * 2 / fps  # Considerando o frame_interval=2
                        
                        # Verificar se a cena tem duração mínima
                        if not scene_transitions or time_sec - scene_transitions[-1]["time"] >= min_scene_length:
                            logger.info(f"Transição detectada em {time_sec:.2f}s, diferença: {mean_diff:.2f}")
                            
                            scene_transitions.append({
                                "frame": i,
                                "time": time_sec,
                                "difference": float(mean_diff)
                            })
                            
                prev_frame_gray = frame_gray
                
            logger.info(f"Detectadas {len(scene_transitions)} transições de cena")
            
            return scene_transitions
            
        except Exception as e:
            logger.error(f"Erro ao detectar transições de cena: {str(e)}")
            raise
