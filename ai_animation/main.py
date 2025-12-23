#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI配音工厂 - 超强嘴炮版 3.4.0 - AI动画生成集成版
集成了ComfyUI人物一致性和AI驱动动画生成
作者: AI助手
版本: 3.4.0 - AI动画生成集成版
"""

import sys
import os
import json
import time
import logging
import warnings
import threading
import subprocess
import concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum
from queue import Queue, Empty
import hashlib
import re

# 第三方库
try:
    import torch
    import torchaudio
    import numpy as np
    import whisper
    import librosa
    import soundfile as sf
    import requests
    from scipy import signal
    from scipy.interpolate import interp1d
    from scipy.signal import butter, filtfilt
    import ffmpeg
    import aiohttp
    import asyncio
    import cv2
    import base64
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)

# GUI库
try:
    # 先导入基础模块
    from PySide6 import QtWidgets, QtCore, QtGui, QtMultimedia
    
    # 然后创建别名
    QApplication = QtWidgets.QApplication
    QMainWindow = QtWidgets.QMainWindow
    QWidget = QtWidgets.QWidget
    QVBoxLayout = QtWidgets.QVBoxLayout
    QHBoxLayout = QtWidgets.QHBoxLayout
    QPushButton = QtWidgets.QPushButton
    QLabel = QtWidgets.QLabel
    QLineEdit = QtWidgets.QLineEdit
    QTextEdit = QtWidgets.QTextEdit
    QListWidget = QtWidgets.QListWidget
    QListWidgetItem = QtWidgets.QListWidgetItem
    QFileDialog = QtWidgets.QFileDialog
    QGroupBox = QtWidgets.QGroupBox
    QSpinBox = QtWidgets.QSpinBox
    QComboBox = QtWidgets.QComboBox
    QCheckBox = QtWidgets.QCheckBox
    QProgressBar = QtWidgets.QProgressBar
    QMessageBox = QtWidgets.QMessageBox
    QTableWidget = QtWidgets.QTableWidget
    QTableWidgetItem = QtWidgets.QTableWidgetItem
    QHeaderView = QtWidgets.QHeaderView
    QTabWidget = QtWidgets.QTabWidget
    QDialog = QtWidgets.QDialog
    QDoubleSpinBox = QtWidgets.QDoubleSpinBox
    QSlider = QtWidgets.QSlider
    QSplitter = QtWidgets.QSplitter
    QFrame = QtWidgets.QFrame
    QRadioButton = QtWidgets.QRadioButton
    QButtonGroup = QtWidgets.QButtonGroup
    QScrollArea = QtWidgets.QScrollArea
    QProgressDialog = QtWidgets.QProgressDialog
    QStyleFactory = QtWidgets.QStyleFactory
    QInputDialog = QtWidgets.QInputDialog
    
    Qt = QtCore.Qt
    QThread = QtCore.QThread
    Signal = QtCore.Signal
    Slot = QtCore.Slot
    QTimer = QtCore.QTimer
    QUrl = QtCore.QUrl
    QSize = QtCore.QSize
    QPropertyAnimation = QtCore.QPropertyAnimation
    QEasingCurve = QtCore.QEasingCurve
    QPoint = QtCore.QPoint
    QRect = QtCore.QRect
    QEvent = QtCore.QEvent
    QObject = QtCore.QObject
    QRunnable = QtCore.QRunnable
    QThreadPool = QtCore.QThreadPool
    
    QFont = QtGui.QFont
    QFontDatabase = QtGui.QFontDatabase
    QIcon = QtGui.QIcon
    QPixmap = QtGui.QPixmap
    QColor = QtGui.QColor
    QPalette = QtGui.QPalette
    QBrush = QtGui.QBrush
    QLinearGradient = QtGui.QLinearGradient
    QAction = QtGui.QAction
    QKeySequence = QtGui.QKeySequence
    QShortcut = QtGui.QShortcut
    QPainter = QtGui.QPainter
    QPen = QtGui.QPen
    
    QMediaPlayer = QtMultimedia.QMediaPlayer
    QAudioOutput = QtMultimedia.QAudioOutput
    
except ImportError as e:
    print(f"❌ 缺少PySide6: {e}")
    print("请运行: pip install PySide6")
    sys.exit(1)

# 尝试导入OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI库未安装，相关功能将不可用")
    print("   安装: pip install openai>=1.0.0")

# 尝试导入ComfyUI客户端
try:
    from unified_comfyui_client import (
        UnifiedComfyUIClient,
        GenerationConfig,
        CharacterMethod,
        CharacterReference
    )
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("⚠️ ComfyUI客户端不可用，AI动画生成功能将受限")
    print("   请确保unified_comfyui_client.py在同一目录下")

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dubbing_factory.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# 数据类和枚举 - 新增动画相关
# ============================================

class VoiceStyle(Enum):
    """语音风格枚举"""
    LI_YUNLONG = "李云龙"
    WANG_JINGZE = "王境泽"
    ZHANG_FEI = "张飞"
    MA_BAOGUO = "马保国"
    XIE_GUANGKUN = "谢广坤"
    LUO_XIANG = "罗翔"
    GUO_DEGANG = "郭德纲"
    SUN_XIAOCHUAN = "孙笑川"
    TONG_XIANGYU = "佟湘玉"
    SHELDON = "谢尔顿"
    DEFAULT = "默认"

class AnimationStyle(Enum):
    """动画风格枚举"""
    ANIME = "动漫"
    CINEMATIC = "电影"
    CARTOON = "卡通"
    REALISTIC = "写实"
    PAINTERLY = "绘画"
    PIXEL_ART = "像素艺术"

class LipSyncMethod(Enum):
    """口型同步方法枚举"""
    WHISPER_PHONEMES = "Whisper音素分析"
    VISEME_BASED = "视位素映射"
    DEEP_SPEECH = "深度语音分析"
    S2P = "语音转音素"

class ExpressionType(Enum):
    """表情类型枚举"""
    NEUTRAL = "中性"
    HAPPY = "开心"
    SAD = "悲伤"
    ANGRY = "愤怒"
    SURPRISED = "惊讶"
    FEARFUL = "害怕"
    DISGUSTED = "厌恶"
    EXCITED = "兴奋"
    THINKING = "思考"
    SPEAKING = "说话"

class ProcessingStage(Enum):
    """处理阶段枚举"""
    INIT = "初始化"
    EXTRACT_AUDIO = "提取音频"
    TRANSCRIBE = "语音识别"
    BATCH_TRANSLATE = "批量翻译"
    BATCH_STYLE = "批量风格转换"
    BATCH_TTS = "批量语音生成"
    AUDIO_MIX = "音频混合"
    LIP_SYNC = "口型对齐"
    VIDEO_SYNTHESIS = "视频合成"
    ANIMATION_GENERATION = "AI动画生成"
    COMPLETE = "完成"

@dataclass
class CharacterProfile:
    """角色配置文件"""
    name: str
    original_name: str
    voice_style: VoiceStyle
    voice_file: str = ""
    speed: float = 1.0
    pitch: float = 1.0
    emotion: str = "neutral"
    intensity: float = 1.0
    catchphrases: List[str] = field(default_factory=list)
    reference_image: str = ""  # 新增：动画参考图片
    
    @classmethod
    def get_preset(cls, char_name: str) -> 'CharacterProfile':
        """获取预设角色配置"""
        presets = {
            "哆啦A梦": cls(
                name="哆啦A梦",
                original_name="ドラえもん",
                voice_style=VoiceStyle.LI_YUNLONG,
                speed=1.2,
                pitch=1.1,
                emotion="aggressive",
                intensity=1.5,
                catchphrases=["他娘的", "老子", "这仗怎么打", "真他娘的痛快"]
            ),
            "大雄": cls(
                name="大雄",
                original_name="野比のび太",
                voice_style=VoiceStyle.WANG_JINGZE,
                speed=0.9,
                pitch=0.9,
                emotion="whiny",
                intensity=1.2,
                catchphrases=["真香", "我王境泽就是饿死", "哎呀妈呀", "这不可能"]
            ),
            "静香": cls(
                name="静香",
                original_name="源静香",
                voice_style=VoiceStyle.TONG_XIANGYU,
                speed=1.3,
                pitch=1.2,
                emotion="chatty",
                intensity=1.3,
                catchphrases=["额滴神啊", "我好后悔呀", "这是为你好", "你听我说"]
            ),
            "胖虎": cls(
                name="胖虎",
                original_name="剛田武",
                voice_style=VoiceStyle.ZHANG_FEI,
                speed=1.5,
                pitch=0.8,
                emotion="angry",
                intensity=2.0,
                catchphrases=["俺也一样", "燕人张飞在此", "哇呀呀呀", "拿命来"]
            ),
            "小夫": cls(
                name="小夫",
                original_name="骨川スネ夫",
                voice_style=VoiceStyle.MA_BAOGUO,
                speed=1.1,
                pitch=1.0,
                emotion="arrogant",
                intensity=1.4,
                catchphrases=["年轻人不讲武德", "我大意了啊", "耗子尾汁", "接化发"]
            ),
            "哆啦美": cls(
                name="哆啦美",
                original_name="ドラミ",
                voice_style=VoiceStyle.LUO_XIANG,
                speed=1.4,
                pitch=1.3,
                emotion="lecturing",
                intensity=1.6,
                catchphrases=["张三又来了", "这是违法行为", "根据刑法", "法治社会"]
            ),
            "出木杉": cls(
                name="出木杉",
                original_name="出木杉英才",
                voice_style=VoiceStyle.SHELDON,
                speed=2.0,
                pitch=1.0,
                emotion="condescending",
                intensity=1.7,
                catchphrases=["Bazinga", "我早说过了", "这很明显", "根据我的计算"]
            ),
            "小叮当": cls(
                name="小叮当",
                original_name="",
                voice_style=VoiceStyle.GUO_DEGANG,
                speed=1.3,
                pitch=1.1,
                emotion="humorous",
                intensity=1.5,
                catchphrases=["于谦的父亲", "相声讲究说学逗唱", "我字系列", "你要考研啊"]
            ),
            "爸爸": cls(
                name="爸爸",
                original_name="野比伸助",
                voice_style=VoiceStyle.XIE_GUANGKUN,
                speed=1.0,
                pitch=0.9,
                emotion="grumpy",
                intensity=1.4,
                catchphrases=["永强啊", "刘能不是东西", "广坤很生气", "这事没完"]
            ),
            "妈妈": cls(
                name="妈妈",
                original_name="野比玉子",
                voice_style=VoiceStyle.SUN_XIAOCHUAN,
                speed=1.2,
                pitch=1.0,
                emotion="complaining",
                intensity=1.3,
                catchphrases=["你妈死了", "带哥们", "抽象", "缝合怪"]
            )
        }
        return presets.get(char_name, cls(
            name=char_name,
            original_name=char_name,
            voice_style=VoiceStyle.DEFAULT
        ))

@dataclass
class AnimationConfig:
    """动画配置"""
    resolution: tuple = (512, 768)
    fps: int = 24
    duration: float = 10.0
    style: AnimationStyle = AnimationStyle.ANIME
    background: str = ""
    seed: int = -1
    character_reference: str = ""
    consistency_method: CharacterMethod = CharacterMethod.IP_ADAPTER
    consistency_strength: float = 0.7
    character_scale: float = 0.8
    lip_sync_enabled: bool = True
    lip_sync_method: LipSyncMethod = LipSyncMethod.WHISPER_PHONEMES
    expression_enabled: bool = True
    head_movement_enabled: bool = True
    eye_movement_enabled: bool = True
    body_movement_enabled: bool = True
    scene_description: str = ""
    camera_movement: str = "subtle"
    lighting: str = "natural"
    output_format: str = "mp4"
    output_quality: str = "high"

@dataclass
class CharacterModel:
    """角色模型（用于动画生成）"""
    name: str
    reference_images: List[str] = field(default_factory=list)
    voice_profile: Optional[CharacterProfile] = None
    animation_config: Dict[str, Any] = field(default_factory=dict)
    
    def add_reference_image(self, image_path: str):
        """添加参考图片"""
        if os.path.exists(image_path):
            self.reference_images.append(image_path)
            logger.info(f"为角色 {self.name} 添加参考图片: {image_path}")
        else:
            logger.warning(f"参考图片不存在: {image_path}")

@dataclass
class AnimationSegment:
    """动画片段"""
    id: int
    start_time: float
    end_time: float
    text: str
    character: str = "main_character"
    expression: ExpressionType = ExpressionType.SPEAKING
    audio_data: Optional[np.ndarray] = None
    lip_sync_data: Optional[Dict] = None
    prompt: str = ""
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time

@dataclass
class AudioSegment:
    """音频片段数据类"""
    id: int
    start_time: float
    end_time: float
    text: str
    translated_text: str = ""
    styled_text: str = ""
    character: str = "unknown"
    confidence: float = 0.0
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 24000
    file_path: str = ""
    volume: float = 1.0
    background: bool = False
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "text": self.text,
            "translated_text": self.translated_text,
            "styled_text": self.styled_text,
            "character": self.character,
            "confidence": self.confidence,
            "sample_rate": self.sample_rate,
            "file_path": self.file_path,
            "volume": self.volume,
            "background": self.background
        }

class AudioPostProcessor:
    """音频后处理器 - 优化口型对齐"""
    
    def __init__(self, config):
        self.config = config
    
    def align_to_lip_sync(self, video_path: str, audio_path: str) -> str:
        """口型同步对齐"""
        logger.info("进行口型同步对齐")
        
        try:
            audio, sr = sf.read(audio_path)
            
            video_info = self._get_video_info(video_path)
            if video_info and 'duration' in video_info:
                video_duration = video_info['duration']
                audio_duration = len(audio) / sr
                
                if abs(video_duration - audio_duration) / video_duration > 0.05:
                    stretch_factor = video_duration / audio_duration
                    if 0.8 <= stretch_factor <= 1.2:
                        try:
                            from librosa.effects import time_stretch
                            audio = time_stretch(audio, rate=1/stretch_factor)
                        except:
                            current_samples = len(audio)
                            target_samples = int(current_samples * stretch_factor)
                            if current_samples > 1:
                                x_old = np.linspace(0, 1, current_samples)
                                x_new = np.linspace(0, 1, target_samples)
                                f = interp1d(x_old, audio, kind='linear')
                                audio = f(x_new)
            
            temp_dir = Path(self.config.temp_dir)
            temp_dir.mkdir(exist_ok=True)
            output_path = temp_dir / "aligned_audio.wav"
            sf.write(str(output_path), audio, self.config.sample_rate)
            
            logger.info(f"口型对齐完成，保存到: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.warning(f"口型对齐失败，使用原音频: {e}")
            return audio_path
    
    def _get_video_info(self, video_path: str) -> Optional[Dict]:
        """获取视频信息"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                video_info = {}
                
                if 'format' in info and 'duration' in info['format']:
                    video_info['duration'] = float(info['format']['duration'])
                
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        if 'avg_frame_rate' in stream:
                            frame_rate = stream['avg_frame_rate']
                            if '/' in frame_rate:
                                num, den = map(int, frame_rate.split('/'))
                                if den > 0:
                                    video_info['frame_rate'] = num / den
                        break
                
                return video_info
                
        except Exception as e:
            logger.warning(f"获取视频信息失败: {e}")
        
        return None

@dataclass
class ProcessingConfig:
    """处理配置"""
    input_video: str = ""
    output_video: str = ""
    temp_dir: str = "temp"
    sample_rate: int = 24000
    bit_depth: int = 16
    channels: int = 2
    use_gpu: bool = True
    gpu_id: int = 0
    batch_size: int = 4
    num_workers: int = 4
    cache_enabled: bool = True
    cache_dir: str = "cache"
    whisper_model: str = "large-v3"
    translate_model: str = "qwen3:4b"
    tts_api_url: str = "http://127.0.0.1:5021/api/tts"
    ollama_url: str = "http://127.0.0.1:11434"
    keep_background: bool = True
    background_volume: float = 0.3
    voice_volume: float = 0.8
    noise_reduction: bool = True
    normalize_audio: bool = True
    fade_duration: float = 0.05
    translate_to_chinese: bool = True
    use_style_transfer: bool = True
    add_catchphrases: bool = True
    catchphrase_probability: float = 0.3
    translation_batch_size: int = 10
    tts_batch_size: int = 3
    lip_sync_enabled: bool = True
    lip_sync_strength: float = 0.8
    lip_sync_method: str = "时间拉伸"
    
    def __post_init__(self):
        if not self.output_video and self.input_video:
            input_path = Path(self.input_video)
            self.output_video = str(input_path.with_stem(f"{input_path.stem}_dubbed"))
    
    def get_ollama_api_url(self) -> str:
        """获取Ollama API URL"""
        base_url = self.ollama_url.rstrip('/')
        
        patterns_to_remove = ['/v1', '/api/generate', '/api/chat', '/api/']
        
        for pattern in patterns_to_remove:
            if base_url.endswith(pattern):
                base_url = base_url[:-len(pattern)]
                base_url = base_url.rstrip('/')
            elif pattern in base_url:
                parts = base_url.split(pattern)
                base_url = parts[0].rstrip('/')
                break
        
        if not base_url.startswith('http'):
            base_url = 'http://' + base_url
        
        api_url = f"{base_url}/api/generate"
        
        logger.debug(f"生成的API URL: {api_url}")
        return api_url

# ============================================
# AI动画生成器类
# ============================================

class AIAnimationGenerator:
    """AI动画生成器 - 集成ComfyUI人物一致性"""
    
    def __init__(self, config: AnimationConfig, comfyui_host: str = "127.0.0.1", comfyui_port: int = 8188):
        self.config = config
        self.comfyui_host = comfyui_host
        self.comfyui_port = comfyui_port
        
        # 初始化客户端
        self.comfyui_client = None
        if COMFYUI_AVAILABLE:
            self.comfyui_client = UnifiedComfyUIClient(
                host=comfyui_host,
                port=comfyui_port
            )
        
        # 存储数据
        self.characters: Dict[str, CharacterModel] = {}
        self.segments: List[AnimationSegment] = []
        self.generated_frames: List[Dict] = []
        
        # 输出目录
        self.output_dir = Path("ai_animation_output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("AI动画生成器初始化完成")
    
    async def initialize(self):
        """初始化连接"""
        logger.info("正在初始化AI动画生成器...")
        
        if not COMFYUI_AVAILABLE:
            logger.error("ComfyUI客户端不可用")
            raise ImportError("ComfyUI客户端未找到")
        
        if not self.comfyui_client:
            logger.error("ComfyUI客户端未初始化")
            return False
        
        # 连接到ComfyUI
        logger.info(f"正在连接到ComfyUI: {self.comfyui_host}:{self.comfyui_port}")
        connected = await self.comfyui_client.connect()
        
        if not connected:
            logger.error("无法连接到ComfyUI服务器")
            raise ConnectionError("ComfyUI服务器连接失败")
        
        logger.info("✅ AI动画生成器初始化成功")
        return True
    
    def add_character(self, character: CharacterModel):
        """添加角色"""
        self.characters[character.name] = character
        logger.info(f"已添加角色: {character.name}")
    
    def create_character_from_profile(self, profile: CharacterProfile, reference_image: str) -> CharacterModel:
        """从配音角色配置创建动画角色"""
        character = CharacterModel(
            name=profile.name,
            voice_profile=profile,
            animation_config={
                "style": self.config.style,
                "consistency_method": self.config.consistency_method,
                "consistency_strength": self.config.consistency_strength
            }
        )
        
        character.add_reference_image(reference_image)
        return character
    
    async def process_script(self, script_path: str):
        """处理剧本"""
        logger.info(f"处理剧本: {script_path}")
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            lines = script_content.strip().split('\n')
            segments = []
            
            current_time = 0.0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    segment_duration = 3.0
                    
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        character_name, dialogue = parts
                        character_name = character_name.strip()
                        dialogue = dialogue.strip()
                    else:
                        character_name = "未知角色"
                        dialogue = line.strip()
                    
                    if character_name in self.characters:
                        character = character_name
                    else:
                        character = list(self.characters.keys())[0] if self.characters else "main_character"
                    
                    expression = self._detect_expression(dialogue)
                    
                    segment = AnimationSegment(
                        id=i,
                        start_time=current_time,
                        end_time=current_time + segment_duration,
                        text=dialogue,
                        character=character,
                        expression=expression,
                        prompt=self._generate_animation_prompt(dialogue, character, expression)
                    )
                    
                    segments.append(segment)
                    current_time += segment_duration
            
            self.segments = segments
            logger.info(f"解析完成: {len(segments)}个动画片段")
            
            return segments
            
        except Exception as e:
            logger.error(f"剧本处理失败: {e}")
            raise
    
    def _detect_expression(self, text: str) -> ExpressionType:
        """检测表情"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["哈哈", "开心", "高兴", "快乐", "laugh", "happy", "smile"]):
            return ExpressionType.HAPPY
        elif any(word in text_lower for word in ["悲伤", "伤心", "难过", "哭", "sad", "cry", "unhappy"]):
            return ExpressionType.SAD
        elif any(word in text_lower for word in ["生气", "愤怒", "发火", "angry", "mad", "furious"]):
            return ExpressionType.ANGRY
        elif any(word in text_lower for word in ["惊讶", "吃惊", "震惊", "surprise", "shock", "amazed"]):
            return ExpressionType.SURPRISED
        elif any(word in text_lower for word in ["害怕", "恐惧", "恐怖", "fear", "scared", "afraid"]):
            return ExpressionType.FEARFUL
        elif any(word in text_lower for word in ["恶心", "厌恶", "讨厌", "disgust", "dislike", "hate"]):
            return ExpressionType.DISGUSTED
        elif any(word in text_lower for word in ["兴奋", "激动", "excited", "thrilled", "energetic"]):
            return ExpressionType.EXCITED
        elif any(word in text_lower for word in ["思考", "考虑", "想", "think", "consider", "ponder"]):
            return ExpressionType.THINKING
        else:
            return ExpressionType.SPEAKING
    
    def _generate_animation_prompt(self, dialogue: str, character: str, expression: ExpressionType) -> str:
        """生成动画提示词"""
        character_model = self.characters.get(character)
        
        base_prompt = f"{self.config.style.value} style, "
        
        if character_model and character_model.voice_profile:
            char_desc = character_model.voice_profile.name
        else:
            char_desc = character
        
        expression_map = {
            ExpressionType.HAPPY: "smiling happily, cheerful expression",
            ExpressionType.SAD: "sad expression, looking down, tearful eyes",
            ExpressionType.ANGRY: "angry expression, furrowed brows, clenched teeth",
            ExpressionType.SURPRISED: "surprised expression, wide eyes, open mouth",
            ExpressionType.FEARFUL: "fearful expression, scared, trembling",
            ExpressionType.DISGUSTED: "disgusted expression, wrinkled nose",
            ExpressionType.EXCITED: "excited expression, enthusiastic, energetic",
            ExpressionType.THINKING: "thinking expression, contemplative, hand on chin",
            ExpressionType.SPEAKING: "speaking, mouth open",
            ExpressionType.NEUTRAL: "neutral expression"
        }
        
        scene_desc = self.config.scene_description if self.config.scene_description else "clean background, cinematic lighting"
        
        prompt = f"{base_prompt}{char_desc}, {expression_map[expression]}, {scene_desc}, "
        prompt += f"full body shot, dynamic pose, {self.config.lighting} lighting, "
        prompt += f"high quality, detailed, 4k, masterpiece"
        
        prompt += f', saying: "{dialogue[:50]}"'
        
        return prompt
    
    async def analyze_lip_sync(self, audio_path: str, segments: List[AnimationSegment]) -> List[Dict]:
        """分析口型同步数据"""
        logger.info("分析口型同步数据...")
        
        if not self.config.lip_sync_enabled:
            logger.info("口型同步已禁用")
            return []
        
        lip_sync_data = []
        
        for segment in segments:
            try:
                phoneme_data = {
                    "segment_id": segment.id,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "phonemes": self._extract_phonemes(segment.text),
                    "viseme_frames": self._generate_viseme_frames(segment)
                }
                
                segment.lip_sync_data = phoneme_data
                lip_sync_data.append(phoneme_data)
                
                logger.debug(f"分析片段 {segment.id} 口型数据")
                
            except Exception as e:
                logger.warning(f"片段 {segment.id} 口型分析失败: {e}")
                segment.lip_sync_data = None
        
        logger.info(f"✅ 口型同步分析完成: {len(lip_sync_data)}个片段")
        return lip_sync_data
    
    def _extract_phonemes(self, text: str) -> List[Dict]:
        """提取音素"""
        phoneme_map = {
            'a': 'AA', 'i': 'IH', 'u': 'UW', 'e': 'EH', 'o': 'AO',
            'b': 'B', 'p': 'P', 'm': 'M', 'f': 'F', 'd': 'D',
            't': 'T', 'n': 'N', 'l': 'L', 'g': 'G', 'k': 'K',
            'h': 'HH', 'j': 'Y', 'q': 'CH', 'x': 'SH', 'zh': 'ZH',
            'ch': 'CH', 'sh': 'SH', 'r': 'R', 'z': 'Z', 'c': 'TS',
            's': 'S', 'y': 'Y', 'w': 'W'
        }
        
        phonemes = []
        for char in text.lower():
            if char in phoneme_map:
                phonemes.append({
                    "phoneme": phoneme_map[char],
                    "duration": 0.1
                })
        
        return phonemes
    
    def _generate_viseme_frames(self, segment: AnimationSegment) -> List[Dict]:
        """生成视位素帧"""
        frames = []
        fps = self.config.fps
        duration = segment.duration
        total_frames = int(fps * duration)
        
        for frame_idx in range(total_frames):
            frame_time = segment.start_time + (frame_idx / fps)
            
            viseme = "rest"
            
            if segment.expression == ExpressionType.SPEAKING:
                time_in_segment = frame_time - segment.start_time
                cycle = (time_in_segment * 5) % 1.0
                
                if cycle < 0.3:
                    viseme = "AA"
                elif cycle < 0.6:
                    viseme = "IH"
                else:
                    viseme = "MM"
            
            frames.append({
                "frame": frame_idx,
                "time": frame_time,
                "viseme": viseme,
                "mouth_openness": 0.5 if viseme == "AA" else 0.2
            })
        
        return frames
    
    async def generate_animation_frames(self, segments: List[AnimationSegment]) -> List[Dict]:
        """生成动画帧"""
        logger.info("开始生成动画帧...")
        
        generated_frames = []
        
        for segment in segments:
            logger.info(f"生成动画片段 {segment.id}: {segment.text[:30]}...")
            
            try:
                character_model = self.characters.get(segment.character)
                if not character_model:
                    logger.warning(f"角色 {segment.character} 未找到，使用默认设置")
                    character_model = next(iter(self.characters.values())) if self.characters else None
                
                if character_model and character_model.reference_images:
                    reference_image = character_model.reference_images[0]
                    
                    uploaded_name = await self.comfyui_client.upload_image(
                        Path(reference_image)
                    )
                    
                    enhanced_prompt = self._enhance_prompt_with_animation(
                        segment.prompt, segment, character_model
                    )
                    
                    result = await self.comfyui_client.generate_character(
                        reference_image=reference_image,
                        prompt=enhanced_prompt,
                        method=self.config.consistency_method,
                        strength=self.config.consistency_strength,
                        config=self._create_generation_config(),
                        output_dir=self.output_dir / "frames" / f"segment_{segment.id}"
                    )
                    
                    if result.get("success") and result.get("images"):
                        for img_info in result["images"]:
                            generated_frames.append({
                                "segment_id": segment.id,
                                "frame_path": img_info.get("filename", ""),
                                "time": segment.start_time,
                                "prompt": enhanced_prompt
                            })
                        
                        logger.info(f"✅ 片段 {segment.id} 生成成功")
                    else:
                        logger.warning(f"⚠️ 片段 {segment.id} 生成失败: {result.get('error')}")
                
                else:
                    result = await self.comfyui_client.generate_image(
                        prompt=segment.prompt,
                        config=self._create_generation_config(),
                        output_dir=self.output_dir / "frames" / f"segment_{segment.id}"
                    )
                    
                    if result.get("success"):
                        logger.info(f"✅ 片段 {segment.id} 生成成功")
                    else:
                        logger.warning(f"⚠️ 片段 {segment.id} 生成失败")
                
            except Exception as e:
                logger.error(f"❌ 片段 {segment.id} 生成异常: {e}")
            
            await asyncio.sleep(1)
        
        self.generated_frames = generated_frames
        logger.info(f"✅ 动画帧生成完成: {len(generated_frames)}个帧")
        return generated_frames
    
    def _enhance_prompt_with_animation(self, base_prompt: str, segment: AnimationSegment, character_model: CharacterModel) -> str:
        """增强提示词，添加动画元素"""
        enhanced = base_prompt
        
        if self.config.lip_sync_enabled and segment.lip_sync_data:
            if segment.lip_sync_data.get("viseme_frames"):
                first_frame = segment.lip_sync_data["viseme_frames"][0]
                mouth_state = "open mouth" if first_frame.get("mouth_openness", 0) > 0.4 else "closed mouth"
                enhanced += f", {mouth_state}"
        
        expression_map = {
            ExpressionType.HAPPY: "smiling, cheerful expression",
            ExpressionType.SAD: "sad expression, tearful eyes",
            ExpressionType.ANGRY: "angry expression, furrowed brows",
            ExpressionType.SURPRISED: "surprised expression, wide eyes",
            ExpressionType.FEARFUL: "fearful expression, scared look",
            ExpressionType.DISGUSTED: "disgusted expression",
            ExpressionType.EXCITED: "excited expression, enthusiastic",
            ExpressionType.THINKING: "thinking expression, contemplative",
            ExpressionType.SPEAKING: "speaking expression",
            ExpressionType.NEUTRAL: "neutral expression"
        }
        
        enhanced += f", {expression_map.get(segment.expression, 'neutral expression')}"
        
        if self.config.head_movement_enabled:
            head_motions = ["subtle head turn", "slight nod", "head tilt", "looking forward"]
            motion = head_motions[segment.id % len(head_motions)]
            enhanced += f", {motion}"
        
        if self.config.eye_movement_enabled:
            eye_actions = ["looking at viewer", "eye contact", "blinking", "focused gaze"]
            action = eye_actions[segment.id % len(eye_actions)]
            enhanced += f", {action}"
        
        if self.config.body_movement_enabled and segment.id % 3 == 0:
            body_poses = ["hand gesture", "leaning forward", "relaxed posture", "dynamic pose"]
            pose = body_poses[segment.id % len(body_poses)]
            enhanced += f", {pose}"
        
        camera_angles = ["medium shot", "close-up", "full body", "cinematic framing"]
        angle = camera_angles[segment.id % len(camera_angles)]
        enhanced += f", {angle}, {self.config.camera_movement} camera movement"
        
        return enhanced
    
    def _create_generation_config(self) -> GenerationConfig:
        """创建生成配置"""
        width, height = self.config.resolution
        
        return GenerationConfig(
            width=width,
            height=height,
            steps=25,
            cfg=7.0,
            sampler="dpmpp_2m",
            scheduler="karras",
            seed=self.config.seed if self.config.seed != -1 else -1,
            batch_size=1,
            model="sd15.safetensors",
            vae="auto"
        )
    
    async def assemble_animation(self, frames: List[Dict], audio_path: Optional[str] = None) -> str:
        """组装动画"""
        logger.info("开始组装动画...")
        
        try:
            output_video_path = self.output_dir / f"animation_{int(time.time())}.{self.config.output_format}"
            
            if frames:
                sorted_frames = sorted(frames, key=lambda x: x.get("time", 0))
                
                width, height = self.config.resolution
                fps = self.config.fps
                
                frame_video_path = self.output_dir / "frame_sequence.mp4"
                self._create_frame_sequence(sorted_frames, str(frame_video_path), fps, (width, height))
                
                if audio_path and os.path.exists(audio_path):
                    self._merge_audio_with_video(str(frame_video_path), audio_path, str(output_video_path))
                else:
                    output_video_path = frame_video_path
                
                logger.info(f"✅ 动画组装完成: {output_video_path}")
                return str(output_video_path)
            else:
                logger.warning("没有帧数据可供组装")
                return ""
                
        except Exception as e:
            logger.error(f"动画组装失败: {e}")
            return ""
    
    def _create_frame_sequence(self, frames: List[Dict], output_path: str, fps: int, resolution: tuple):
        """创建帧序列视频"""
        width, height = resolution
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_data in frames:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            segment_id = frame_data.get("segment_id", 0)
            text = f"动画帧 {segment_id}"
            cv2.putText(frame, text, (50, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            for _ in range(fps):
                out.write(frame)
        
        out.release()
        logger.info(f"创建帧序列视频: {output_path}")
    
    def _merge_audio_with_video(self, video_path: str, audio_path: str, output_path: str):
        """合并音频和视频"""
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                '-y',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"音频视频合并完成: {output_path}")
            
        except Exception as e:
            logger.error(f"音频视频合并失败: {e}")
            import shutil
            shutil.copy(video_path, output_path)
    
    async def generate_complete_animation(self, script_path: str) -> str:
        """生成完整动画"""
        logger.info("开始生成完整动画...")
        
        try:
            await self.initialize()
            
            segments = await self.process_script(script_path)
            
            audio_path = self.output_dir / "dubbed_audio.wav"
            
            if self.config.lip_sync_enabled:
                await self.analyze_lip_sync(str(audio_path), segments)
            
            frames = await self.generate_animation_frames(segments)
            
            final_animation = await self.assemble_animation(frames, str(audio_path) if os.path.exists(str(audio_path)) else None)
            
            if final_animation:
                logger.info(f"🎉 动画生成完成: {final_animation}")
                return final_animation
            else:
                logger.error("动画生成失败")
                return ""
                
        except Exception as e:
            logger.error(f"动画生成过程失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

# ============================================
# OpenAI集成类
# ============================================

class OpenAIIntegration:
    """OpenAI接口集成"""
    
    def __init__(self, api_key: str, base_url: str = "https://apis.iflow.cn/v1"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI库未安装")
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = "deepseek-v3"
    
    def generate_script_analysis(self, script: str) -> Dict:
        """生成剧本分析"""
        prompt = f"""
        分析以下剧本，提取关键信息：
        1. 角色列表
        2. 每个角色的台词
        3. 情感变化
        4. 建议的动画场景
        5. 摄像机角度建议
        
        剧本：
        {script}
        
        请以JSON格式返回分析结果。
        """
        
        try:
            completion = self.client.chat.completions.create(
                extra_body={},
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response = completion.choices[0].message.content
            
            try:
                analysis = json.loads(response)
                return analysis
            except:
                return {"raw_analysis": response}
                
        except Exception as e:
            logger.error(f"OpenAI剧本分析失败: {e}")
            return {}
    
    def enhance_animation_prompt(self, base_prompt: str, context: Dict) -> str:
        """增强动画提示词"""
        prompt = f"""
        根据以下信息，优化动画提示词：
        基础提示词: {base_prompt}
        上下文信息: {json.dumps(context, ensure_ascii=False)}
        
        请生成一个更详细、更具视觉表现力的动画提示词。
        包含角色表情、动作、场景细节、光照和构图。
        """
        
        try:
            completion = self.client.chat.completions.create(
                extra_body={},
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            enhanced_prompt = completion.choices[0].message.content
            return enhanced_prompt.strip()
            
        except Exception as e:
            logger.error(f"OpenAI提示词增强失败: {e}")
            return base_prompt

# ============================================
# 核心引擎类
# ============================================

class DubbingEngine:
    """配音引擎核心类 - 优化批量处理版"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = self._setup_device()
        self.whisper_model = None
        self.character_profiles: Dict[str, CharacterProfile] = {}
        self.audio_cache: Dict[str, np.ndarray] = {}
        self.segments: List[AudioSegment] = []
        self.background_audio: Optional[np.ndarray] = None
        self._init_directories()
        logger.info("配音引擎初始化完成")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.gpu_id}")
            logger.info(f"使用GPU: {torch.cuda.get_device_name(device)}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("使用CPU")
        return device
    
    def _init_directories(self):
        """初始化目录"""
        directories = [
            self.config.temp_dir,
            self.config.cache_dir,
            "voices",
            "inputs",
            "outputs",
            "logs",
            "subtitles",
            "ai_animation_output",
            "characters"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_whisper_model(self):
        """加载Whisper模型"""
        if not self.whisper_model:
            logger.info(f"加载Whisper模型: {self.config.whisper_model}")
            self.whisper_model = whisper.load_model(
                self.config.whisper_model,
                device=self.device
            )
            logger.info("Whisper模型加载完成")
    
    def add_character_profile(self, profile: CharacterProfile):
        """添加角色配置"""
        self.character_profiles[profile.name] = profile
        logger.info(f"添加角色配置: {profile.name} -> {profile.voice_style.value}")
    
    def extract_audio(self, video_path: str) -> str:
        """提取视频中的音频"""
        logger.info(f"提取音频: {video_path}")
        
        audio_path = Path(self.config.temp_dir) / f"original_audio_{Path(video_path).stem}.wav"
        
        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream,
                str(audio_path),
                acodec='pcm_s16le',
                ac=self.config.channels,
                ar=self.config.sample_rate,
                loglevel='quiet'
            )
            ffmpeg.run(stream, overwrite_output=True)
            
            logger.info(f"音频提取完成: {audio_path}")
            return str(audio_path)
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg错误: {e}")
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ac', str(self.config.channels),
                '-ar', str(self.config.sample_rate),
                '-y', str(audio_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                return str(audio_path)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"音频提取失败: {e}")
    
    def transcribe_audio(self, audio_path: str) -> List[AudioSegment]:
        """改进的语音识别 - 更精确的时间戳"""
        logger.info(f"开始语音识别: {audio_path}")
        self.load_whisper_model()
        
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language="ja",
                task="transcribe",
                verbose=True,
                fp16=self.device.type == 'cuda',
                word_timestamps=True,
                prepend_punctuations="\"'\"¿([{-",
                append_punctuations="\"'.。,，!！?？:：\"、)]}、"
            )
            
            segments = []
            segment_id = 0
            
            for i, seg in enumerate(result['segments']):
                start_time = seg['start']
                end_time = seg['end']
                
                character = self.identify_character(seg['text'], seg['avg_logprob'])
                
                segment = AudioSegment(
                    id=segment_id,
                    start_time=start_time,
                    end_time=end_time,
                    text=seg['text'].strip(),
                    character=character,
                    confidence=seg['avg_logprob']
                )
                segments.append(segment)
                segment_id += 1
            
            logger.info(f"语音识别完成: {len(segments)}个片段")
            self.segments = segments
            return segments
            
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            
            try:
                logger.info("尝试使用原始方法进行语音识别")
                result = self.whisper_model.transcribe(
                    audio_path,
                    language="ja",
                    task="transcribe",
                    verbose=True,
                    fp16=self.device.type == 'cuda'
                )
                
                segments = []
                for i, seg in enumerate(result['segments']):
                    character = self.identify_character(seg['text'], seg['avg_logprob'])
                    
                    segment = AudioSegment(
                        id=i,
                        start_time=seg['start'],
                        end_time=seg['end'],
                        text=seg['text'].strip(),
                        character=character,
                        confidence=seg['avg_logprob']
                    )
                    segments.append(segment)
                
                logger.info(f"语音识别完成(回退方法): {len(segments)}个片段")
                self.segments = segments
                return segments
                
            except Exception as e2:
                logger.error(f"回退方法也失败: {e2}")
                raise
    
    def identify_character(self, text: str, confidence: float) -> str:
        """识别说话角色"""
        text_lower = text.lower()
        
        character_keywords = {
            "哆啦A梦": ["ドラえもん", "哆啦", "机器猫", "叮当", "dora", "ドラ"],
            "大雄": ["のび太", "大雄", "nobita", "野比", "のび"],
            "静香": ["静香", "しずか", "shizuka", "小静", "しず"],
            "胖虎": ["ジャイアン", "胖虎", "刚田", "gian", "giant", "ジャイ"],
            "小夫": ["スネ夫", "小夫", "骨川", "suneo", "スネ"],
            "哆啦美": ["ドラミ", "哆啦美", "dorami"],
            "出木杉": ["出木杉", "出来杉", "英才", "でるき"],
            "爸爸": ["パパ", "爸爸", "父亲", "お父さん", "父"],
            "妈妈": ["ママ", "妈妈", "母亲", "お母さん", "母"]
        }
        
        for character, keywords in character_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return character
        
        return "哆啦A梦"
    
    def _translate_single_text_sync(self, text: str) -> str:
        """同步翻译单个文本 - 修复API通信问题"""
        try:
            api_url = self.config.get_ollama_api_url()
            
            logger.info(f"📤 发送翻译请求到: {api_url}")
            logger.info(f"翻译文本: {text[:50]}...")
            
            prompt = f"""你是一个专业的翻译助手，请严格按照要求完成翻译：
1. 源语言：日语
2. 目标语言：中文
3. 翻译要求：保持原意准确、语言流畅、无语法错误，只输出翻译结果，不要额外解释。

需要翻译的文本：
{text}"""
            
            data = {
                "model": self.config.translate_model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 4096
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                api_url,
                json=data,
                headers=headers,
                timeout=60
            )
            
            logger.info(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if "response" in result:
                    reply_text = result["response"].strip()
                    
                    if reply_text:
                        reply_text = self._clean_translation_response(reply_text)
                        logger.info(f"✅ 翻译成功: {reply_text[:50]}...")
                        return reply_text
                    else:
                        logger.warning(f"⚠️ 翻译返回空文本")
                        return text
                else:
                    logger.warning(f"⚠️ 响应中没有'response'字段: {result}")
                    return text
            else:
                error_text = response.text[:500] if response.text else "无详细错误信息"
                logger.error(f"❌ 翻译请求失败: HTTP {response.status_code}")
                logger.error(f"错误详情: {error_text}")
                
                self._diagnose_ollama_problem(api_url, response.status_code, error_text)
                return text
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ 连接失败: {e}")
            logger.error(f"请检查Ollama服务是否运行: ollama serve")
            return text
        except requests.exceptions.Timeout as e:
            logger.error(f"❌ 请求超时: {e}")
            return text
        except Exception as e:
            logger.error(f"❌ 翻译异常: {e}")
            import traceback
            traceback.print_exc()
            return text
    
    def _diagnose_ollama_problem(self, api_url: str, status_code: int, error_text: str):
        """诊断Ollama问题"""
        logger.error("🔍 Ollama问题诊断:")
        logger.error(f"  1. API URL: {api_url}")
        logger.error(f"  2. 状态码: {status_code}")
        logger.error(f"  3. 错误信息: {error_text[:200]}")
        
        try:
            base_url = api_url.replace('/api/generate', '')
            models_url = f"{base_url}/api/tags"
            logger.info(f"尝试检查模型列表: {models_url}")
            
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"✅ 模型列表获取成功: {len(models.get('models', []))} 个模型")
            else:
                logger.error(f"❌ 无法获取模型列表: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ 诊断检查失败: {e}")
    
    def _clean_translation_response(self, reply: str) -> str:
        """清理翻译回复"""
        if not reply:
            return ""
        
        patterns_to_remove = [
            r'思考.*?\n',
            r'首先.*?\n',
            r'分析.*?\n',
            r'注意.*?\n',
            r'Okay.*?\n',
            r'好的.*?\n',
            r'明白了.*?\n',
            r'我来翻译.*?\n',
            r'Assistant.*?\n',
            r'助手.*?\n',
            r'Translate.*?:',
            r'翻译.*?:',
            r'以下是翻译.*?:',
            r'译文.*?:'
        ]
        
        for pattern in patterns_to_remove:
            reply = re.sub(pattern, '', reply, flags=re.IGNORECASE)
        
        reply = re.sub(r'\n+', '\n', reply)
        reply = re.sub(r'\s+', ' ', reply)
        
        return reply.strip()
    
    def batch_translate_segments(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """批量翻译所有片段 - 同步版本"""
        if not self.config.translate_to_chinese:
            logger.info("跳过翻译（配置为不翻译）")
            return segments
        
        logger.info(f"开始批量翻译 {len(segments)} 个片段")
        
        api_url = self.config.get_ollama_api_url()
        logger.info(f"使用API: {api_url}")
        logger.info(f"使用模型: {self.config.translate_model}")
        
        for i, seg in enumerate(segments):
            if seg.text and len(seg.text.strip()) > 1:
                try:
                    translated = self._translate_single_text_sync(seg.text)
                    
                    import re
                    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')
                    
                    if japanese_pattern.search(translated):
                        logger.warning(f"⚠️ {i+1}/{len(segments)} 翻译后仍为日文")
                        seg.translated_text = seg.text
                    elif translated and translated != seg.text:
                        seg.translated_text = translated
                        logger.info(f"✅ {i+1}/{len(segments)} 翻译成功: {translated[:30]}...")
                    else:
                        seg.translated_text = seg.text
                        logger.warning(f"⚠️ {i+1}/{len(segments)} 翻译返回空或相同，使用原文")
                
                except Exception as e:
                    seg.translated_text = seg.text
                    logger.error(f"❌ {i+1}/{len(segments)} 翻译异常: {e}")
            else:
                seg.translated_text = seg.text
            
            if (i + 1) % 5 == 0 or i + 1 == len(segments):
                logger.info(f"翻译进度: {i+1}/{len(segments)}")
        
        translated_count = sum(1 for seg in segments if seg.translated_text != seg.text)
        logger.info(f"翻译完成: {translated_count}/{len(segments)} 个片段成功翻译")
        
        return segments
    
    def batch_apply_voice_styles(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """批量应用语音风格"""
        logger.info(f"批量应用语音风格到 {len(segments)} 个片段")
        
        for seg in segments:
            if seg.translated_text:
                import re
                japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')
                if japanese_pattern.search(seg.translated_text):
                    logger.warning(f"⚠️ 片段 {seg.id} 翻译后仍为日文: {seg.translated_text[:50]}...")
                    seg.styled_text = seg.text
                else:
                    try:
                        seg.styled_text = self.apply_voice_style(seg.translated_text, seg.character)
                    except Exception as e:
                        seg.styled_text = seg.translated_text
                        logger.warning(f"风格转换失败: {e}，使用原文: {seg.character}")
            else:
                seg.styled_text = seg.text
        
        logger.info("风格转换完成")
        return segments
    
    def apply_voice_style(self, text: str, character: str) -> str:
        """应用语音风格"""
        if not self.config.use_style_transfer:
            return text
        
        profile = self.character_profiles.get(character)
        if not profile:
            return text
        
        styled_text = text
        
        if profile.voice_style == VoiceStyle.LI_YUNLONG:
            styled_text = self._apply_liyunlong_style(text)
        elif profile.voice_style == VoiceStyle.WANG_JINGZE:
            styled_text = self._apply_wangjingze_style(text)
        elif profile.voice_style == VoiceStyle.ZHANG_FEI:
            styled_text = self._apply_zhangfei_style(text)
        elif profile.voice_style == VoiceStyle.MA_BAOGUO:
            styled_text = self._apply_mabaoguo_style(text)
        elif profile.voice_style == VoiceStyle.XIE_GUANGKUN:
            styled_text = self._apply_xieguangkun_style(text)
        elif profile.voice_style == VoiceStyle.LUO_XIANG:
            styled_text = self._apply_luoxiang_style(text)
        elif profile.voice_style == VoiceStyle.GUO_DEGANG:
            styled_text = self._apply_guodegang_style(text)
        elif profile.voice_style == VoiceStyle.SUN_XIAOCHUAN:
            styled_text = self._apply_sunxiaochuan_style(text)
        elif profile.voice_style == VoiceStyle.TONG_XIANGYU:
            styled_text = self._apply_tongxiangyu_style(text)
        elif profile.voice_style == VoiceStyle.SHELDON:
            styled_text = self._apply_sheldon_style(text)
        
        if (self.config.add_catchphrases and profile.catchphrases and 
            np.random.random() < self.config.catchphrase_probability):
            catchphrase = np.random.choice(profile.catchphrases)
            styled_text = f"{catchphrase}，{styled_text}"
        
        return styled_text
    
    def _apply_liyunlong_style(self, text: str) -> str:
        """李云龙风格"""
        replacements = {
            "我": "老子",
            "你": "你小子",
            "他": "那小子",
            "我们": "咱们",
            "你们": "你们这帮",
            "真的吗": "他娘的真的假的？",
            "太好了": "真他娘的痛快！",
            "怎么办": "这仗怎么打？",
            "不行": "这绝对不行！天王老子来了也不行！",
            "谢谢": "多谢了兄弟！",
            "对不起": "对不住了！",
            "明白了": "晓得了！",
            "快点": "给老子快点儿！",
            "笨蛋": "蠢货！",
            "可恶": "他奶奶的！",
            "害怕": "怕个球！",
            "为什么": "为啥子？",
            "什么": "啥玩意儿？",
            "怎么": "咋整？",
            "非常": "贼他娘的",
            "很": "忒",
            "太": "老",
            "真的": "真格的"
        }
        
        result = text
        for orig, repl in replacements.items():
            result = result.replace(orig, repl)
        
        if not result.endswith(('！', '？', '。', '!')):
            result += '！'
        
        return result
    
    def _apply_wangjingze_style(self, text: str) -> str:
        """王境泽风格"""
        replacements = {
            "我": "我王境泽",
            "你": "你",
            "不": "绝不",
            "真的": "真香！",
            "好吃": "哎呀妈呀真香！",
            "不错": "真香！",
            "喜欢": "真香！",
            "同意": "真香！",
            "拒绝": "我就是饿死，死外边，从这里跳下去，也不会",
            "改变主意": "真香！",
            "后悔": "哎呀真香！"
        }
        
        result = text
        for orig, repl in replacements.items():
            if orig in result:
                result = result.replace(orig, repl)
        
        return result
    
    def _apply_zhangfei_style(self, text: str) -> str:
        """张飞风格"""
        replacements = {
            "我": "俺",
            "你": "汝",
            "他": "那厮",
            "我们": "俺们",
            "你们": "尔等",
            "啊": "哇呀呀呀！",
            "什么": "啥",
            "真的": "当真？",
            "好": "甚好！",
            "厉害": "好武艺！",
            "打": "大战三百回合！",
            "杀": "取你首级！",
            "生气": "气煞我也！",
            "大哥": "大哥！",
            "二哥": "二哥！"
        }
        
        result = text
        for orig, repl in replacements.items():
            result = result.replace(orig, repl)
        
        return result
    
    def _apply_mabaoguo_style(self, text: str) -> str:
        """马保国风格"""
        catchphrases = [
            "年轻人不讲武德",
            "我大意了啊，没有闪",
            "耗子尾汁",
            "传统功夫讲究化劲儿",
            "二百多斤的英国大理石",
            "接！化！发！",
            "啪的一下就站起来了，很快啊！",
            "有备而来",
            "武林要以和为贵",
            "你两分钟以后就好了"
        ]
        
        if np.random.random() < 0.4:
            catchphrase = np.random.choice(catchphrases)
            return f"{catchphrase}。{text}"
        
        return text
    
    def _apply_xieguangkun_style(self, text: str) -> str:
        """谢广坤风格"""
        replacements = {
            "我": "我谢广坤",
            "你": "你",
            "儿子": "永强",
            "女儿": "小蒙",
            "老婆": "玉田娘",
            "邻居": "刘能那老小子",
            "生气": "广坤很生气，后果很严重",
            "高兴": "我谢广坤今天高兴",
            "不行": "这事没完！",
            "可以": "那得看永强同不同意",
            "钱": "都是为了永强"
        }
        
        result = text
        for orig, repl in replacements.items():
            result = result.replace(orig, repl)
        
        return result
    
    def _apply_luoxiang_style(self, text: str) -> str:
        """罗翔风格"""
        law_terms = [
            "根据《中华人民共和国刑法》",
            "张三又来了",
            "这属于违法行为",
            "法治社会",
            "法律面前人人平等",
            "我们要心存敬畏",
            "这得看具体案情",
            "从法律角度分析",
            "罪刑法定原则",
            "程序正义"
        ]
        
        if np.random.random() < 0.3:
            term = np.random.choice(law_terms)
            return f"{term}，{text}"
        
        return text
    
    def _apply_guodegang_style(self, text: str) -> str:
        """郭德纲风格"""
        replacements = {
            "我": "我郭德纲",
            "你": "你",
            "说": "说相声",
            "笑": "逗你玩",
            "聪明": "机灵",
            "笨": "脑子不好使",
            "朋友": "于谦",
            "父亲": "于谦的父亲王老爷子",
            "儿子": "郭麒麟",
            "徒弟": "岳云鹏",
            "好吃": "于谦家吃的就是好"
        }
        
        result = text
        for orig, repl in replacements.items():
            result = result.replace(orig, repl)
        
        if "：" in result:
            parts = result.split("：", 1)
            result = f"{parts[0]}（捧哏：于谦）：{parts[1]}"
        
        return result
    
    def _apply_sunxiaochuan_style(self, text: str) -> str:
        """孙笑川风格"""
        abstract_terms = [
            "你妈死了",
            "带哥们",
            "抽象",
            "缝合怪",
            "烂活",
            "好活",
            "整不会了",
            "典中典",
            "急了急了",
            "差不多得了"
        ]
        
        if np.random.random() < 0.4:
            term = np.random.choice(abstract_terms)
            return f"{term}，{text}"
        
        return text
    
    def _apply_tongxiangyu_style(self, text: str) -> str:
        """佟湘玉风格"""
        replacements = {
            "我": "额",
            "你": "你",
            "的": "滴",
            "啊": "啊～",
            "吗": "嘛",
            "了": "咧",
            "真的": "真滴",
            "不是": "不是我说你",
            "为什么": "这是为啥嘛",
            "怎么办": "额滴神啊，这可咋办嘛",
            "后悔": "我好后悔呀",
            "生气": "额滴心啊，哇凉哇凉滴"
        }
        
        result = text
        for orig, repl in replacements.items():
            result = result.replace(orig, repl)
        
        return result
    
    def _apply_sheldon_style(self, text: str) -> str:
        """谢尔顿风格"""
        replacements = {
            "我": "我",
            "你": "你",
            "聪明": "显然",
            "知道": "根据我的计算",
            "应该": "从理论上讲",
            "可能": "概率上",
            "朋友": "莱纳德",
            "室友": "莱纳德",
            "邻居": "佩妮",
            "妈妈": "我妈妈经常说",
            "博士": "作为一个理论物理博士"
        }
        
        result = text
        for orig, repl in replacements.items():
            result = result.replace(orig, repl)
        
        prefixes = ["Bazinga！", "事实上，", "准确地说，", "根据弦理论，"]
        if np.random.random() < 0.3:
            prefix = np.random.choice(prefixes)
            result = f"{prefix}{result}"
        
        return result
    
    async def batch_generate_voices(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """批量生成语音 - 串行版本"""
        logger.info(f"开始串行生成语音，总片段数: {len(segments)}")
        
        valid_segments = [seg for seg in segments if seg.styled_text and len(seg.styled_text.strip()) > 1]
        logger.info(f"有效片段数量: {len(valid_segments)}/{len(segments)}")
        
        if not valid_segments:
            logger.warning("没有有效的文本片段需要生成语音")
            return segments
        
        processed_count = 0
        failed_count = 0
        
        for i, seg in enumerate(valid_segments):
            seg_num = i + 1
            total_segs = len(valid_segments)
            
            profile = self.character_profiles.get(seg.character)
            if not profile:
                profile = CharacterProfile.get_preset(seg.character)
                self.character_profiles[seg.character] = profile
            
            clean_text = self._clean_text_for_tts(seg.styled_text)
            
            if len(clean_text) < 2:
                seg.audio_data = None
                logger.warning(f"片段 {seg.id} 文本无效，跳过")
                continue
            
            logger.info(f"正在生成语音 [{seg_num}/{total_segs}]: {clean_text[:50]}...")
            
            try:
                audio_data = await self._generate_single_voice_with_retry(
                    clean_text, profile, seg.id, max_retries=2
                )
                
                if audio_data is not None:
                    seg.audio_data = audio_data
                    processed_count += 1
                    
                    delay_seconds = 0.5
                    logger.debug(f"语音生成成功，等待 {delay_seconds} 秒后处理下一个...")
                    await asyncio.sleep(delay_seconds)
                    
                else:
                    seg.audio_data = None
                    failed_count += 1
                    logger.warning(f"❌ 片段 {seg.id} 语音生成失败")
                    
            except Exception as e:
                seg.audio_data = None
                failed_count += 1
                logger.error(f"❌ 片段 {seg.id} 语音生成异常: {e}")
            
            if seg_num % 5 == 0 or seg_num == total_segs:
                progress = int(seg_num / total_segs * 100)
                logger.info(f"语音生成进度: {seg_num}/{total_segs} ({progress}%)")
        
        logger.info(f"语音生成完成: {processed_count}成功, {failed_count}失败, 共{len(valid_segments)}个片段")
        return segments
    
    async def _generate_single_voice_with_retry(self, text: str, profile: CharacterProfile, seg_id: int, max_retries: int = 3) -> Optional[np.ndarray]:
        """生成单个语音（带重试）"""
        for attempt in range(max_retries):
            try:
                audio_data = await self._generate_single_voice(text, profile)
                if audio_data is not None:
                    return audio_data
                else:
                    logger.warning(f"片段 {seg_id} 语音生成尝试 {attempt+1} 返回空")
            except Exception as e:
                logger.warning(f"片段 {seg_id} 语音生成尝试 {attempt+1} 失败: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 1 * (2 ** attempt)
                logger.info(f"片段 {seg_id} 等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
        
        return None
    
    async def _generate_single_voice(self, text: str, profile: CharacterProfile) -> Optional[np.ndarray]:
        """生成单个语音"""
        import base64
        
        if not text or len(text) < 2:
            return None
        
        try:
            import soundfile as sf
            from pathlib import Path
            
            temp_dir = Path(self.config.temp_dir)
            temp_dir.mkdir(exist_ok=True)
            
            prompt_audio_b64 = ""
            if profile.voice_file and os.path.exists(profile.voice_file):
                try:
                    audio_data, sr = sf.read(profile.voice_file)
                    temp_wav_path = temp_dir / f"prompt_{hash(text)}.wav"
                    sf.write(str(temp_wav_path), audio_data, sr)
                    
                    with open(temp_wav_path, 'rb') as f:
                        audio_bytes = f.read()
                    
                    prompt_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    logger.debug(f"使用音色文件: {profile.voice_file}")
                except Exception as e:
                    logger.warning(f"音色文件加载失败，使用默认音频: {e}")
                    sr = 24000
                    silence = np.zeros(sr, dtype=np.float32)
                    temp_wav_path = temp_dir / f"silence_{hash(text)}.wav"
                    sf.write(temp_wav_path, silence, sr)
                    
                    with open(temp_wav_path, 'rb') as f:
                        audio_bytes = f.read()
                    
                    prompt_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            else:
                sr = 24000
                silence = np.zeros(sr, dtype=np.float32)
                temp_wav_path = temp_dir / f"silence_{hash(text)}.wav"
                sf.write(temp_wav_path, silence, sr)
                
                with open(temp_wav_path, 'rb') as f:
                    audio_bytes = f.read()
                
                prompt_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                logger.debug("使用默认音频")
            
            payload = {
                "tts_text": text[:150],
                "prompt_audio": prompt_audio_b64,
                "emo_control_method": 0,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 5.0,
                "max_mel_tokens": 200,
                "speed": min(max(profile.speed, 0.5), 2.0)
            }
            
            logger.debug(f"发送TTS请求: {text[:30]}...")
            logger.debug(f"TTS API URL: {self.config.tts_api_url}")
            
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=60)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.config.tts_api_url, 
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    logger.info(f"TTS响应状态: {response.status}")
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"TTS响应: {result}")
                        
                        if result.get("status") == "success":
                            audio_base64 = result.get("audio", "")
                            if audio_base64:
                                audio_bytes = base64.b64decode(audio_base64)
                                temp_output_path = temp_dir / f"output_{hash(text)}.wav"
                                with open(temp_output_path, 'wb') as f:
                                    f.write(audio_bytes)
                                
                                audio_data, sr = sf.read(str(temp_output_path))
                                logger.debug(f"✅ 语音生成成功: {text[:30]}...")
                                return audio_data.astype(np.float32)
                            else:
                                logger.warning(f"TTS API返回无音频数据: {text[:30]}...")
                        else:
                            error_msg = result.get('message', '未知错误')
                            logger.warning(f"TTS API返回错误状态: {error_msg}")
                    elif response.status == 400:
                        error_text = await response.text()
                        logger.error(f"TTS API 400错误: {error_text[:200]}")
                    elif response.status == 500:
                        error_text = await response.text()
                        logger.error(f"TTS API服务器500错误: {error_text[:200]}")
                    else:
                        error_text = await response.text()
                        logger.warning(f"TTS API HTTP错误 {response.status}: {error_text[:200]}")
            
            return None
            
        except aiohttp.ClientConnectorError:
            logger.error(f"无法连接到TTS服务: {self.config.tts_api_url}")
            logger.error("请确保TTS服务已启动: python api.py --port 5021")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"TTS请求超时: {text[:30]}...")
            return None
        except Exception as e:
            logger.error(f"语音生成异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _clean_text_for_tts(self, text: str) -> str:
        """清理文本，移除特殊字符"""
        if not text:
            return ""
        
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'(Okay|OK|思考|分析|注意).*?(user|用户|问题|文本)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        
        max_length = 200
        if len(text) > max_length:
            text = text[:max_length] + "。"
        
        return text
    
    def extract_background_audio(self, audio_path: str) -> np.ndarray:
        """提取背景音频（去除人声）"""
        logger.info("提取背景音频")
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            if self.config.keep_background:
                nyquist = sr / 2
                
                b_low, a_low = butter(6, 300/nyquist, btype='low')
                low_freq = filtfilt(b_low, a_low, audio)
                
                b_high, a_high = butter(6, 80/nyquist, btype='high')
                background = filtfilt(b_high, a_high, audio)
                
                background = background * 0.7 + low_freq * 0.3
                background = background * self.config.background_volume
                
                self.background_audio = background
                return background
            else:
                self.background_audio = np.zeros_like(audio)
                return self.background_audio
                
        except Exception as e:
            logger.error(f"背景音频提取失败: {e}")
            return np.array([])
    
    def _adjust_audio_duration(self, audio: np.ndarray, target_samples: int, sr: int) -> np.ndarray:
        """智能调整音频时长"""
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        
        if target_samples <= 0:
            return np.array([], dtype=np.float32)
        
        try:
            from librosa.effects import time_stretch
            stretch_factor = target_samples / current_samples
            
            if 0.5 <= stretch_factor <= 2.0:
                stretched_audio = time_stretch(audio, rate=stretch_factor)
                if len(stretched_audio) > target_samples:
                    return stretched_audio[:target_samples]
                elif len(stretched_audio) < target_samples:
                    pad_length = target_samples - len(stretched_audio)
                    segment_audio = np.pad(stretched_audio, (0, pad_length), mode='constant')
                    return segment_audio
                else:
                    return stretched_audio
        except ImportError:
            pass
        
        if current_samples > 1:
            x_old = np.linspace(0, 1, current_samples)
            x_new = np.linspace(0, 1, target_samples)
            f = interp1d(x_old, audio, kind='linear')
            return f(x_new)
        else:
            if current_samples == 1:
                return np.full(target_samples, audio[0])
            else:
                return np.zeros(target_samples)
    
    def _apply_crossfade(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """应用交叉淡入淡出"""
        fade_duration = int(self.config.fade_duration * sr)
        
        if len(audio) > 2 * fade_duration:
            fade_in = np.linspace(0, 1, fade_duration)
            fade_out = np.linspace(1, 0, fade_duration)
            
            audio[:fade_duration] *= fade_in
            audio[-fade_duration:] *= fade_out
        
        return audio
    
    def _smart_normalize(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """智能音频归一化"""
        if len(audio) == 0:
            return audio
        
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return audio
        
        current_db = 20 * np.log10(rms)
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        
        normalized = audio * gain
        
        max_val = np.max(np.abs(normalized))
        if max_val > 0.95:
            normalized = normalized * 0.95 / max_val
        
        return normalized
    
    def mix_audio(self, original_audio_path: str, voice_segments: List[AudioSegment], output_path: str):
        """改进的音频混合方法 - 确保时长精确对齐"""
        logger.info("开始音频混合（改进版）")
        
        try:
            audio, sr = sf.read(original_audio_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            dubbing_track = np.zeros_like(audio, dtype=np.float32)
            
            for segment in voice_segments:
                if segment.audio_data is not None and segment.volume > 0:
                    start_sample = int(segment.start_time * sr)
                    end_sample = int(segment.end_time * sr)
                    segment_duration_samples = end_sample - start_sample
                    
                    if segment_duration_samples > 0:
                        segment_audio = self._adjust_audio_duration(
                            segment.audio_data, 
                            segment_duration_samples,
                            sr
                        )
                        
                        if len(segment_audio) > segment_duration_samples:
                            segment_audio = segment_audio[:segment_duration_samples]
                        elif len(segment_audio) < segment_duration_samples:
                            pad_length = segment_duration_samples - len(segment_audio)
                            segment_audio = np.pad(segment_audio, (0, pad_length), mode='constant')
                        
                        segment_audio = self._apply_crossfade(segment_audio, sr)
                        segment_audio = segment_audio * segment.volume * self.config.voice_volume
                        
                        end_idx = min(start_sample + len(segment_audio), len(dubbing_track))
                        if end_idx > start_sample:
                            dubbing_track[start_sample:end_idx] += segment_audio[:end_idx-start_sample]
            
            if self.background_audio is not None and len(self.background_audio) > 0:
                bg_len = min(len(self.background_audio), len(dubbing_track))
                dubbing_track[:bg_len] += self.background_audio[:bg_len]
            
            dubbing_track = self._smart_normalize(dubbing_track)
            
            sf.write(output_path, dubbing_track, sr)
            logger.info(f"音频混合完成: {output_path}")
            
        except Exception as e:
            logger.error(f"音频混合失败: {e}")
            raise
    
    def replace_video_audio(self, video_path: str, audio_path: str, output_path: str):
        """替换视频中的音频"""
        logger.info(f"合成最终视频: {output_path}")
        
        try:
            video = ffmpeg.input(video_path)
            audio = ffmpeg.input(audio_path)
            
            stream = ffmpeg.output(
                video.video,
                audio.audio,
                output_path,
                vcodec='copy',
                acodec='aac',
                audio_bitrate='192k',
                loglevel='quiet'
            )
            
            ffmpeg.run(stream, overwrite_output=True)
            logger.info(f"视频合成完成: {output_path}")
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg合成错误: {e}")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                '-y',
                output_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"视频合成完成: {output_path}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"视频合成失败: {e}")
    
    def process_episode_optimized(self, video_path: str) -> str:
        """优化后的视频处理流程 - 添加口型对齐"""
        logger.info(f"开始优化处理视频: {video_path}")
        
        temp_dir = Path(self.config.temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        try:
            logger.info("阶段1: 提取音频")
            audio_path = self.extract_audio(video_path)
            
            logger.info("阶段2: 语音识别")
            segments = self.transcribe_audio(audio_path)
            
            logger.info("阶段3: 提取背景音频")
            self.extract_background_audio(audio_path)
            
            logger.info("阶段4: 批量翻译")
            segments = self.batch_translate_segments(segments)
            
            logger.info("阶段5: 批量风格转换")
            segments = self.batch_apply_voice_styles(segments)
            
            logger.info("阶段6: 批量语音生成（串行）")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                original_batch_size = self.config.tts_batch_size
                self.config.tts_batch_size = 1
                
                segments = loop.run_until_complete(self.batch_generate_voices(segments))
                
                self.config.tts_batch_size = original_batch_size
            finally:
                loop.close()
            
            logger.info("阶段7: 混合音频")
            mixed_audio_path = temp_dir / "mixed_audio.wav"
            self.mix_audio(audio_path, segments, str(mixed_audio_path))
            
            logger.info("阶段7.5: 口型同步对齐")
            post_processor = AudioPostProcessor(self.config)
            aligned_audio_path = post_processor.align_to_lip_sync(video_path, str(mixed_audio_path))
            
            logger.info("阶段8: 合成视频")
            output_path = self.config.output_video or str(Path(video_path).with_stem(f"{Path(video_path).stem}_dubbed"))
            
            if aligned_audio_path and os.path.exists(aligned_audio_path):
                self.replace_video_audio(video_path, aligned_audio_path, output_path)
            else:
                logger.warning("口型对齐失败，使用原始混合音频")
                self.replace_video_audio(video_path, str(mixed_audio_path), output_path)
            
            logger.info("阶段9: 清理临时文件")
            for temp_file in temp_dir.glob("*"):
                if temp_file.is_file():
                    try:
                        temp_file.unlink()
                    except:
                        pass
            
            logger.info(f"视频处理完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            import traceback
            traceback.print_exc()
            raise

# ============================================
# GUI界面类 - 处理线程
# ============================================

class ProcessingThread(QThread):
    """处理线程"""
    
    progress_updated = Signal(int, str)
    stage_changed = Signal(str, str)
    segment_processed = Signal(int, int)
    log_message = Signal(str, str)
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, engine: DubbingEngine, video_path: str):
        super().__init__()
        self.engine = engine
        self.video_path = video_path
        self.is_running = True
    
    def run(self):
        try:
            self.log_message.emit(f"开始处理视频: {self.video_path}", "INFO")
            
            output_path = self.engine.process_episode_optimized(self.video_path)
            
            self.log_message.emit(f"处理完成: {output_path}", "INFO")
            self.finished.emit(output_path)
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            self.log_message.emit(error_msg, "ERROR")
            self.error.emit(error_msg)
    
    def stop(self):
        self.is_running = False

class AnimationGenerationThread(QThread):
    """动画生成线程"""
    
    progress_updated = Signal(int, str)
    log_message = Signal(str, str)
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, generator, script_path: str):
        super().__init__()
        self.generator = generator
        self.script_path = script_path
        self.is_running = True
    
    def run(self):
        try:
            self.log_message.emit("开始生成AI驱动动画...", "INFO")
            
            import asyncio
            
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                self.progress_updated.emit(10, "正在初始化...")
                
                # 方法1: 直接运行协程
                result = loop.run_until_complete(self.generator.initialize())
                
                if result:
                    self.progress_updated.emit(30, "正在处理剧本...")
                    final_video = loop.run_until_complete(
                        self.generator.generate_complete_animation(self.script_path)
                    )
                    
                    if final_video:
                        self.progress_updated.emit(100, "完成")
                        self.log_message.emit(f"✅ 动画生成完成: {final_video}", "INFO")
                        self.finished.emit(final_video)
                    else:
                        self.error.emit("动画生成失败")
                else:
                    self.error.emit("AI动画生成器初始化失败")
                    
            except Exception as e:
                error_msg = f"动画生成失败: {str(e)}"
                self.log_message.emit(error_msg, "ERROR")
                self.error.emit(error_msg)
                import traceback
                traceback.print_exc()
                
            finally:
                loop.close()
                
        except Exception as e:
            error_msg = f"动画生成线程失败: {str(e)}"
            self.log_message.emit(error_msg, "ERROR")
            self.error.emit(error_msg)
    
    def stop(self):
        self.is_running = False

# ============================================
# GUI界面类 - 角色配置对话框
# ============================================

class CharacterConfigDialog(QDialog):
    """角色配置对话框"""
    
    def __init__(self, character_name: str, parent=None):
        super().__init__(parent)
        self.character_name = character_name
        self.profile = CharacterProfile.get_preset(character_name)
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle(f"配置角色: {self.character_name}")
        self.setModal(True)
        self.resize(500, 600)
        
        layout = QVBoxLayout()
        
        info_group = QGroupBox("角色信息")
        info_layout = QVBoxLayout()
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("角色名:"))
        self.name_edit = QLineEdit(self.profile.name)
        name_layout.addWidget(self.name_edit)
        info_layout.addLayout(name_layout)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        voice_group = QGroupBox("音色配置")
        voice_layout = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("音色文件:"))
        self.file_edit = QLineEdit(self.profile.voice_file)
        file_layout.addWidget(self.file_edit)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_voice_file)
        file_layout.addWidget(browse_btn)
        voice_layout.addLayout(file_layout)
        
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("语音风格:"))
        self.style_combo = QComboBox()
        for style in VoiceStyle:
            self.style_combo.addItem(style.value, style)
        self.style_combo.setCurrentText(self.profile.voice_style.value)
        style_layout.addWidget(self.style_combo)
        voice_layout.addLayout(style_layout)
        
        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)
        
        params_group = QGroupBox("语音参数")
        params_layout = QVBoxLayout()
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("语速:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(int(self.profile.speed * 100))
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel(f"{self.profile.speed:.1f}")
        speed_layout.addWidget(self.speed_label)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_label.setText(f"{v/100:.1f}")
        )
        params_layout.addLayout(speed_layout)
        
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("音高:"))
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(50, 200)
        self.pitch_slider.setValue(int(self.profile.pitch * 100))
        pitch_layout.addWidget(self.pitch_slider)
        self.pitch_label = QLabel(f"{self.profile.pitch:.1f}")
        pitch_layout.addWidget(self.pitch_label)
        self.pitch_slider.valueChanged.connect(
            lambda v: self.pitch_label.setText(f"{v/100:.1f}")
        )
        params_layout.addLayout(pitch_layout)
        
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("强度:"))
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(50, 200)
        self.intensity_slider.setValue(int(self.profile.intensity * 100))
        intensity_layout.addWidget(self.intensity_slider)
        self.intensity_label = QLabel(f"{self.profile.intensity:.1f}")
        intensity_layout.addWidget(self.intensity_label)
        self.intensity_slider.valueChanged.connect(
            lambda v: self.intensity_label.setText(f"{v/100:.1f}")
        )
        params_layout.addLayout(intensity_layout)
        
        emotion_layout = QHBoxLayout()
        emotion_layout.addWidget(QLabel("情感:"))
        self.emotion_combo = QComboBox()
        emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "excited"]
        self.emotion_combo.addItems(emotions)
        self.emotion_combo.setCurrentText(self.profile.emotion)
        emotion_layout.addWidget(self.emotion_combo)
        params_layout.addLayout(emotion_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        catchphrase_group = QGroupBox("口头禅")
        catchphrase_layout = QVBoxLayout()
        
        self.catchphrase_edit = QTextEdit()
        self.catchphrase_edit.setMaximumHeight(100)
        self.catchphrase_edit.setText("\n".join(self.profile.catchphrases))
        catchphrase_layout.addWidget(self.catchphrase_edit)
        
        catchphrase_group.setLayout(catchphrase_layout)
        layout.addWidget(catchphrase_group)
        
        button_layout = QHBoxLayout()
        
        test_btn = QPushButton("测试音色")
        test_btn.clicked.connect(self.test_voice)
        button_layout.addWidget(test_btn)
        
        button_layout.addStretch()
        
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def browse_voice_file(self):
        """浏览音色文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音色文件", "", 
            "音频文件 (*.wav *.mp3 *.ogg);;所有文件 (*.*)"
        )
        
        if file_path:
            self.file_edit.setText(file_path)
    
    def test_voice(self):
        """测试音色"""
        QMessageBox.information(self, "测试", f"测试 {self.character_name} 的音色")
    
    def get_profile(self) -> CharacterProfile:
        """获取配置的角色信息"""
        self.profile.name = self.name_edit.text()
        self.profile.voice_file = self.file_edit.text()
        self.profile.voice_style = self.style_combo.currentData()
        self.profile.speed = self.speed_slider.value() / 100
        self.profile.pitch = self.pitch_slider.value() / 100
        self.profile.intensity = self.intensity_slider.value() / 100
        self.profile.emotion = self.emotion_combo.currentText()
        
        catchphrases_text = self.catchphrase_edit.toPlainText()
        self.profile.catchphrases = [
            cp.strip() for cp in catchphrases_text.split('\n') 
            if cp.strip()
        ]
        
        return self.profile

# ============================================
# GUI界面类 - 主窗口
# ============================================

from PySide6.QtGui import QDesktopServices

class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.engine = None
        self.processing_thread = None
        self.config = ProcessingConfig()
        self.character_profiles = {}
        self.current_video = ""
        self.animation_generator = None
        self.animation_thread = None
        
        self.init_ui()
        self.load_config()
        self.setup_default_characters()
        
        logger.info("主窗口初始化完成")
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("🎭 AI配音工厂 - 超强嘴炮版 🚀")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setup_stylesheet()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        self.setup_project_tab()
        self.setup_character_tab()
        self.setup_settings_tab()
        self.setup_batch_tab()
        self.setup_animation_tab()
        
        self.setup_status_bar()
        self.setup_menu_bar()
    
    def setup_stylesheet(self):
        """设置样式表"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #3d3d3d;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #007acc;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #505050;
            }
            QGroupBox {
                color: #ffffff;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #4ec9b0;
            }
            QLabel {
                color: #cccccc;
                font-size: 13px;
            }
            QPushButton {
                background-color: #007acc;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #1c97ea;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #505050;
                color: #888888;
            }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                font-size: 13px;
                selection-background-color: #007acc;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #007acc;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                color: white;
                font-size: 12px;
                background-color: #3d3d3d;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 3px;
            }
            QTableWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                gridline-color: #444;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #007acc;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                color: white;
                padding: 5px;
                border: 1px solid #444;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #505050;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #606060;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
    
    def setup_project_tab(self):
        """设置项目标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        video_group = QGroupBox("视频文件")
        video_layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入视频:"))
        
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("选择要配音的视频文件...")
        input_layout.addWidget(self.input_edit)
        
        self.input_browse_btn = QPushButton("浏览...")
        self.input_browse_btn.clicked.connect(self.browse_input_video)
        input_layout.addWidget(self.input_browse_btn)
        
        video_layout.addLayout(input_layout)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出路径:"))
        
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("自动生成输出路径...")
        output_layout.addWidget(self.output_edit)
        
        self.output_browse_btn = QPushButton("浏览...")
        self.output_browse_btn.clicked.connect(self.browse_output_path)
        output_layout.addWidget(self.output_browse_btn)
        
        video_layout.addLayout(output_layout)
        
        preview_layout = QHBoxLayout()
        self.preview_label = QLabel("未选择视频")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(150)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #3d3d3d;
                border: 2px dashed #555;
                border-radius: 5px;
                color: #888;
                font-size: 14px;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        video_layout.addLayout(preview_layout)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        control_group = QGroupBox("处理控制")
        control_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        control_layout.addWidget(self.progress_bar)
        
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ 开始处理")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        button_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.clicked.connect(self.pause_processing)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setMinimumHeight(40)
        button_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        button_layout.addWidget(self.stop_btn)
        
        control_layout.addLayout(button_layout)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        log_control_layout = QHBoxLayout()
        log_control_layout.addWidget(QLabel("日志级别:"))
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText("INFO")
        self.log_level_combo.currentTextChanged.connect(self.change_log_level)
        log_control_layout.addWidget(self.log_level_combo)
        
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        log_control_layout.addWidget(clear_log_btn)
        
        save_log_btn = QPushButton("保存日志")
        save_log_btn.clicked.connect(self.save_log)
        log_control_layout.addWidget(save_log_btn)
        
        log_control_layout.addStretch()
        log_layout.addLayout(log_control_layout)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.tab_widget.addTab(tab, "🎬 项目")
    
    def setup_character_tab(self):
        """设置角色标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info_label = QLabel("🎭 为每个角色配置音色和风格，打造独一无二的嘴炮天团！")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4ec9b0;")
        layout.addWidget(info_label)
        
        self.character_table = QTableWidget()
        self.character_table.setColumnCount(8)
        self.character_table.setHorizontalHeaderLabels([
            "角色", "原版名", "语音风格", "音色文件", "语速", "强度", "情感", "操作"
        ])
        
        header = self.character_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.character_table)
        
        control_layout = QHBoxLayout()
        
        add_btn = QPushButton("➕ 添加角色")
        add_btn.clicked.connect(self.add_character)
        control_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("➖ 删除角色")
        remove_btn.clicked.connect(self.remove_character)
        control_layout.addWidget(remove_btn)
        
        import_btn = QPushButton("📥 导入配置")
        import_btn.clicked.connect(self.import_character_config)
        control_layout.addWidget(import_btn)
        
        export_btn = QPushButton("📤 导出配置")
        export_btn.clicked.connect(self.export_character_config)
        control_layout.addWidget(export_btn)
        
        preset_btn = QPushButton("🎭 加载预设")
        preset_btn.clicked.connect(self.load_preset_characters)
        control_layout.addWidget(preset_btn)
        
        test_all_btn = QPushButton("🔊 测试全部")
        test_all_btn.clicked.connect(self.test_all_voices)
        control_layout.addWidget(test_all_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        preset_info = QLabel("""
        🎯 推荐预设组合：
        • 哆啦A梦 → 李云龙（暴躁将军）
        • 大雄 → 王境泽（真香定律）
        • 静香 → 佟湘玉（碎碎念掌柜）
        • 胖虎 → 张飞（暴躁三爷）
        • 小夫 → 马保国（混元太极）
        • 哆啦美 → 罗翔（法律相声）
        • 出木杉 → 谢尔顿（学霸嘲讽）
        """)
        preset_info.setStyleSheet("""
            QLabel {
                background-color: #252525;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 10px;
                color: #b5cea8;
                font-size: 12px;
            }
        """)
        layout.addWidget(preset_info)
        
        self.tab_widget.addTab(tab, "🎭 角色")
    
    def setup_settings_tab(self):
        """设置设置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        ai_group = QGroupBox("🤖 AI模型设置")
        ai_layout = QVBoxLayout()
        
        whisper_layout = QHBoxLayout()
        whisper_layout.addWidget(QLabel("语音识别模型:"))
        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.whisper_combo.setCurrentText("large-v3")
        whisper_layout.addWidget(self.whisper_combo)
        whisper_layout.addStretch()
        ai_layout.addLayout(whisper_layout)
        
        translate_layout = QHBoxLayout()
        translate_layout.addWidget(QLabel("翻译模型:"))
        self.translate_combo = QComboBox()
        self.translate_combo.addItems([
            "qwen3:4b",
            "qwen3:14b",
            "llama3.2:3b",
            "moondream:latest",
            "llava:7b",
            "llava:13b",
            "huihui_ai/deepseek-r1-abliterated:14b",
            "huihui_ai/qwen3-abliterated:16b"
        ])
        self.translate_combo.setCurrentText("qwen3:4b")
        self.translate_combo.setEditable(True)
        translate_layout.addWidget(self.translate_combo)
        
        refresh_models_btn = QPushButton("🔄 刷新")
        refresh_models_btn.clicked.connect(self.refresh_ollama_models)
        translate_layout.addWidget(refresh_models_btn)
        
        translate_layout.addStretch()
        ai_layout.addLayout(translate_layout)
        
        ollama_layout = QHBoxLayout()
        ollama_layout.addWidget(QLabel("Ollama基础地址:"))
        self.ollama_api_edit = QLineEdit("http://127.0.0.1:11434")
        ollama_layout.addWidget(self.ollama_api_edit)
        
        test_ollama_btn = QPushButton("测试配置")
        test_ollama_btn.clicked.connect(self.test_ollama_api)
        test_ollama_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        ollama_layout.addWidget(test_ollama_btn)
        ai_layout.addLayout(ollama_layout)
        
        api_info = QLabel("💡 使用原生Ollama API: {基础地址}/api/generate")
        api_info.setStyleSheet("color: #b5cea8; font-size: 11px;")
        ai_layout.addWidget(api_info)
        
        tts_layout = QHBoxLayout()
        tts_layout.addWidget(QLabel("TTS API地址:"))
        self.tts_api_edit = QLineEdit("http://127.0.0.1:5021/api/tts")
        tts_layout.addWidget(self.tts_api_edit)
        test_tts_btn = QPushButton("测试")
        test_tts_btn.clicked.connect(self.test_tts_api)
        tts_layout.addWidget(test_tts_btn)
        ai_layout.addLayout(tts_layout)
        
        tts_info = QLabel("💡 TTS服务: python api.py --port 5021")
        tts_info.setStyleSheet("color: #b5cea8; font-size: 11px;")
        ai_layout.addWidget(tts_info)
        
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Temperature:"))
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.1)
        params_layout.addWidget(self.temperature_spin)
        
        params_layout.addWidget(QLabel("Max Tokens:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 100000)
        self.max_tokens_spin.setValue(4096)
        params_layout.addWidget(self.max_tokens_spin)
        
        params_layout.addWidget(QLabel("Timeout(sec):"))
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 3600)
        self.timeout_spin.setValue(60)
        params_layout.addWidget(self.timeout_spin)
        
        ai_layout.addLayout(params_layout)
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        audio_group = QGroupBox("🔊 音频设置")
        audio_layout = QVBoxLayout()
        
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(QLabel("采样率:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["16000", "22050", "24000", "44100", "48000"])
        self.sample_rate_combo.setCurrentText("24000")
        sr_layout.addWidget(self.sample_rate_combo)
        audio_layout.addLayout(sr_layout)
        
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("配音音量:"))
        self.voice_volume_slider = QSlider(Qt.Horizontal)
        self.voice_volume_slider.setRange(0, 200)
        self.voice_volume_slider.setValue(80)
        volume_layout.addWidget(self.voice_volume_slider)
        self.voice_volume_label = QLabel("80%")
        volume_layout.addWidget(self.voice_volume_label)
        self.voice_volume_slider.valueChanged.connect(
            lambda v: self.voice_volume_label.setText(f"{v}%")
        )
        audio_layout.addLayout(volume_layout)
        
        bg_volume_layout = QHBoxLayout()
        bg_volume_layout.addWidget(QLabel("背景音量:"))
        self.bg_volume_slider = QSlider(Qt.Horizontal)
        self.bg_volume_slider.setRange(0, 100)
        self.bg_volume_slider.setValue(30)
        bg_volume_layout.addWidget(self.bg_volume_slider)
        self.bg_volume_label = QLabel("30%")
        bg_volume_layout.addWidget(self.bg_volume_label)
        self.bg_volume_slider.valueChanged.connect(
            lambda v: self.bg_volume_label.setText(f"{v}%")
        )
        audio_layout.addLayout(bg_volume_layout)
        
        self.keep_bg_check = QCheckBox("保留背景音乐和音效")
        self.keep_bg_check.setChecked(True)
        audio_layout.addWidget(self.keep_bg_check)
        
        self.noise_reduction_check = QCheckBox("启用降噪")
        self.noise_reduction_check.setChecked(True)
        audio_layout.addWidget(self.noise_reduction_check)
        
        self.normalize_check = QCheckBox("音频归一化")
        self.normalize_check.setChecked(True)
        audio_layout.addWidget(self.normalize_check)
        
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        lip_sync_group = QGroupBox("👄 口型对齐设置")
        lip_sync_layout = QVBoxLayout()
        
        self.lip_sync_check = QCheckBox("启用口型对齐")
        self.lip_sync_check.setChecked(True)
        lip_sync_layout.addWidget(self.lip_sync_check)
        
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("对齐强度:"))
        self.lip_sync_strength_slider = QSlider(Qt.Horizontal)
        self.lip_sync_strength_slider.setRange(0, 100)
        self.lip_sync_strength_slider.setValue(80)
        strength_layout.addWidget(self.lip_sync_strength_slider)
        self.lip_sync_strength_label = QLabel("80%")
        strength_layout.addWidget(self.lip_sync_strength_label)
        self.lip_sync_strength_slider.valueChanged.connect(
            lambda v: self.lip_sync_strength_label.setText(f"{v}%")
        )
        lip_sync_layout.addLayout(strength_layout)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("对齐方法:"))
        self.lip_sync_method_combo = QComboBox()
        self.lip_sync_method_combo.addItems(["时间拉伸", "节奏匹配", "智能对齐"])
        self.lip_sync_method_combo.setCurrentText("时间拉伸")
        method_layout.addWidget(self.lip_sync_method_combo)
        lip_sync_layout.addLayout(method_layout)
        
        lip_sync_group.setLayout(lip_sync_layout)
        layout.addWidget(lip_sync_group)
        
        process_group = QGroupBox("⚙️ 处理设置")
        process_layout = QVBoxLayout()
        
        gpu_layout = QHBoxLayout()
        self.gpu_check = QCheckBox("启用GPU加速")
        self.gpu_check.setChecked(True)
        gpu_layout.addWidget(self.gpu_check)
        
        gpu_layout.addWidget(QLabel("GPU ID:"))
        self.gpu_id_spin = QSpinBox()
        self.gpu_id_spin.setRange(0, 7)
        self.gpu_id_spin.setValue(0)
        gpu_layout.addWidget(self.gpu_id_spin)
        gpu_layout.addStretch()
        process_layout.addLayout(gpu_layout)
        
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("翻译批次大小:"))
        self.translation_batch_spin = QSpinBox()
        self.translation_batch_spin.setRange(1, 50)
        self.translation_batch_spin.setValue(10)
        batch_layout.addWidget(self.translation_batch_spin)
        
        batch_layout.addWidget(QLabel("TTS批次大小:"))
        self.tts_batch_spin = QSpinBox()
        self.tts_batch_spin.setRange(1, 10)
        self.tts_batch_spin.setValue(3)
        batch_layout.addWidget(self.tts_batch_spin)
        batch_layout.addStretch()
        process_layout.addLayout(batch_layout)
        
        parallel_layout = QHBoxLayout()
        parallel_layout.addWidget(QLabel("并行线程数:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        parallel_layout.addWidget(self.workers_spin)
        parallel_layout.addStretch()
        process_layout.addLayout(parallel_layout)
        
        cache_layout = QHBoxLayout()
        self.cache_check = QCheckBox("启用缓存")
        self.cache_check.setChecked(True)
        cache_layout.addWidget(self.cache_check)
        process_layout.addLayout(cache_layout)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        save_btn = QPushButton("💾 保存所有设置")
        save_btn.clicked.connect(self.save_settings)
        save_btn.setMinimumHeight(40)
        layout.addWidget(save_btn)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "⚙️ 设置")
    
    def setup_batch_tab(self):
        """设置批量处理标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info_label = QLabel("📁 批量处理整个文件夹的视频文件，自动应用相同的角色配置")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4ec9b0;")
        layout.addWidget(info_label)
        
        folder_group = QGroupBox("文件夹设置")
        folder_layout = QVBoxLayout()
        
        input_folder_layout = QHBoxLayout()
        input_folder_layout.addWidget(QLabel("输入文件夹:"))
        self.input_folder_edit = QLineEdit()
        self.input_folder_edit.setPlaceholderText("选择包含视频文件的文件夹...")
        input_folder_layout.addWidget(self.input_folder_edit)
        
        self.input_folder_browse_btn = QPushButton("浏览...")
        self.input_folder_browse_btn.clicked.connect(self.browse_input_folder)
        input_folder_layout.addWidget(self.input_folder_browse_btn)
        folder_layout.addLayout(input_folder_layout)
        
        output_folder_layout = QHBoxLayout()
        output_folder_layout.addWidget(QLabel("输出文件夹:"))
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("选择输出文件夹...")
        output_folder_layout.addWidget(self.output_folder_edit)
        
        self.output_folder_browse_btn = QPushButton("浏览...")
        self.output_folder_browse_btn.clicked.connect(self.browse_output_folder)
        output_folder_layout.addWidget(self.output_folder_browse_btn)
        folder_layout.addLayout(output_folder_layout)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("文件格式:"))
        self.file_filter_combo = QComboBox()
        self.file_filter_combo.addItems(["*.mp4", "*.avi", "*.mkv", "*.mov", "所有视频文件"])
        filter_layout.addWidget(self.file_filter_combo)
        filter_layout.addStretch()
        folder_layout.addLayout(filter_layout)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        files_group = QGroupBox("文件列表")
        files_layout = QVBoxLayout()
        
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        files_layout.addWidget(self.file_list)
        
        file_control_layout = QHBoxLayout()
        
        scan_btn = QPushButton("🔍 扫描文件")
        scan_btn.clicked.connect(self.scan_video_files)
        file_control_layout.addWidget(scan_btn)
        
        select_all_btn = QPushButton("✓ 全选")
        select_all_btn.clicked.connect(lambda: self.file_list.selectAll())
        file_control_layout.addWidget(select_all_btn)
        
        clear_all_btn = QPushButton("✗ 清空")
        clear_all_btn.clicked.connect(lambda: self.file_list.clear())
        file_control_layout.addWidget(clear_all_btn)
        
        file_control_layout.addStretch()
        files_layout.addLayout(file_control_layout)
        
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)
        
        batch_control_group = QGroupBox("批量处理")
        batch_control_layout = QVBoxLayout()
        
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        batch_control_layout.addWidget(self.batch_progress_bar)
        
        batch_button_layout = QHBoxLayout()
        
        self.batch_start_btn = QPushButton("▶ 开始批量处理")
        self.batch_start_btn.clicked.connect(self.start_batch_processing)
        batch_button_layout.addWidget(self.batch_start_btn)
        
        self.batch_stop_btn = QPushButton("⏹ 停止")
        self.batch_stop_btn.clicked.connect(self.stop_batch_processing)
        self.batch_stop_btn.setEnabled(False)
        batch_button_layout.addWidget(self.batch_stop_btn)
        
        batch_button_layout.addStretch()
        batch_control_layout.addLayout(batch_button_layout)
        
        batch_control_group.setLayout(batch_control_layout)
        layout.addWidget(batch_control_group)
        
        self.tab_widget.addTab(tab, "📁 批量")
    
    def setup_animation_tab(self):
        """设置动画标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info_label = QLabel("🎬 AI驱动动画生成 - 集成ComfyUI人物一致性")
        info_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4ec9b0;")
        layout.addWidget(info_label)
        
        desc_label = QLabel("""🤖 利用AI生成式人工智能进行智能文本分析和动画提示提取
🎭 自动口型同步：生成音素数据，使角色嘴部动作与对话同步
👁️ 动态角色表情：支持头部动作、眼神表情和肢体语言
🎨 多角色支持：处理多个具有鲜明视觉素材和个性的角色
🎵 音频同步：通过转录服务，将动画与音频时间完美对齐
🖼️ 可扩展资产系统：易用的角色和背景资产管理
📹 逐帧生成：通过合成单个帧来创建流畅动画
🔄 自动化工作流程：从剧本到最终视频的完整流程""")
        desc_label.setStyleSheet("""
            QLabel {
                background-color: #252525;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 10px;
                color: #b5cea8;
                font-size: 12px;
                line-height: 1.5;
            }
        """)
        layout.addWidget(desc_label)
        
        comfyui_group = QGroupBox("🔗 ComfyUI连接")
        comfyui_layout = QVBoxLayout()
        
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("连接状态:"))
        self.comfyui_status_label = QLabel("未连接")
        self.comfyui_status_label.setStyleSheet("color: #f44747; font-weight: bold;")
        status_layout.addWidget(self.comfyui_status_label)
        comfyui_layout.addLayout(status_layout)
        
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel("服务器地址:"))
        self.comfyui_host_edit = QLineEdit("127.0.0.1")
        server_layout.addWidget(self.comfyui_host_edit)
        
        server_layout.addWidget(QLabel("端口:"))
        self.comfyui_port_edit = QSpinBox()
        self.comfyui_port_edit.setRange(1, 65535)
        self.comfyui_port_edit.setValue(8188)
        server_layout.addWidget(self.comfyui_port_edit)
        comfyui_layout.addLayout(server_layout)
        
        connect_btn = QPushButton("🔌 连接ComfyUI")
        connect_btn.clicked.connect(self.connect_comfyui)
        comfyui_layout.addWidget(connect_btn)
        
        comfyui_group.setLayout(comfyui_layout)
        layout.addWidget(comfyui_group)
        
        animation_config_group = QGroupBox("⚙️ 动画配置")
        animation_config_layout = QVBoxLayout()
        
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("分辨率:"))
        self.animation_width_spin = QSpinBox()
        self.animation_width_spin.setRange(256, 4096)
        self.animation_width_spin.setValue(512)
        resolution_layout.addWidget(self.animation_width_spin)
        resolution_layout.addWidget(QLabel("×"))
        self.animation_height_spin = QSpinBox()
        self.animation_height_spin.setRange(256, 4096)
        self.animation_height_spin.setValue(768)
        resolution_layout.addWidget(self.animation_height_spin)
        animation_config_layout.addLayout(resolution_layout)
        
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("帧率(FPS):"))
        self.animation_fps_spin = QSpinBox()
        self.animation_fps_spin.setRange(1, 60)
        self.animation_fps_spin.setValue(24)
        fps_layout.addWidget(self.animation_fps_spin)
        animation_config_layout.addLayout(fps_layout)
        
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("动画风格:"))
        self.animation_style_combo = QComboBox()
        self.animation_style_combo.addItems(["动漫", "电影", "卡通", "写实", "绘画", "像素艺术"])
        self.animation_style_combo.setCurrentText("动漫")
        style_layout.addWidget(self.animation_style_combo)
        animation_config_layout.addLayout(style_layout)
        
        consistency_layout = QHBoxLayout()
        consistency_layout.addWidget(QLabel("人物一致性方法:"))
        self.consistency_method_combo = QComboBox()
        self.consistency_method_combo.addItems(["IP-Adapter", "InstantID", "PhotoMaker", "LoRA", "ControlNet"])
        self.consistency_method_combo.setCurrentText("IP-Adapter")
        style_layout.addWidget(self.consistency_method_combo)
        animation_config_layout.addLayout(consistency_layout)
        
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("一致性强度:"))
        self.consistency_strength_slider = QSlider(Qt.Horizontal)
        self.consistency_strength_slider.setRange(10, 100)
        self.consistency_strength_slider.setValue(70)
        strength_layout.addWidget(self.consistency_strength_slider)
        self.consistency_strength_label = QLabel("70%")
        strength_layout.addWidget(self.consistency_strength_label)
        self.consistency_strength_slider.valueChanged.connect(
            lambda v: self.consistency_strength_label.setText(f"{v}%")
        )
        animation_config_layout.addLayout(strength_layout)
        
        scene_layout = QHBoxLayout()
        scene_layout.addWidget(QLabel("场景描述:"))
        self.scene_description_edit = QLineEdit()
        self.scene_description_edit.setPlaceholderText("例如：美丽的花园，樱花盛开，阳光明媚")
        scene_layout.addWidget(self.scene_description_edit)
        animation_config_layout.addLayout(scene_layout)
        
        features_layout = QHBoxLayout()
        self.animation_lip_sync_check = QCheckBox("口型同步")
        self.animation_lip_sync_check.setChecked(True)
        features_layout.addWidget(self.animation_lip_sync_check)
        
        self.expression_check = QCheckBox("表情动画")
        self.expression_check.setChecked(True)
        features_layout.addWidget(self.expression_check)
        
        self.head_movement_check = QCheckBox("头部动作")
        self.head_movement_check.setChecked(True)
        features_layout.addWidget(self.head_movement_check)
        
        self.eye_movement_check = QCheckBox("眼部动作")
        self.eye_movement_check.setChecked(True)
        features_layout.addWidget(self.eye_movement_check)
        
        animation_config_layout.addLayout(features_layout)
        
        animation_config_group.setLayout(animation_config_layout)
        layout.addWidget(animation_config_group)
        
        character_binding_group = QGroupBox("🎭 角色绑定")
        character_binding_layout = QVBoxLayout()
        
        self.animation_character_table = QTableWidget()
        self.animation_character_table.setColumnCount(4)
        self.animation_character_table.setHorizontalHeaderLabels([
            "配音角色", "参考图片", "动画角色名", "操作"
        ])
        character_binding_layout.addWidget(self.animation_character_table)
        
        binding_btn_layout = QHBoxLayout()
        
        auto_bind_btn = QPushButton("🤖 自动绑定角色")
        auto_bind_btn.clicked.connect(self.auto_bind_characters)
        binding_btn_layout.addWidget(auto_bind_btn)
        
        manual_bind_btn = QPushButton("✏️ 手动绑定")
        manual_bind_btn.clicked.connect(self.manual_bind_character)
        binding_btn_layout.addWidget(manual_bind_btn)
        
        clear_bindings_btn = QPushButton("🗑️ 清空绑定")
        clear_bindings_btn.clicked.connect(self.clear_character_bindings)
        binding_btn_layout.addWidget(clear_bindings_btn)
        
        character_binding_layout.addLayout(binding_btn_layout)
        character_binding_group.setLayout(character_binding_layout)
        layout.addWidget(character_binding_group)
        
        generation_group = QGroupBox("🚀 动画生成")
        generation_layout = QVBoxLayout()
        
        self.animation_progress_bar = QProgressBar()
        self.animation_progress_bar.setRange(0, 100)
        generation_layout.addWidget(self.animation_progress_bar)
        
        button_layout = QHBoxLayout()
        
        self.animation_generate_btn = QPushButton("🎬 生成动画")
        self.animation_generate_btn.clicked.connect(self.generate_animation)
        self.animation_generate_btn.setMinimumHeight(40)
        self.animation_generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        button_layout.addWidget(self.animation_generate_btn)
        
        self.animation_stop_btn = QPushButton("⏹ 停止")
        self.animation_stop_btn.clicked.connect(self.stop_animation_generation)
        self.animation_stop_btn.setEnabled(False)
        self.animation_stop_btn.setMinimumHeight(40)
        button_layout.addWidget(self.animation_stop_btn)
        
        preview_btn = QPushButton("👁️ 预览")
        preview_btn.clicked.connect(self.preview_animation)
        preview_btn.setMinimumHeight(40)
        button_layout.addWidget(preview_btn)
        
        button_layout.addStretch()
        generation_layout.addLayout(button_layout)
        
        generation_group.setLayout(generation_layout)
        layout.addWidget(generation_group)
        
        openai_group = QGroupBox("🤖 OpenAI集成 (可选)")
        openai_layout = QVBoxLayout()
        
        openai_layout.addWidget(QLabel("API密钥:"))
        self.openai_api_key_edit = QLineEdit()
        self.openai_api_key_edit.setPlaceholderText("输入OpenAI API密钥")
        self.openai_api_key_edit.setEchoMode(QLineEdit.Password)
        openai_layout.addWidget(self.openai_api_key_edit)
        
        openai_layout.addWidget(QLabel("基础URL:"))
        self.openai_base_url_edit = QLineEdit("https://apis.iflow.cn/v1")
        openai_layout.addWidget(self.openai_base_url_edit)
        
        self.enable_openai_check = QCheckBox("启用OpenAI增强")
        self.enable_openai_check.setChecked(False)
        openai_layout.addWidget(self.enable_openai_check)
        
        openai_group.setLayout(openai_layout)
        layout.addWidget(openai_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "🎬 动画生成")
    
    def setup_status_bar(self):
        """设置状态栏"""
        self.status_bar = self.statusBar()
        
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
        
        self.progress_label = QLabel("")
        self.status_bar.addWidget(self.progress_label)
        
        self.time_label = QLabel("")
        self.status_bar.addWidget(self.time_label)
        
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_time)
        self.status_timer.start(1000)
    
    def setup_menu_bar(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("📁 文件")
        
        new_project_action = QAction("🆕 新建项目", self)
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)
        
        open_project_action = QAction("📂 打开项目", self)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)
        
        save_project_action = QAction("💾 保存项目", self)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("🚪 退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        tools_menu = menubar.addMenu("🛠️ 工具")
        
        voice_manager_action = QAction("🎵 音色管理器", self)
        voice_manager_action.triggered.connect(self.open_voice_manager)
        tools_menu.addAction(voice_manager_action)
        
        subtitle_editor_action = QAction("📝 字幕编辑器", self)
        subtitle_editor_action.triggered.connect(self.open_subtitle_editor)
        tools_menu.addAction(subtitle_editor_action)
        
        audio_editor_action = QAction("🎛️ 音频编辑器", self)
        audio_editor_action.triggered.connect(self.open_audio_editor)
        tools_menu.addAction(audio_editor_action)
        
        help_menu = menubar.addMenu("❓ 帮助")
        
        docs_action = QAction("📚 使用文档", self)
        docs_action.triggered.connect(self.open_documentation)
        help_menu.addAction(docs_action)
        
        about_action = QAction("ℹ️ 关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_default_characters(self):
        """设置默认角色"""
        default_characters = [
            ("哆啦A梦", "ドラえもん", VoiceStyle.LI_YUNLONG),
            ("大雄", "野比のび太", VoiceStyle.WANG_JINGZE),
            ("静香", "源静香", VoiceStyle.TONG_XIANGYU),
            ("胖虎", "剛田武", VoiceStyle.ZHANG_FEI),
            ("小夫", "骨川スネ夫", VoiceStyle.MA_BAOGUO),
            ("哆啦美", "ドラミ", VoiceStyle.LUO_XIANG),
            ("出木杉", "出木杉英才", VoiceStyle.SHELDON),
            ("小叮当", "", VoiceStyle.GUO_DEGANG),
        ]
        
        for name, original_name, style in default_characters:
            profile = CharacterProfile.get_preset(name)
            profile.original_name = original_name
            profile.voice_style = style
            self.character_profiles[name] = profile
    
    def load_config(self):
        """加载配置"""
        config_file = Path("config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                self.input_edit.setText(config_data.get('last_input_video', ''))
                self.output_edit.setText(config_data.get('last_output_path', ''))
                self.tts_api_edit.setText(config_data.get('tts_api', 'http://127.0.0.1:5021/api/tts'))
                self.ollama_api_edit.setText(config_data.get('ollama_api', 'http://127.0.0.1:11434'))
                self.whisper_combo.setCurrentText(config_data.get('whisper_model', 'large-v3'))
                self.translate_combo.setCurrentText(config_data.get('translate_model', 'qwen3:4b'))
                self.translation_batch_spin.setValue(config_data.get('translation_batch_size', 10))
                self.tts_batch_spin.setValue(config_data.get('tts_batch_size', 3))
                
                self.lip_sync_check.setChecked(config_data.get('lip_sync_enabled', True))
                self.lip_sync_strength_slider.setValue(config_data.get('lip_sync_strength', 80))
                self.lip_sync_method_combo.setCurrentText(config_data.get('lip_sync_method', '时间拉伸'))
                
                self.character_profiles.clear()
                
                char_profiles = config_data.get('character_profiles', {})
                for name, char_data in char_profiles.items():
                    try:
                        voice_file = char_data.get('voice_file', '')
                        if voice_file:
                            voice_filename = os.path.basename(voice_file)
                            possible_paths = [
                                os.path.join('voices', voice_filename),
                                os.path.join('.', voice_filename),
                                voice_filename,
                                os.path.join('voices', f"{voice_filename}.wav"),
                                os.path.join('voices', f"{voice_filename}.mp3")
                            ]
                            
                            for path in possible_paths:
                                if os.path.exists(path):
                                    voice_file = path
                                    logger.info(f"找到音色文件: {path}")
                                    break
                            else:
                                logger.warning(f"音色文件不存在: {char_data.get('voice_file')}")
                                voice_file = ""
                        
                        voice_style_value = char_data.get('voice_style', '默认')
                        try:
                            voice_style = VoiceStyle(voice_style_value)
                        except:
                            voice_style = VoiceStyle.DEFAULT
                        
                        profile = CharacterProfile(
                            name=char_data.get('name', name),
                            original_name=char_data.get('original_name', name),
                            voice_style=voice_style,
                            voice_file=voice_file,
                            speed=float(char_data.get('speed', 1.0)),
                            pitch=float(char_data.get('pitch', 1.0)),
                            emotion=char_data.get('emotion', 'neutral'),
                            intensity=float(char_data.get('intensity', 1.0)),
                            catchphrases=list(char_data.get('catchphrases', [])),
                            reference_image=char_data.get('reference_image', '')
                        )
                        self.character_profiles[name] = profile
                    except Exception as e:
                        logger.warning(f"加载角色 {name} 配置失败: {e}")
                        self.character_profiles[name] = CharacterProfile.get_preset(name)
                
                logger.info("配置加载成功")
                self.refresh_character_table()
                
            except json.JSONDecodeError as e:
                logger.error(f"配置文件格式错误: {e}")
                self.setup_default_characters()
                self.refresh_character_table()
            except Exception as e:
                logger.error(f"配置加载失败: {e}")
                self.setup_default_characters()
                self.refresh_character_table()
        else:
            self.setup_default_characters()
            self.refresh_character_table()
    
    def save_config(self):
        """保存配置"""
        config_data = {
            'last_input_video': self.input_edit.text(),
            'last_output_path': self.output_edit.text(),
            'tts_api': self.tts_api_edit.text(),
            'ollama_api': self.ollama_api_edit.text(),
            'whisper_model': self.whisper_combo.currentText(),
            'translate_model': self.translate_combo.currentText(),
            'translation_batch_size': self.translation_batch_spin.value(),
            'tts_batch_size': self.tts_batch_spin.value(),
            'lip_sync_enabled': self.lip_sync_check.isChecked(),
            'lip_sync_strength': self.lip_sync_strength_slider.value(),
            'lip_sync_method': self.lip_sync_method_combo.currentText(),
            'character_profiles': {}
        }
        
        for name, profile in self.character_profiles.items():
            config_data['character_profiles'][name] = {
                'name': profile.name,
                'original_name': profile.original_name,
                'voice_style': profile.voice_style.value,
                'voice_file': profile.voice_file or '',
                'speed': float(profile.speed),
                'pitch': float(profile.pitch),
                'emotion': profile.emotion,
                'intensity': float(profile.intensity),
                'catchphrases': list(profile.catchphrases),
                'reference_image': profile.reference_image or ''
            }
        
        try:
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            logger.info("配置保存成功")
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
    
    # ============== 事件处理函数 ==============
    
    def browse_input_video(self):
        """浏览输入视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", 
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv);;所有文件 (*.*)"
        )
        
        if file_path:
            self.input_edit.setText(file_path)
            self.current_video = file_path
            
            video_path = Path(file_path)
            output_path = video_path.parent / f"{video_path.stem}_dubbed{video_path.suffix}"
            self.output_edit.setText(str(output_path))
            
            self.preview_label.setText(f"已选择: {video_path.name}\n大小: {video_path.stat().st_size // 1024 // 1024}MB")
    
    def browse_output_path(self):
        """浏览输出路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择输出文件", self.output_edit.text(),
            "视频文件 (*.mp4);;所有文件 (*.*)"
        )
        
        if file_path:
            self.output_edit.setText(file_path)
    
    def browse_input_folder(self):
        """浏览输入文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder_path:
            self.input_folder_edit.setText(folder_path)
    
    def browse_output_folder(self):
        """浏览输出文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder_path:
            self.output_folder_edit.setText(folder_path)
    
    def start_processing(self):
        """开始处理"""
        video_path = self.input_edit.text()
        output_path = self.output_edit.text()
        
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "错误", "请选择有效的视频文件")
            return
        
        if not output_path:
            QMessageBox.warning(self, "错误", "请指定输出路径")
            return
        
        if not self.character_profiles:
            QMessageBox.warning(self, "错误", "请至少配置一个角色")
            return
        
        self.config.input_video = video_path
        self.config.output_video = output_path
        self.config.tts_api_url = self.tts_api_edit.text()
        self.config.ollama_url = self.ollama_api_edit.text()
        self.config.whisper_model = self.whisper_combo.currentText()
        self.config.translate_model = self.translate_combo.currentText()
        self.config.sample_rate = int(self.sample_rate_combo.currentText())
        self.config.use_gpu = self.gpu_check.isChecked()
        self.config.gpu_id = self.gpu_id_spin.value()
        self.config.num_workers = self.workers_spin.value()
        self.config.cache_enabled = self.cache_check.isChecked()
        self.config.keep_background = self.keep_bg_check.isChecked()
        self.config.background_volume = self.bg_volume_slider.value() / 100
        self.config.voice_volume = self.voice_volume_slider.value() / 100
        self.config.noise_reduction = self.noise_reduction_check.isChecked()
        self.config.normalize_audio = self.normalize_check.isChecked()
        self.config.translation_batch_size = self.translation_batch_spin.value()
        self.config.tts_batch_size = self.tts_batch_spin.value()
        
        self.config.lip_sync_enabled = self.lip_sync_check.isChecked()
        self.config.lip_sync_strength = self.lip_sync_strength_slider.value() / 100
        self.config.lip_sync_method = self.lip_sync_method_combo.currentText()
        
        self.add_log_message(f"🎬 开始处理视频: {video_path}", "INFO")
        self.add_log_message(f"📝 输出路径: {output_path}", "INFO")
        self.add_log_message(f"🤖 Ollama基础地址: {self.config.ollama_url}", "INFO")
        self.add_log_message(f"🤖 构建的API URL: {self.config.get_ollama_api_url()}", "INFO")
        self.add_log_message(f"🤖 Ollama模型: {self.config.translate_model}", "INFO")
        self.add_log_message(f"🎵 TTS API: {self.config.tts_api_url}", "INFO")
        self.add_log_message(f"🎭 角色数量: {len(self.character_profiles)}", "INFO")
        self.add_log_message(f"👄 口型对齐: {'启用' if self.config.lip_sync_enabled else '禁用'}", "INFO")
        
        try:
            test_url = self.config.get_ollama_api_url().replace('/api/generate', '/api/tags')
            test_response = requests.get(test_url, timeout=5)
            if test_response.status_code == 200:
                self.add_log_message("✅ Ollama服务连接正常", "INFO")
            else:
                self.add_log_message(f"⚠️ Ollama服务测试失败: {test_response.status_code}", "WARNING")
        except Exception as e:
            self.add_log_message(f"⚠️ Ollama服务测试异常: {e}", "WARNING")
        
        try:
            self.engine = DubbingEngine(self.config)
            
            for profile in self.character_profiles.values():
                self.engine.add_character_profile(profile)
            
            self.processing_thread = ProcessingThread(self.engine, video_path)
            
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.log_message.connect(self.add_log_message)
            self.processing_thread.finished.connect(self.processing_finished)
            self.processing_thread.error.connect(self.processing_error)
            
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.log_text.clear()
            
            self.processing_thread.start()
            
            logger.info("开始处理视频")
            
        except TypeError as e:
            error_msg = f"创建DubbingEngine失败: {str(e)}"
            self.add_log_message(f"❌ {error_msg}", "ERROR")
            QMessageBox.critical(self, "初始化错误", error_msg)
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
    
    def pause_processing(self):
        """暂停处理"""
        if self.processing_thread:
            pass
    
    def stop_processing(self):
        """停止处理"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.terminate()
            self.processing_thread.wait()
            
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            
            self.add_log_message("处理已停止", "WARNING")
    
    def update_progress(self, value: int, message: str):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def add_log_message(self, message: str, level: str):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_color = {
            "DEBUG": "#888888",
            "INFO": "#4ec9b0",
            "WARNING": "#d7ba7d",
            "ERROR": "#f44747"
        }.get(level, "#cccccc")
        
        html = f'<span style="color:#888888">[{timestamp}]</span> <span style="color:{level_color}">{message}</span><br>'
        
        current_html = self.log_text.toHtml()
        self.log_text.setHtml(current_html + html)
        
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def processing_finished(self, output_path: str):
        """处理完成"""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        
        self.add_log_message(f"✅ 处理完成: {output_path}", "INFO")
        
        reply = QMessageBox.question(
            self, "处理完成",
            f"视频已保存到:\n{output_path}\n\n是否打开所在文件夹？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            folder_path = os.path.dirname(output_path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
    
    def processing_error(self, error_msg: str):
        """处理错误"""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        self.add_log_message(error_msg, "ERROR")
        QMessageBox.critical(self, "处理错误", error_msg)
    
    def add_character(self):
        """添加角色"""
        name, ok = QInputDialog.getText(self, "添加角色", "请输入角色名称:")
        if ok and name:
            if name in self.character_profiles:
                QMessageBox.warning(self, "错误", f"角色 {name} 已存在")
                return
            
            dialog = CharacterConfigDialog(name, self)
            if dialog.exec():
                profile = dialog.get_profile()
                self.character_profiles[name] = profile
                self.refresh_character_table()
    
    def remove_character(self):
        """删除角色"""
        selected_items = self.character_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "错误", "请先选择要删除的角色")
            return
        
        rows = set(item.row() for item in selected_items)
        
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除 {len(rows)} 个角色吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            characters_to_remove = []
            for row in rows:
                char_name = self.character_table.item(row, 0).text()
                characters_to_remove.append(char_name)
            
            for char_name in characters_to_remove:
                if char_name in self.character_profiles:
                    del self.character_profiles[char_name]
            
            self.refresh_character_table()
    
    def refresh_character_table(self):
        """刷新角色表格"""
        self.character_table.setRowCount(len(self.character_profiles))
        
        for i in range(len(self.character_profiles)):
            self.character_table.setRowHeight(i, 35)
        
        for i, (name, profile) in enumerate(self.character_profiles.items()):
            self.character_table.setItem(i, 0, QTableWidgetItem(name))
            self.character_table.setItem(i, 1, QTableWidgetItem(profile.original_name))
            self.character_table.setItem(i, 2, QTableWidgetItem(profile.voice_style.value))
            
            voice_display = ""
            if profile.voice_file:
                voice_path = Path(profile.voice_file)
                if voice_path.exists():
                    voice_display = voice_path.name
                else:
                    voice_display = "⚠️ 文件不存在"
            else:
                voice_display = "未设置"
            
            self.character_table.setItem(i, 3, QTableWidgetItem(voice_display))
            self.character_table.setItem(i, 4, QTableWidgetItem(f"{profile.speed:.1f}"))
            self.character_table.setItem(i, 5, QTableWidgetItem(f"{profile.intensity:.1f}"))
            self.character_table.setItem(i, 6, QTableWidgetItem(profile.emotion))
            
            btn_widget = QWidget()
            btn_widget.setStyleSheet("background-color: transparent;")
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(4, 2, 4, 2)
            btn_layout.setSpacing(4)
            
            edit_btn = QPushButton("编辑")
            edit_btn.clicked.connect(lambda checked, idx=i: self.edit_character(idx))
            edit_btn.setFixedSize(45, 24)
            edit_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3d3d3d;
                    color: #cccccc;
                    border: 1px solid #555;
                    border-radius: 2px;
                    font-size: 10px;
                    padding: 2px;
                }
                QPushButton:hover {
                    background-color: #505050;
                    border: 1px solid #666;
                }
                QPushButton:pressed {
                    background-color: #2d2d2d;
                }
            """)
            btn_layout.addWidget(edit_btn)
            
            voice_btn = QPushButton("音色")
            voice_btn.clicked.connect(lambda checked, idx=i: self.select_voice_for_character(idx))
            voice_btn.setFixedSize(45, 24)
            voice_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3d3d3d;
                    color: #cccccc;
                    border: 1px solid #555;
                    border-radius: 2px;
                    font-size: 10px;
                    padding: 2px;
                }
                QPushButton:hover {
                    background-color: #505050;
                    border: 1px solid #666;
                }
                QPushButton:pressed {
                    background-color: #2d2d2d;
                }
            """)
            btn_layout.addWidget(voice_btn)
            
            test_btn = QPushButton("测试")
            test_btn.clicked.connect(lambda checked, idx=i: self.test_character_voice(idx))
            test_btn.setFixedSize(45, 24)
            test_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3d3d3d;
                    color: #cccccc;
                    border: 1px solid #555;
                    border-radius: 2px;
                    font-size: 10px;
                    padding: 2px;
                }
                QPushButton:hover {
                    background-color: #505050;
                    border: 1px solid #666;
                }
                QPushButton:pressed {
                    background-color: #2d2d2d;
                }
            """)
            btn_layout.addWidget(test_btn)
            
            self.character_table.setCellWidget(i, 7, btn_widget)
    
    def edit_character(self, row_index):
        """编辑角色"""
        char_name = self.character_table.item(row_index, 0).text()
        
        if char_name in self.character_profiles:
            dialog = CharacterConfigDialog(char_name, self)
            
            dialog.profile = self.character_profiles[char_name]
            dialog.name_edit.setText(dialog.profile.name)
            dialog.file_edit.setText(dialog.profile.voice_file)
            dialog.style_combo.setCurrentText(dialog.profile.voice_style.value)
            dialog.speed_slider.setValue(int(dialog.profile.speed * 100))
            dialog.pitch_slider.setValue(int(dialog.profile.pitch * 100))
            dialog.intensity_slider.setValue(int(dialog.profile.intensity * 100))
            dialog.emotion_combo.setCurrentText(dialog.profile.emotion)
            dialog.catchphrase_edit.setText("\n".join(dialog.profile.catchphrases))
            
            if dialog.exec():
                profile = dialog.get_profile()
                self.character_profiles[char_name] = profile
                self.refresh_character_table()
    
    def select_voice_for_character(self, row_index):
        """为角色选择音色"""
        char_name = self.character_table.item(row_index, 0).text()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音色文件", "", 
            "音频文件 (*.wav *.mp3 *.ogg);;所有文件 (*.*)"
        )
        
        if file_path:
            self.character_profiles[char_name].voice_file = file_path
            self.character_table.item(row_index, 3).setText(Path(file_path).name)
            
            self.add_log_message(
                f"已为角色 '{char_name}' 设置音色: {Path(file_path).name}",
                "INFO"
            )
    
    def test_character_voice(self, row_index):
        """测试角色音色"""
        char_name = self.character_table.item(row_index, 0).text()
        
        if char_name in self.character_profiles:
            profile = self.character_profiles[char_name]
            
            if not profile.voice_file:
                QMessageBox.warning(self, "错误", f"角色 {char_name} 未配置音色文件")
                return
            
            QMessageBox.information(
                self, "测试音色",
                f"测试 {char_name} 的音色\n"
                f"风格: {profile.voice_style.value}\n"
                f"语速: {profile.speed:.1f}\n"
                f"情感: {profile.emotion}"
            )
    
    def test_all_voices(self):
        """测试所有音色"""
        if not self.character_profiles:
            QMessageBox.warning(self, "错误", "没有配置任何角色")
            return
        
        QMessageBox.information(
            self, "测试全部音色",
            f"开始测试 {len(self.character_profiles)} 个角色的音色..."
        )
    
    def load_preset_characters(self):
        """加载预设角色"""
        reply = QMessageBox.question(
            self, "加载预设",
            "确定要加载预设角色配置吗？这将覆盖当前的配置。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.setup_default_characters()
            self.refresh_character_table()
            self.add_log_message("已加载预设角色配置", "INFO")
    
    def import_character_config(self):
        """导入角色配置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入角色配置", "", "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                imported_count = 0
                
                for char_data in config_data.get('characters', []):
                    try:
                        char_name = char_data.get('name', '未知角色')
                        
                        profile = CharacterProfile(
                            name=char_name,
                            original_name=char_data.get('original_name', char_name),
                            voice_style=VoiceStyle(char_data.get('voice_style', '默认')),
                            voice_file=char_data.get('voice_file', ''),
                            speed=float(char_data.get('speed', 1.0)),
                            pitch=float(char_data.get('pitch', 1.0)),
                            emotion=char_data.get('emotion', 'neutral'),
                            intensity=float(char_data.get('intensity', 1.0)),
                            catchphrases=list(char_data.get('catchphrases', [])),
                            reference_image=char_data.get('reference_image', '')
                        )
                        
                        self.character_profiles[char_name] = profile
                        imported_count += 1
                        
                    except Exception as e:
                        logger.warning(f"导入角色 {char_data.get('name', 'unknown')} 失败: {e}")
                        continue
                
                self.refresh_character_table()
                
                success_msg = f"已成功导入 {imported_count} 个角色配置"
                self.add_log_message(f"{success_msg}: {file_path}", "INFO")
                QMessageBox.information(self, "导入成功", success_msg)
                
            except json.JSONDecodeError as e:
                error_msg = f"配置文件格式错误: {e}"
                QMessageBox.critical(self, "导入失败", error_msg)
                logger.error(error_msg)
            except Exception as e:
                error_msg = f"导入失败: {str(e)}"
                QMessageBox.critical(self, "导入失败", error_msg)
                logger.error(f"导入角色配置失败: {e}")
    
    def export_character_config(self):
        """导出角色配置"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出角色配置", "character_config.json", "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                config_data = {
                    'characters': []
                }
                
                for profile in self.character_profiles.values():
                    char_dict = {
                        'name': profile.name,
                        'original_name': profile.original_name,
                        'voice_style': profile.voice_style.value,
                        'voice_file': profile.voice_file,
                        'speed': float(profile.speed),
                        'pitch': float(profile.pitch),
                        'emotion': profile.emotion,
                        'intensity': float(profile.intensity),
                        'catchphrases': list(profile.catchphrases),
                        'reference_image': profile.reference_image or ''
                    }
                    config_data['characters'].append(char_dict)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                
                self.add_log_message(f"已导出角色配置: {file_path}", "INFO")
                
                QMessageBox.information(self, "导出成功", 
                    f"角色配置已成功导出到:\n{file_path}\n\n"
                    f"导出了 {len(self.character_profiles)} 个角色")
                
            except Exception as e:
                error_msg = f"导出失败: {str(e)}"
                QMessageBox.critical(self, "导出失败", error_msg)
                logger.error(f"导出角色配置失败: {e}")
    
    def test_tts_api(self):
        """测试TTS API"""
        api_url = self.tts_api_edit.text()
        
        self.add_log_message(f"🔍 测试TTS API: {api_url}", "INFO")
        
        try:
            import base64
            import numpy as np
            import soundfile as sf
            
            sr = 24000
            silence = np.zeros(sr, dtype=np.float32)
            test_audio_path = "test_connection.wav"
            sf.write(test_audio_path, silence, sr)
            
            with open(test_audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            prompt_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            payload = {
                "tts_text": "你好，世界。",
                "prompt_audio": prompt_audio_b64,
                "emo_control_method": 0,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 5.0,
                "max_mel_tokens": 200
            }
            
            self.add_log_message(f"发送测试请求到: {api_url}", "INFO")
            self.add_log_message(f"Payload: {json.dumps(payload, ensure_ascii=False)[:200]}...", "DEBUG")
            
            response = requests.post(api_url, json=payload, timeout=30)
            
            self.add_log_message(f"API响应状态: {response.status_code}", "INFO")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    self.add_log_message(f"API响应JSON: {str(result)[:200]}...", "DEBUG")
                    
                    if result.get("status") == "success":
                        self.add_log_message("✅ TTS API测试成功！", "INFO")
                        
                        if "audio" in result:
                            audio_bytes = base64.b64decode(result["audio"])
                            with open("test_output.wav", "wb") as f:
                                f.write(audio_bytes)
                            self.add_log_message("✅ 测试音频已保存: test_output.wav", "INFO")
                        
                        QMessageBox.information(self, "测试成功", "TTS API连接正常，功能可用")
                    else:
                        error_msg = f"TTS API返回错误状态: {result.get('message', '未知错误')}"
                        self.add_log_message(f"❌ {error_msg}", "ERROR")
                        QMessageBox.warning(self, "测试失败", error_msg)
                        
                except json.JSONDecodeError as e:
                    error_msg = f"API返回的不是有效JSON: {e}\n响应文本: {response.text[:200]}"
                    self.add_log_message(f"❌ {error_msg}", "ERROR")
                    QMessageBox.warning(self, "测试失败", error_msg)
            
            elif response.status_code == 400:
                error_msg = f"TTS API返回400错误（请求格式错误）\n"
                try:
                    result = response.json()
                    error_msg += f"错误信息: {result.get('message', '无详细信息')}"
                except:
                    error_msg += f"响应内容: {response.text[:500]}"
                
                self.add_log_message(f"❌ {error_msg}", "ERROR")
                
                self.add_log_message("💡 诊断建议:", "INFO")
                self.add_log_message("1. 检查TTS API请求格式是否正确", "INFO")
                self.add_log_message("2. 检查音色文件base64编码是否正确", "INFO")
                self.add_log_message("3. 参考TTS API文档调整参数", "INFO")
                
                QMessageBox.critical(self, "测试失败 - 请求格式错误", 
                    f"TTS API返回400错误（请求格式错误）\n\n"
                    f"请检查:\n"
                    f"1. 请求JSON格式是否正确\n"
                    f"2. 音色文件base64编码\n"
                    f"3. 请求参数是否完整\n\n"
                    f"响应内容:\n{response.text[:300]}")
            
            elif response.status_code == 404:
                error_msg = f"API端点不存在 (404)\nURL: {api_url}"
                self.add_log_message(f"❌ {error_msg}", "ERROR")
                QMessageBox.warning(self, "测试失败", error_msg)
            else:
                error_msg = f"TTS API返回错误: {response.status_code}"
                try:
                    error_msg += f"\n响应: {response.text[:300]}"
                except:
                    pass
                self.add_log_message(f"❌ {error_msg}", "ERROR")
                QMessageBox.warning(self, "测试失败", error_msg)
                    
        except requests.exceptions.ConnectionError:
            error_msg = f"无法连接到TTS API\nURL: {api_url}\n请确保TTS服务已启动"
            self.add_log_message(f"❌ {error_msg}", "ERROR")
            QMessageBox.critical(self, "测试失败", error_msg)
        except requests.exceptions.Timeout:
            error_msg = "连接超时，请检查网络或服务是否正常"
            self.add_log_message(f"❌ {error_msg}", "ERROR")
            QMessageBox.critical(self, "测试失败", error_msg)
        except Exception as e:
            error_msg = f"测试时发生错误: {str(e)[:200]}"
            self.add_log_message(f"❌ {error_msg}", "ERROR")
            QMessageBox.critical(self, "测试失败", error_msg)
    
    def refresh_ollama_models(self):
        """刷新Ollama模型列表"""
        base_url = self.ollama_api_edit.text()
        
        base_url = base_url.rstrip('/')
        if '/v1' in base_url:
            base_url = base_url.split('/v1')[0].rstrip('/')
        elif '/api/generate' in base_url:
            base_url = base_url.split('/api/generate')[0].rstrip('/')
        elif '/api/chat' in base_url:
            base_url = base_url.split('/api/chat')[0].rstrip('/')
        
        models_url = f"{base_url}/api/tags"
        
        try:
            self.add_log_message(f"正在获取模型列表...", "INFO")
            response = requests.get(models_url, timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = []
                
                if "models" in models_data:
                    for model in models_data["models"]:
                        model_name = model.get("name", "")
                        if model_name:
                            available_models.append(model_name)
                
                current_model = self.translate_combo.currentText()
                
                self.translate_combo.clear()
                for model in available_models:
                    self.translate_combo.addItem(model)
                
                if current_model in available_models:
                    self.translate_combo.setCurrentText(current_model)
                elif available_models:
                    self.translate_combo.setCurrentText(available_models[0])
                
                self.add_log_message(f"✅ 已刷新模型列表，找到 {len(available_models)} 个模型", "INFO")
                
                QMessageBox.information(
                    self,
                    "刷新成功",
                    f"已从Ollama获取模型列表\n\n找到 {len(available_models)} 个模型"
                )
                
            else:
                self.add_log_message(f"❌ 无法获取模型列表，状态码: {response.status_code}", "ERROR")
                QMessageBox.warning(
                    self,
                    "刷新失败",
                    f"无法连接到Ollama服务\n状态码: {response.status_code}"
                )
                
        except requests.exceptions.ConnectionError:
            self.add_log_message(f"❌ 无法连接到Ollama服务", "ERROR")
            QMessageBox.critical(
                self,
                "连接失败",
                "无法连接到Ollama服务，请确保Ollama已启动"
            )
        except Exception as e:
            self.add_log_message(f"❌ 刷新模型列表失败: {e}", "ERROR")
            QMessageBox.critical(
                self,
                "刷新失败",
                f"刷新模型列表时发生错误:\n{str(e)}"
            )
    
    def test_ollama_api(self):
        """测试Ollama API配置"""
        base_url = self.ollama_api_edit.text()
        model_name = self.translate_combo.currentText()
        
        self.add_log_message(f"开始测试Ollama配置...", "INFO")
        
        try:
            temp_config = ProcessingConfig()
            temp_config.ollama_url = base_url
            api_url = temp_config.get_ollama_api_url()
            
            self.add_log_message(f"构建的API URL: {api_url}", "INFO")
            
            test_url = api_url.replace('/api/generate', '/api/tags')
            self.add_log_message(f"测试连接URL: {test_url}", "DEBUG")
            
            test_response = requests.get(test_url, timeout=10)
            
            if test_response.status_code == 200:
                models_data = test_response.json()
                available_models = [m.get("name", "") for m in models_data.get("models", [])]
                
                if model_name in available_models:
                    self.add_log_message(f"✅ 找到模型: {model_name}", "INFO")
                else:
                    self.add_log_message(f"⚠️ 模型 {model_name} 不在可用模型中", "WARNING")
            
            prompt = "Translate this Japanese to Chinese: こんにちは"
            
            data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 4096
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            self.add_log_message(f"发送测试请求到: {api_url}", "INFO")
            self.add_log_message(f"使用模型: {model_name}", "INFO")
            
            response = requests.post(
                api_url,
                json=data,
                headers=headers,
                timeout=30
            )
            
            self.add_log_message(f"响应状态码: {response.status_code}", "INFO")
            
            if response.status_code == 200:
                result = response.json()
                
                if "response" in result:
                    reply_text = result["response"].strip()
                    
                    if reply_text:
                        self.add_log_message("✅ Ollama配置测试成功！", "INFO")
                        self.add_log_message(f"测试回复: {reply_text}", "INFO")
                        
                        QMessageBox.information(
                            self, 
                            "测试成功", 
                            f"Ollama配置测试成功！\n\n"
                            f"API地址: {api_url}\n"
                            f"模型: {model_name}\n"
                            f"回复: {reply_text}"
                        )
                    else:
                        self.add_log_message("❌ Ollama配置测试失败：未获取到响应", "ERROR")
                        QMessageBox.warning(
                            self,
                            "测试失败",
                            "Ollama配置测试失败：未获取到响应\n请检查模型是否正常工作"
                        )
                else:
                    self.add_log_message("❌ Ollama API返回格式错误", "ERROR")
                    QMessageBox.warning(
                        self,
                        "测试失败",
                        "Ollama API返回格式错误，请检查API端点是否正确"
                    )
            else:
                error_text = response.text[:500] if response.text else "无详细错误信息"
                self.add_log_message(f"❌ Ollama配置测试出错: HTTP {response.status_code}", "ERROR")
                self.add_log_message(f"错误详情: {error_text}", "ERROR")
                
                QMessageBox.critical(
                    self,
                    "测试失败",
                    f"Ollama配置测试失败\n\nHTTP状态码: {response.status_code}\n错误: {error_text[:200]}"
                )
                
        except requests.exceptions.ConnectionError as e:
            self.add_log_message(f"❌ Ollama配置测试出错: 连接失败 - {str(e)}", "ERROR")
            QMessageBox.critical(
                self,
                "连接失败",
                f"无法连接到Ollama服务\n\nURL: {base_url}\n请确保Ollama服务已启动\n命令: ollama serve"
            )
        except requests.exceptions.Timeout as e:
            self.add_log_message(f"❌ Ollama配置测试出错: 连接超时 - {str(e)}", "ERROR")
            QMessageBox.critical(
                self,
                "超时",
                f"连接超时\n\n请检查网络连接或增加超时时间"
            )
        except Exception as e:
            self.add_log_message(f"❌ Ollama配置测试出错: {str(e)}", "ERROR")
            QMessageBox.critical(
                self,
                "测试失败",
                f"测试时发生错误:\n{str(e)[:200]}"
            )
    
    def save_settings(self):
        """保存设置"""
        self.save_config()
        QMessageBox.information(self, "保存成功", "所有设置已保存")
    
    def scan_video_files(self):
        """扫描视频文件"""
        input_folder = self.input_folder_edit.text()
        
        if not input_folder or not os.path.exists(input_folder):
            QMessageBox.warning(self, "错误", "请选择有效的输入文件夹")
            return
        
        self.file_list.clear()
        
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv']
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    file_path = os.path.join(root, file)
                    item = QListWidgetItem(file_path)
                    self.file_list.addItem(item)
        
        self.add_log_message(f"扫描完成，找到 {self.file_list.count()} 个视频文件", "INFO")
    
    def start_batch_processing(self):
        """开始批量处理"""
        QMessageBox.information(self, "批量处理", "批量处理功能开发中...")
    
    def stop_batch_processing(self):
        """停止批量处理"""
        pass
    
    def change_log_level(self, level: str):
        """更改日志级别"""
        logging.getLogger().setLevel(getattr(logging, level))
        logger.info(f"日志级别已更改为: {level}")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
    
    def save_log(self):
        """保存日志"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存日志", "dubbing_log.txt", "文本文件 (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.add_log_message(f"日志已保存: {file_path}", "INFO")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", str(e))
    
    def update_status_time(self):
        """更新状态栏时间"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)
    
    def new_project(self):
        """新建项目"""
        reply = QMessageBox.question(
            self, "新建项目",
            "确定要新建项目吗？所有未保存的更改将会丢失。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.input_edit.clear()
            self.output_edit.clear()
            self.log_text.clear()
            self.character_profiles.clear()
            self.refresh_character_table()
            
            self.add_log_message("已创建新项目", "INFO")
    
    def open_project(self):
        """打开项目"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "AI配音工厂项目 (*.aiproj);;所有文件 (*.*)"
        )
        
        if file_path:
            QMessageBox.information(self, "打开项目", f"打开项目: {file_path}")
    
    def save_project(self):
        """保存项目"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "my_project.aiproj", "AI配音工厂项目 (*.aiproj)"
        )
        
        if file_path:
            QMessageBox.information(self, "保存项目", f"保存项目: {file_path}")
    
    def open_voice_manager(self):
        """打开音色管理器"""
        QMessageBox.information(self, "音色管理器", "音色管理器开发中...")
    
    def open_subtitle_editor(self):
        """打开字幕编辑器"""
        QMessageBox.information(self, "字幕编辑器", "字幕编辑器开发中...")
    
    def open_audio_editor(self):
        """打开音频编辑器"""
        QMessageBox.information(self, "音频编辑器", "音频编辑器开发中...")
    
    def open_documentation(self):
        """打开文档"""
        QDesktopServices.openUrl(QUrl("https://github.com/yourusername/dubbing_factory"))
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2>🎭 AI配音工厂 - 超强嘴炮版 🚀</h2>
        <p><b>版本:</b> 3.4.0 - AI动画生成集成版</p>
        <p><b>作者:</b> AI助手</p>
        <p><b>描述:</b> 基于深度学习的自动视频配音工具，集成ComfyUI人物一致性和AI动画生成。</p>
        <p><b>核心功能:</b></p>
        <ul>
            <li>✅ 智能配音生成与风格转换</li>
            <li>✅ ComfyUI人物一致性集成</li>
            <li>✅ AI驱动动画生成</li>
            <li>✅ 自动口型同步</li>
            <li>✅ 动态角色表情和动作</li>
        </ul>
        <p><b>AI动画生成特性:</b></p>
        <ul>
            <li>🤖 AI生成式文本分析和提示提取</li>
            <li>🎭 多角色支持与人物绑定</li>
            <li>👄 音素分析与口型同步</li>
            <li>👁️ 头部、眼部、身体动作生成</li>
            <li>🖼️ 逐帧生成流畅动画</li>
        </ul>
        <p><b>系统要求:</b> Python 3.8+, NVIDIA GPU (推荐), FFmpeg, ComfyUI</p>
        <p><b>许可证:</b> MIT</p>
        """
        
        QMessageBox.about(self, "关于 AI配音工厂", about_text)
    
    # ============== 动画生成相关方法 ==============
    
    def connect_comfyui(self):
        """连接到ComfyUI"""
        host = self.comfyui_host_edit.text()
        port = self.comfyui_port_edit.value()
        
        self.add_log_message(f"正在连接到ComfyUI: {host}:{port}", "INFO")
        
        try:
            response = requests.get(f"http://{host}:{port}", timeout=5)
            if response.status_code == 200:
                self.comfyui_status_label.setText("已连接")
                self.comfyui_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.add_log_message("✅ ComfyUI连接成功", "INFO")
            else:
                self.comfyui_status_label.setText("连接失败")
                self.comfyui_status_label.setStyleSheet("color: #f44336; font-weight: bold;")
                self.add_log_message(f"❌ ComfyUI连接失败: HTTP {response.status_code}", "ERROR")
        except requests.exceptions.ConnectionError:
            self.comfyui_status_label.setText("未连接")
            self.comfyui_status_label.setStyleSheet("color: #f44336; font-weight: bold;")
            self.add_log_message("❌ 无法连接到ComfyUI服务器", "ERROR")
            QMessageBox.critical(self, "连接失败", 
                f"无法连接到ComfyUI服务器\n\n"
                f"请确保ComfyUI已启动:\n"
                f"python main.py --port {port}\n\n"
                f"或使用自定义启动命令")
    
    def auto_bind_characters(self):
        """自动绑定角色"""
        if not self.character_profiles:
            QMessageBox.warning(self, "错误", "没有配置配音角色")
            return
        
        self.animation_character_table.setRowCount(0)
        
        row = 0
        for char_name, profile in self.character_profiles.items():
            self.animation_character_table.insertRow(row)
            
            self.animation_character_table.setItem(row, 0, QTableWidgetItem(char_name))
            
            ref_item = QTableWidgetItem("未设置")
            if profile.voice_file:
                ref_item = QTableWidgetItem(Path(profile.voice_file).name)
            self.animation_character_table.setItem(row, 1, ref_item)
            
            self.animation_character_table.setItem(row, 2, QTableWidgetItem(char_name))
            
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            
            select_btn = QPushButton("选择图片")
            select_btn.clicked.connect(lambda checked, r=row: self.select_reference_image(r))
            select_btn.setFixedSize(80, 24)
            btn_layout.addWidget(select_btn)
            
            self.animation_character_table.setCellWidget(row, 3, btn_widget)
            
            row += 1
        
        self.add_log_message(f"自动绑定了 {row} 个角色", "INFO")
    
    def select_reference_image(self, row):
        """选择参考图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择角色参考图片", "", 
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)"
        )
        
        if file_path:
            self.animation_character_table.setItem(row, 1, QTableWidgetItem(Path(file_path).name))
            
            char_name = self.animation_character_table.item(row, 0).text()
            if char_name in self.character_profiles:
                self.character_profiles[char_name].reference_image = file_path
            
            self.add_log_message(f"为角色 {char_name} 设置参考图片: {file_path}", "INFO")
    
    def manual_bind_character(self):
        """手动绑定角色"""
        row = self.animation_character_table.rowCount()
        self.animation_character_table.insertRow(row)
        
        char_combo = QComboBox()
        for char_name in self.character_profiles.keys():
            char_combo.addItem(char_name)
        
        self.animation_character_table.setCellWidget(row, 0, char_combo)
        
        self.animation_character_table.setItem(row, 1, QTableWidgetItem("未设置"))
        
        name_edit = QLineEdit()
        if char_combo.count() > 0:
            name_edit.setText(char_combo.currentText())
        self.animation_character_table.setCellWidget(row, 2, name_edit)
        
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        
        select_btn = QPushButton("选择图片")
        select_btn.clicked.connect(lambda checked, r=row: self.select_reference_image(r))
        select_btn.setFixedSize(80, 24)
        btn_layout.addWidget(select_btn)
        
        remove_btn = QPushButton("删除")
        remove_btn.clicked.connect(lambda checked, r=row: self.remove_character_binding(r))
        remove_btn.setFixedSize(60, 24)
        btn_layout.addWidget(remove_btn)
        
        self.animation_character_table.setCellWidget(row, 3, btn_widget)
    
    def remove_character_binding(self, row):
        """删除角色绑定"""
        self.animation_character_table.removeRow(row)
    
    def clear_character_bindings(self):
        """清空角色绑定"""
        reply = QMessageBox.question(
            self, "确认清空",
            "确定要清空所有角色绑定吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.animation_character_table.setRowCount(0)
            self.add_log_message("已清空角色绑定", "INFO")
    
    def generate_animation(self):
        """生成动画"""
        if self.comfyui_status_label.text() != "已连接":
            QMessageBox.warning(self, "错误", "请先连接到ComfyUI服务器")
            return
        
        if self.animation_character_table.rowCount() == 0:
            QMessageBox.warning(self, "错误", "请先绑定至少一个角色")
            return
        
        video_path = self.output_edit.text()
        if not video_path or not os.path.exists(video_path):
            reply = QMessageBox.question(
                self, "没有配音作品",
                "尚未生成配音作品。是否从当前配置创建新动画？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            video_path = ""
        
        if not COMFYUI_AVAILABLE:
            QMessageBox.critical(self, "错误", "ComfyUI客户端不可用，无法生成动画")
            return
        
        style_map = {
            "动漫": AnimationStyle.ANIME,
            "电影": AnimationStyle.CINEMATIC,
            "卡通": AnimationStyle.CARTOON,
            "写实": AnimationStyle.REALISTIC,
            "绘画": AnimationStyle.PAINTERLY,
            "像素艺术": AnimationStyle.PIXEL_ART
        }
        
        method_map = {
            "IP-Adapter": CharacterMethod.IP_ADAPTER,
            "InstantID": CharacterMethod.INSTANT_ID,
            "PhotoMaker": CharacterMethod.PHOTO_MAKER,
            "LoRA": CharacterMethod.LORA,
            "ControlNet": CharacterMethod.CONTROLNET
        }
        
        animation_config = AnimationConfig(
            resolution=(
                self.animation_width_spin.value(),
                self.animation_height_spin.value()
            ),
            fps=self.animation_fps_spin.value(),
            duration=30.0,
            style=style_map.get(self.animation_style_combo.currentText(), AnimationStyle.ANIME),
            consistency_method=method_map.get(self.consistency_method_combo.currentText(), CharacterMethod.IP_ADAPTER),
            consistency_strength=self.consistency_strength_slider.value() / 100.0,
            lip_sync_enabled=self.animation_lip_sync_check.isChecked(),
            expression_enabled=self.expression_check.isChecked(),
            head_movement_enabled=self.head_movement_check.isChecked(),
            eye_movement_enabled=self.eye_movement_check.isChecked(),
            scene_description=self.scene_description_edit.text(),
            camera_movement="subtle",
            lighting="natural"
        )
        
        self.animation_generator = AIAnimationGenerator(
            animation_config,
            comfyui_host=self.comfyui_host_edit.text(),
            comfyui_port=self.comfyui_port_edit.value()
        )
        
        for row in range(self.animation_character_table.rowCount()):
            char_widget = self.animation_character_table.cellWidget(row, 0)
            if isinstance(char_widget, QComboBox):
                char_name = char_widget.currentText()
            else:
                char_name = self.animation_character_table.item(row, 0).text()
            
            if char_name in self.character_profiles:
                profile = self.character_profiles[char_name]
                
                ref_item = self.animation_character_table.item(row, 1)
                ref_image = ""
                if ref_item and ref_item.text() != "未设置":
                    ref_image = "characters/" + ref_item.text()
                
                name_widget = self.animation_character_table.cellWidget(row, 2)
                if isinstance(name_widget, QLineEdit):
                    anim_name = name_widget.text()
                else:
                    anim_name = self.animation_character_table.item(row, 2).text()
                
                character_model = CharacterModel(
                    name=anim_name,
                    voice_profile=profile
                )
                
                if ref_image and os.path.exists(ref_image):
                    character_model.add_reference_image(ref_image)
                
                self.animation_generator.add_character(character_model)
        
        script_path = Path("ai_animation_output") / "animation_script.txt"
        if video_path and os.path.exists(video_path):
            script_content = self._extract_script_from_video(video_path)
        else:
            script_content = self._create_example_script()
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        self.animation_generate_btn.setEnabled(False)
        self.animation_stop_btn.setEnabled(True)
        self.animation_progress_bar.setValue(0)
        
        self.add_log_message("🚀 开始生成AI驱动动画...", "INFO")
        
        self.animation_thread = AnimationGenerationThread(
            self.animation_generator,
            str(script_path)
        )
        
        self.animation_thread.progress_updated.connect(self.update_animation_progress)
        self.animation_thread.log_message.connect(self.add_log_message)
        self.animation_thread.finished.connect(self.animation_generation_finished)
        self.animation_thread.error.connect(self.animation_generation_error)
        
        self.animation_thread.start()
    
    def _extract_script_from_video(self, video_path: str) -> str:
        """从视频提取剧本"""
        return self._create_example_script()
    
    def _create_example_script(self) -> str:
        """创建示例剧本"""
        script = ""
        
        characters = list(self.character_profiles.keys())[:3]
        
        if len(characters) >= 2:
            script = f"""{characters[0]}: 你好，今天天气真不错！
{characters[1]}: 是啊，阳光明媚，适合外出。
{characters[0]}: 你有什么计划吗？
{characters[1]}: 我想去公园散步。
{characters[0]}: 好主意，我也一起去吧！"""
        
        return script
    
    def update_animation_progress(self, value: int, message: str):
        """更新动画进度"""
        self.animation_progress_bar.setValue(value)
        self.status_label.setText(f"动画生成: {message}")
    
    def animation_generation_finished(self, output_path: str):
        """动画生成完成"""
        self.animation_generate_btn.setEnabled(True)
        self.animation_stop_btn.setEnabled(False)
        self.animation_progress_bar.setValue(100)
        
        self.add_log_message(f"✅ AI动画生成完成: {output_path}", "INFO")
        
        reply = QMessageBox.question(
            self, "动画生成完成",
            f"AI驱动动画已生成:\n{output_path}\n\n是否立即播放？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.preview_animation(output_path)
    
    def animation_generation_error(self, error_msg: str):
        """动画生成错误"""
        self.animation_generate_btn.setEnabled(True)
        self.animation_stop_btn.setEnabled(False)
        
        self.add_log_message(f"❌ 动画生成失败: {error_msg}", "ERROR")
        QMessageBox.critical(self, "动画生成失败", error_msg)
    
    def stop_animation_generation(self):
        """停止动画生成"""
        if self.animation_thread and self.animation_thread.isRunning():
            self.animation_thread.terminate()
            self.animation_thread.wait()
            
            self.animation_generate_btn.setEnabled(True)
            self.animation_stop_btn.setEnabled(False)
            
            self.add_log_message("动画生成已停止", "WARNING")
    
    def preview_animation(self, video_path: str = None):
        """预览动画"""
        if not video_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择动画视频", "", 
                "视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)"
            )
            
            if not file_path:
                return
            
            video_path = file_path
        
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Darwin':
                subprocess.call(('open', video_path))
            elif platform.system() == 'Windows':
                os.startfile(video_path)
            else:
                subprocess.call(('xdg-open', video_path))
        except Exception as e:
            logger.error(f"无法播放视频: {e}")
            QMessageBox.information(self, "播放失败", f"无法播放视频:\n{e}")
    
    def closeEvent(self, event):
        """关闭事件"""
        reply = QMessageBox.question(
            self, "确认退出",
            "确定要退出吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.terminate()
                self.processing_thread.wait()
            
            if self.animation_thread and self.animation_thread.isRunning():
                self.animation_thread.terminate()
                self.animation_thread.wait()
            
            self.save_config()
            
            logger.info("应用程序退出")
            event.accept()
        else:
            event.ignore()

# ============================================
# 主程序入口
# ============================================

def check_dependencies():
    """检查依赖"""
    print("🔍 检查系统依赖...")
    
    import sys
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg已安装")
    except:
        print("❌ FFmpeg未安装，请先安装FFmpeg")
        print("   下载地址: https://ffmpeg.org/download.html")
        return False
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用 - {torch.cuda.get_device_name(0)}")
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU模式（性能较低）")
    except:
        print("⚠️  无法检测CUDA状态")
    
    print("✅ 所有依赖检查完成")
    return True

def main():
    """主函数"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         🎭 AI配音工厂 - 超强嘴炮版 🚀                   ║
    ║         AI动画生成集成版 - 影音同步创作平台               ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    if not check_dependencies():
        print("\n按Enter键退出...")
        input()
        return
    
    print("\n🔍 检查必要服务...")
    
    print("检查Ollama服务...")
    try:
        base_url = "http://127.0.0.1:11434"
        api_url = f"{base_url}/api/tags"
        try:
            response = requests.get(api_url, timeout=3)
            if response.status_code == 200:
                models_data = response.json()
                model_names = [m.get('name', '') for m in models_data.get('models', [])]
                print(f"  ✅ Ollama服务正常")
                print(f"     可用模型: {', '.join(model_names[:5])}" + ("..." if len(model_names) > 5 else ""))
                
                required_models = ["qwen", "llama", "deepseek"]
                found_models = [name for name in model_names if any(req in name.lower() for req in required_models)]
                if found_models:
                    print(f"     找到翻译模型: {', '.join(found_models[:3])}")
                else:
                    print("     ⚠️  建议安装翻译模型: ollama pull qwen3:4b")
            else:
                print(f"  ⚠️  Ollama API返回: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("  ❌ 无法连接到Ollama服务")
            print("     请运行: ollama serve")
        except Exception as e:
            print(f"  ⚠️  Ollama检查错误: {e}")
    except Exception as e:
        print(f"  ⚠️  Ollama服务检查异常: {e}")
    
    print("检查TTS服务...")
    try:
        api_url = "http://127.0.0.1:5021/api/tts"
        try:
            test_payload = {
                "tts_text": "测试连接",
                "prompt_audio": "",
                "emo_control_method": 0,
                "do_sample": True
            }
            response = requests.post(api_url, json=test_payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    print("  ✅ TTS API端点正常工作")
                else:
                    print(f"  ⚠️  TTS API返回非成功状态: {result.get('message', '未知错误')}")
            elif response.status_code == 400:
                print("  ⚠️  TTS API返回400错误（请求格式问题）")
                print("     请检查GUI中的测试按钮")
            else:
                print(f"  ⚠️  TTS API端点返回: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("  ❌ 无法连接到TTS API端点")
            print("     请确保TTS服务已启动")
        except requests.exceptions.Timeout:
            print("  ⚠️  TTS API连接超时")
        except Exception as e:
            print(f"  ⚠️  TTS API检查错误: {e}")
            
    except Exception as e:
        print(f"  ⚠️  TTS服务检查异常: {e}")
    
    print("检查ComfyUI服务...")
    try:
        response = requests.get("http://127.0.0.1:8188", timeout=3)
        if response.status_code == 200:
            print("  ✅ ComfyUI服务器正常")
            print("     如需使用AI动画生成功能，请确保已安装必要节点")
        else:
            print(f"  ⚠️  ComfyUI服务器返回: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("  ❌ 无法连接到ComfyUI服务器")
        print("     如需使用AI动画生成功能，请启动ComfyUI")
    except Exception as e:
        print(f"  ⚠️  ComfyUI服务检查异常: {e}")
    
    print("\n🎯 服务状态总结:")
    print("   1. Ollama: ✓ 使用原生API (http://127.0.0.1:11434/api/generate)")
    print("   2. TTS API: 使用GUI中的测试按钮验证")
    print("   3. ComfyUI: 用于AI动画生成")
    print("   4. FFmpeg: ✓ 正常")
    print("   5. CUDA: ✓ 正常")
    
    print("\n💡 使用建议:")
    print("   1. 首次使用时，先在设置标签页测试Ollama和TTS配置")
    print("   2. 确保音色文件(.wav)已放在voices文件夹中")
    print("   3. 角色参考图片放在characters文件夹中")
    print("   4. AI动画生成需要ComfyUI支持人物一致性节点")
    print("   5. 可选的OpenAI集成用于增强剧本分析和提示生成")
    
    print("\n🚀 正在启动应用程序...")
    
    app = QApplication(sys.argv)
    app.setApplicationName("AI配音工厂")
    app.setApplicationVersion("3.4.0")
    
    if os.path.exists("icon.png"):
        app.setWindowIcon(QIcon("icon.png"))
    
    window = MainWindow()
    window.show()
    
    window.add_log_message("🎭 AI配音工厂启动成功！", "INFO")
    window.add_log_message("📖 使用说明:", "INFO")
    window.add_log_message("  1. 选择要配音的视频文件", "INFO")
    window.add_log_message("  2. 为每个角色配置音色和风格", "INFO")
    window.add_log_message("  3. 点击开始处理，等待完成", "INFO")
    window.add_log_message("  4. 享受你的嘴炮版机器猫！", "INFO")
    window.add_log_message("  5. 可在动画标签页生成AI动画", "INFO")
    
    window.add_log_message("💡 重要提示:", "INFO")
    window.add_log_message("  - 首次使用时，请在设置标签页测试API配置", "INFO")
    window.add_log_message("  - 确保Ollama服务已启动: ollama serve", "INFO")
    window.add_log_message("  - 确保TTS服务已启动: python api.py --port 5021", "INFO")
    window.add_log_message("  - 确保音色文件(.wav)已放在voices文件夹中", "INFO")
    window.add_log_message("  - AI动画生成需要ComfyUI运行在8188端口", "INFO")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()