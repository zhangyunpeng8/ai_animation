# ai_animation_generator.py
"""
AI动画生成器 - 集成ComfyUI人物一致性和AI驱动动画
"""

import asyncio
import json
import os
import sys
import time
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import numpy as np
import cv2
from enum import Enum

# 导入现有的模块
try:
    from unified_comfyui_client import (
        UnifiedComfyUIClient,
        GenerationConfig,
        CharacterMethod,
        CharacterReference
    )
    from main import DubbingEngine, ProcessingConfig, CharacterProfile, VoiceStyle
except ImportError:
    print("❌ 无法导入必要的模块")
    print("请确保 unified_comfyui_client.py 和 main.py 在同一目录下")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIAnimationGenerator")

class AnimationStyle(Enum):
    """动画风格枚举"""
    ANIME = "anime"              # 动漫风格
    CINEMATIC = "cinematic"      # 电影风格
    CARTOON = "cartoon"          # 卡通风格
    REALISTIC = "realistic"      # 写实风格
    PAINTERLY = "painterly"      # 绘画风格
    PIXEL_ART = "pixel_art"      # 像素风格

class LipSyncMethod(Enum):
    """口型同步方法枚举"""
    WHISPER_PHONEMES = "whisper_phonemes"    # Whisper音素分析
    VISEME_BASED = "viseme_based"           # 视位素映射
    DEEP_SPEECH = "deep_speech"             # 深度语音分析
    S2P = "s2p"                             # Speech2Phoneme

class ExpressionType(Enum):
    """表情类型枚举"""
    NEUTRAL = "neutral"          # 中性
    HAPPY = "happy"              # 开心
    SAD = "sad"                  # 悲伤
    ANGRY = "angry"              # 愤怒
    SURPRISED = "surprised"      # 惊讶
    FEARFUL = "fearful"          # 害怕
    DISGUSTED = "disgusted"      # 厌恶
    EXCITED = "excited"          # 兴奋
    THINKING = "thinking"        # 思考
    SPEAKING = "speaking"        # 说话

@dataclass
class AnimationConfig:
    """动画配置"""
    # 基础配置
    resolution: tuple = (512, 768)           # 分辨率 (宽, 高)
    fps: int = 24                            # 帧率
    duration: float = 10.0                   # 时长(秒)
    style: AnimationStyle = AnimationStyle.ANIME
    background: str = ""                     # 背景图片/描述
    seed: int = -1                           # 随机种子
    
    # 角色配置
    character_reference: str = ""            # 角色参考图片
    consistency_method: CharacterMethod = CharacterMethod.IP_ADAPTER
    consistency_strength: float = 0.7        # 一致性强度
    character_scale: float = 0.8             # 角色缩放
    
    # 动画配置
    lip_sync_enabled: bool = True
    lip_sync_method: LipSyncMethod = LipSyncMethod.WHISPER_PHONEMES
    expression_enabled: bool = True
    head_movement_enabled: bool = True
    eye_movement_enabled: bool = True
    body_movement_enabled: bool = True
    
    # 场景配置
    scene_description: str = ""              # 场景描述
    camera_movement: str = "subtle"          # 摄像机运动
    lighting: str = "natural"                # 光照
    
    # 输出配置
    output_format: str = "mp4"
    output_quality: str = "high"

@dataclass
class AnimationSegment:
    """动画片段"""
    id: int
    start_time: float                        # 开始时间(秒)
    end_time: float                          # 结束时间(秒)
    text: str                                # 台词文本
    character: str = "main_character"        # 说话角色
    expression: ExpressionType = ExpressionType.SPEAKING
    audio_data: Optional[np.ndarray] = None  # 音频数据
    lip_sync_data: Optional[Dict] = None     # 口型同步数据
    prompt: str = ""                         # 动画提示词
    
    def __post_init__(self):
        """后初始化"""
        self.duration = self.end_time - self.start_time

@dataclass
class CharacterModel:
    """角色模型"""
    name: str
    reference_images: List[str] = field(default_factory=list)
    voice_profile: Optional[CharacterProfile] = None
    animation_config: Dict[str, Any] = field(default_factory=dict)
    comfyui_workflow: Optional[Dict] = None
    
    def add_reference_image(self, image_path: str):
        """添加参考图片"""
        if os.path.exists(image_path):
            self.reference_images.append(image_path)
            logger.info(f"为角色 {self.name} 添加参考图片: {image_path}")
        else:
            logger.warning(f"参考图片不存在: {image_path}")
    
    def get_comfyui_reference(self) -> CharacterReference:
        """获取ComfyUI参考对象"""
        if self.reference_images:
            return CharacterReference(
                image_path=self.reference_images[0],
                name=self.name,
                face_weight=0.7,
                style_weight=0.5,
                identity_strength=0.8
            )
        else:
            raise ValueError(f"角色 {self.name} 没有参考图片")

class AIAnimationGenerator:
    """AI动画生成器 - 核心类"""
    
    def __init__(self, config: AnimationConfig, comfyui_host: str = "127.0.0.1", comfyui_port: int = 8188):
        self.config = config
        self.comfyui_host = comfyui_host
        self.comfyui_port = comfyui_port
        
        # 初始化客户端
        self.comfyui_client = UnifiedComfyUIClient(
            host=comfyui_host,
            port=comfyui_port
        )
        
        # 初始化配音引擎
        self.dubbing_engine = None
        
        # 存储数据
        self.characters: Dict[str, CharacterModel] = {}
        self.segments: List[AnimationSegment] = []
        self.audio_segments: List[Any] = []
        self.generated_frames: List[str] = []
        
        # 输出目录
        self.output_dir = Path("ai_animation_output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("AI动画生成器初始化完成")
    
    async def initialize(self):
        """初始化连接"""
        logger.info("正在初始化AI动画生成器...")
        
        # 连接到ComfyUI
        logger.info(f"正在连接到ComfyUI: {self.comfyui_host}:{self.comfyui_port}")
        connected = await self.comfyui_client.connect()
        
        if not connected:
            logger.error("无法连接到ComfyUI服务器")
            raise ConnectionError("ComfyUI服务器连接失败")
        
        # 初始化配音引擎
        dubbing_config = ProcessingConfig()
        self.dubbing_engine = DubbingEngine(dubbing_config)
        
        # 检查关键节点是否可用
        try:
            available_nodes = await self.comfyui_client.check_nodes_available()
            for node_name, is_available in available_nodes.items():
                if not is_available:
                    logger.warning(f"⚠️ 关键节点不可用: {node_name}")
        except:
            logger.warning("无法检查节点可用性，继续处理...")
        
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
                "style": self.config.style.value,
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
            
            # 解析剧本（简单实现，实际需要更复杂的解析）
            lines = script_content.strip().split('\n')
            segments = []
            
            current_time = 0.0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    # 简单分割，实际应该根据时间戳解析
                    segment_duration = 3.0  # 默认3秒
                    
                    # 提取角色和台词
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        character_name, dialogue = parts
                        character_name = character_name.strip()
                        dialogue = dialogue.strip()
                    else:
                        character_name = "未知角色"
                        dialogue = line.strip()
                    
                    # 确定角色
                    if character_name in self.characters:
                        character = character_name
                    else:
                        # 使用第一个角色或创建默认角色
                        character = list(self.characters.keys())[0] if self.characters else "main_character"
                    
                    # 确定表情
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
        
        # 关键词检测
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
        
        # 基础提示词
        base_prompt = f"{self.config.style.value} style, "
        
        # 角色描述
        if character_model and character_model.voice_profile:
            char_desc = character_model.voice_profile.name
        else:
            char_desc = character
        
        # 表情描述
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
        
        # 场景描述
        scene_desc = self.config.scene_description if self.config.scene_description else "clean background, cinematic lighting"
        
        # 完整的提示词
        prompt = f"{base_prompt}{char_desc}, {expression_map[expression]}, {scene_desc}, "
        prompt += f"full body shot, dynamic pose, {self.config.lighting} lighting, "
        prompt += f"high quality, detailed, 4k, masterpiece"
        
        # 添加对话内容
        prompt += f", saying: \"{dialogue[:50]}\""
        
        return prompt
    
    async def generate_dubbing(self, script_path: str, output_audio_path: str):
        """生成配音"""
        logger.info("开始生成配音...")
        
        if not self.dubbing_engine:
            logger.error("配音引擎未初始化")
            return None
        
        try:
            # 创建临时视频用于配音处理（实际应该使用剧本）
            temp_video_path = self.output_dir / "temp_scene.mp4"
            
            # 生成简单的测试视频（这里只是示例，实际需要更复杂的实现）
            self._create_test_video(str(temp_video_path))
            
            # 使用配音引擎处理
            self.dubbing_engine.process_episode_optimized(str(temp_video_path))
            
            # 这里应该返回生成的音频路径
            # 实际实现中，应该从dubbing_engine获取生成的音频
            
            logger.info("✅ 配音生成完成")
            return output_audio_path
            
        except Exception as e:
            logger.error(f"配音生成失败: {e}")
            return None
    
    def _create_test_video(self, output_path: str):
        """创建测试视频（示例）"""
        # 创建一个简单的测试视频
        width, height = self.config.resolution
        fps = self.config.fps
        duration = 5  # 5秒
        
        # 使用OpenCV创建视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i in range(fps * duration):
            # 创建渐变背景
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 添加文本
            text = f"AI Animation Test Frame {i+1}"
            cv2.putText(frame, text, (50, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        logger.info(f"创建测试视频: {output_path}")
    
    async def analyze_lip_sync(self, audio_path: str, segments: List[AnimationSegment]) -> List[Dict]:
        """分析口型同步数据"""
        logger.info("分析口型同步数据...")
        
        if not self.config.lip_sync_enabled:
            logger.info("口型同步已禁用")
            return []
        
        lip_sync_data = []
        
        for segment in segments:
            try:
                # 这里应该使用实际的音素分析库
                # 示例实现，实际应该使用whisper或类似工具
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
        """提取音素（示例实现）"""
        # 简化的音素映射
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
                    "duration": 0.1  # 示例时长
                })
        
        return phonemes
    
    def _generate_viseme_frames(self, segment: AnimationSegment) -> List[Dict]:
        """生成视位素帧（示例实现）"""
        frames = []
        fps = self.config.fps
        duration = segment.duration
        total_frames = int(fps * duration)
        
        for frame_idx in range(total_frames):
            frame_time = segment.start_time + (frame_idx / fps)
            
            # 简化的视位素映射
            viseme = "rest"  # 默认闭合
            
            # 根据时间模拟口型变化
            if segment.expression == ExpressionType.SPEAKING:
                # 说话时口型变化
                time_in_segment = frame_time - segment.start_time
                cycle = (time_in_segment * 5) % 1.0  # 5Hz口型变化
                
                if cycle < 0.3:
                    viseme = "AA"  # 张开
                elif cycle < 0.6:
                    viseme = "IH"  # 半张开
                else:
                    viseme = "MM"  # 闭合
            
            frames.append({
                "frame": frame_idx,
                "time": frame_time,
                "viseme": viseme,
                "mouth_openness": 0.5 if viseme == "AA" else 0.2
            })
        
        return frames
    
    async def generate_animation_frames(self, segments: List[AnimationSegment]) -> List[str]:
        """生成动画帧"""
        logger.info("开始生成动画帧...")
        
        generated_frames = []
        
        for segment in segments:
            logger.info(f"生成动画片段 {segment.id}: {segment.text[:30]}...")
            
            try:
                # 获取角色信息
                character_model = self.characters.get(segment.character)
                if not character_model:
                    logger.warning(f"角色 {segment.character} 未找到，使用默认设置")
                    character_model = next(iter(self.characters.values())) if self.characters else None
                
                # 使用ComfyUI生成帧
                if character_model and character_model.reference_images:
                    # 有参考图片，使用人物一致性
                    reference_image = character_model.reference_images[0]
                    
                    # 上传参考图片到ComfyUI
                    uploaded_name = await self.comfyui_client.upload_image(
                        Path(reference_image)
                    )
                    
                    # 生成提示词（添加表情和动作）
                    enhanced_prompt = self._enhance_prompt_with_animation(
                        segment.prompt, segment, character_model
                    )
                    
                    # 使用IP-Adapter生成图像
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
                    # 无参考图片，使用普通生成
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
            
            # 避免请求过快
            await asyncio.sleep(1)
        
        self.generated_frames = generated_frames
        logger.info(f"✅ 动画帧生成完成: {len(generated_frames)}个帧")
        return generated_frames
    
    def _enhance_prompt_with_animation(self, base_prompt: str, segment: AnimationSegment, character_model: CharacterModel) -> str:
        """增强提示词，添加动画元素"""
        enhanced = base_prompt
        
        # 添加口型信息
        if self.config.lip_sync_enabled and segment.lip_sync_data:
            # 获取当前帧的口型状态
            if segment.lip_sync_data.get("viseme_frames"):
                first_frame = segment.lip_sync_data["viseme_frames"][0]
                mouth_state = "open mouth" if first_frame.get("mouth_openness", 0) > 0.4 else "closed mouth"
                enhanced += f", {mouth_state}"
        
        # 添加表情
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
        
        # 添加头部动作
        if self.config.head_movement_enabled:
            # 简单的头部动作序列
            head_motions = ["subtle head turn", "slight nod", "head tilt", "looking forward"]
            motion = head_motions[segment.id % len(head_motions)]
            enhanced += f", {motion}"
        
        # 添加眼部动作
        if self.config.eye_movement_enabled:
            eye_actions = ["looking at viewer", "eye contact", "blinking", "focused gaze"]
            action = eye_actions[segment.id % len(eye_actions)]
            enhanced += f", {action}"
        
        # 添加肢体语言
        if self.config.body_movement_enabled and segment.id % 3 == 0:
            body_poses = ["hand gesture", "leaning forward", "relaxed posture", "dynamic pose"]
            pose = body_poses[segment.id % len(body_poses)]
            enhanced += f", {pose}"
        
        # 添加摄像机角度
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
            # 创建视频输出路径
            output_video_path = self.output_dir / f"animation_{int(time.time())}.{self.config.output_format}"
            
            # 如果有帧数据，创建视频
            if frames:
                # 按时间排序帧
                sorted_frames = sorted(frames, key=lambda x: x.get("time", 0))
                
                # 创建视频编写器
                width, height = self.config.resolution
                fps = self.config.fps
                
                # 创建帧序列视频
                frame_video_path = self.output_dir / "frame_sequence.mp4"
                self._create_frame_sequence(sorted_frames, str(frame_video_path), fps, (width, height))
                
                # 如果有音频，合并音频
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
        
        # 使用OpenCV创建视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 示例：使用占位图像
        # 实际应该加载生成的帧
        for frame_data in frames:
            # 创建示例帧（实际应该加载生成的图像）
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 添加帧信息
            segment_id = frame_data.get("segment_id", 0)
            text = f"Frame {segment_id}"
            cv2.putText(frame, text, (50, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 根据时间戳重复帧以达到正确时长
            # 这里简化处理，每个帧重复fps次（1秒）
            for _ in range(fps):
                out.write(frame)
        
        out.release()
        logger.info(f"创建帧序列视频: {output_path}")
    
    def _merge_audio_with_video(self, video_path: str, audio_path: str, output_path: str):
        """合并音频和视频"""
        try:
            import subprocess
            
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
            # 如果失败，直接复制视频
            import shutil
            shutil.copy(video_path, output_path)
    
    async def generate_complete_animation(self, script_path: str) -> str:
        """生成完整动画"""
        logger.info("开始生成完整动画...")
        
        try:
            # 1. 初始化
            await self.initialize()
            
            # 2. 处理剧本
            segments = await self.process_script(script_path)
            
            # 3. 生成配音
            audio_path = self.output_dir / "dubbed_audio.wav"
            await self.generate_dubbing(script_path, str(audio_path))
            
            # 4. 分析口型同步
            if self.config.lip_sync_enabled:
                await self.analyze_lip_sync(str(audio_path), segments)
            
            # 5. 生成动画帧
            frames = await self.generate_animation_frames(segments)
            
            # 6. 组装动画
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

# ==================== 使用示例 ====================

async def example_usage():
    """使用示例"""
    
    # 1. 创建动画配置
    animation_config = AnimationConfig(
        resolution=(512, 768),
        fps=24,
        duration=30.0,
        style=AnimationStyle.ANIME,
        consistency_method=CharacterMethod.IP_ADAPTER,
        consistency_strength=0.7,
        lip_sync_enabled=True,
        expression_enabled=True,
        scene_description="a beautiful garden with cherry blossoms",
        camera_movement="subtle pan",
        lighting="soft morning light"
    )
    
    # 2. 创建动画生成器
    generator = AIAnimationGenerator(animation_config)
    
    # 3. 创建角色
    # 从配音工厂的角色配置创建动画角色
    voice_profile = CharacterProfile.get_preset("哆啦A梦")
    voice_profile.voice_style = VoiceStyle.LI_YUNLONG
    
    character = generator.create_character_from_profile(
        voice_profile,
        reference_image="characters/doraemon_reference.jpg"  # 替换为实际图片路径
    )
    
    generator.add_character(character)
    
    # 4. 创建剧本
    script_content = """
哆啦A梦: 大雄，你又考试不及格了！
大雄: 我知道错了，哆啦A梦...
哆啦A梦: 拿出点男子汉的气概来！
大雄: 可是我真的不擅长学习...
哆啦A梦: 那就让我来帮你吧！
"""
    
    script_path = "animation_script.txt"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 5. 生成动画
    final_video = await generator.generate_complete_animation(script_path)
    
    if final_video:
        print(f"\n🎬 动画已生成: {final_video}")
        print("🎉 恭喜！AI驱动动画制作完成！")
    else:
        print("❌ 动画生成失败")

# ==================== 与main.py集成的接口 ====================

def create_animation_from_dubbing_project(dubbing_engine: DubbingEngine, output_video: str) -> str:
    """
    从配音项目创建动画
    
    Args:
        dubbing_engine: 配音引擎实例
        output_video: 配音完成的视频路径
    
    Returns:
        生成的动画视频路径
    """
    # 这个函数可以在main.py中调用，将配音视频转换为AI动画
    
    # 1. 提取音频和字幕
    # 2. 分析口型同步
    # 3. 生成角色动画
    # 4. 合成最终视频
    
    # 返回生成的动画路径
    return ""

# ==================== OpenAI接口集成 ====================

class OpenAIIntegration:
    """OpenAI接口集成"""
    
    def __init__(self, api_key: str, base_url: str = "https://apis.iflow.cn/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = "TBStars2-200B-A13B"
    
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
            
            # 解析JSON响应
            import json
            try:
                analysis = json.loads(response)
                return analysis
            except:
                # 如果响应不是JSON，返回文本
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

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 检查ComfyUI服务
    try:
        response = requests.get("http://127.0.0.1:8188", timeout=5)
        if response.status_code == 200:
            print("✅ ComfyUI服务器正在运行")
            
            # 运行示例
            asyncio.run(example_usage())
        else:
            print(f"⚠️ ComfyUI服务器响应: {response.status_code}")
            print("请先启动ComfyUI: python main.py --port 8188")
            
    except requests.exceptions.ConnectionError:
        print("❌ ComfyUI服务器未运行")
        print("请先启动ComfyUI: python main.py --port 8188")