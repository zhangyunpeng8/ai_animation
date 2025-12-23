# unified_comfyui_client.py
"""
完整的统一 ComfyUI 客户端
集成了通用功能和人物一致性功能
"""

import json
import requests
import uuid
import time
import asyncio
import aiohttp
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedComfyUI")


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CharacterMethod(Enum):
    """人物一致性方法"""
    IP_ADAPTER = "ip_adapter"
    INSTANT_ID = "instant_id"
    PHOTO_MAKER = "photo_maker"
    FACE_DETAILER = "face_detailer"
    LORA = "lora"
    CONTROLNET = "controlnet"


@dataclass
class CharacterReference:
    """人物参考"""
    image_path: str
    name: str = "character"
    face_weight: float = 0.7
    style_weight: float = 0.5
    identity_strength: float = 0.8


@dataclass
class GenerationConfig:
    """生成配置"""
    width: int = 512
    height: int = 768
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "euler"
    scheduler: str = "normal"
    seed: int = -1
    batch_size: int = 1
    model: str = "sd15.safetensors"
    vae: str = "auto"


class UnifiedComfyUIClient:
    """完整的统一 ComfyUI 客户端"""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        use_ssl: bool = False,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.host = host
        self.port = port
        self.protocol = "https" if use_ssl else "http"
        self.base_url = f"{self.protocol}://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.timeout = timeout
        self.api_key = api_key
        self.client_id = str(uuid.uuid4())
        self.session = None
        self.ws = None
        
        # 缓存
        self.uploaded_files = {}
        self.character_templates = {}
        
        # 配置默认工作流
        self.default_config = GenerationConfig()
        
        logger.info(f"Unified ComfyUI Client initialized: {self.base_url}")
    
    async def connect(self):
        """连接服务器"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        # 测试连接
        try:
            response = await self._request("GET", "/system_stats")
            logger.info(f"Connected to ComfyUI v{response.get('system', {}).get('comfyui_version', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()
        if self.ws:
            self.ws.close()
        logger.info("Disconnected from ComfyUI")
    
    async def _request(self, method: str, endpoint: str, data: Optional[Dict] = None, files: Optional[Dict] = None) -> Dict:
        """发送 HTTP 请求"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    return await response.json()
            elif method.upper() == "POST":
                if files:
                    # 文件上传
                    form_data = aiohttp.FormData()
                    for key, value in data.items():
                        form_data.add_field(key, str(value))
                    for key, (filename, fileobj, content_type) in files.items():
                        form_data.add_field(key, fileobj, filename=filename, content_type=content_type)
                    
                    async with self.session.post(url, data=form_data) as response:
                        return await response.json()
                else:
                    async with self.session.post(url, json=data) as response:
                        return await response.json()
            else:
                raise Exception(f"Unsupported method: {method}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    # ==================== 通用 API 方法 ====================
    
    async def get_system_info(self) -> Dict:
        """获取系统信息"""
        return await self._request("GET", "/system_stats")
    
    async def get_all_nodes(self) -> Dict:
        """获取所有节点"""
        return await self._request("GET", "/object_info")
    
    async def get_node_info(self, node_class: str) -> Dict:
        """获取节点信息"""
        return await self._request("GET", f"/object_info/{node_class}")
    
    async def upload_image(self, image_path: Path, image_type: str = "input") -> str:
        """上传图片"""
        try:
            with open(image_path, 'rb') as f:
                files = {
                    'image': (image_path.name, f, 'image/png')
                }
                data = {
                    'type': image_type,
                    'overwrite': 'false'
                }
                
                result = await self._request("POST", "/upload/image", data, files)
                
                filename = result.get('name')
                if filename:
                    self.uploaded_files[filename] = {
                        'path': str(image_path),
                        'type': image_type,
                        'time': time.time()
                    }
                    return filename
                else:
                    raise Exception("Upload failed: no filename returned")
                    
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            raise
    
    async def get_queue(self) -> Dict:
        """获取队列状态"""
        return await self._request("GET", "/queue")
    
    async def clear_queue(self) -> bool:
        """清空队列"""
        try:
            await self._request("POST", "/queue", {"clear": True})
            return True
        except:
            return False
    
    async def interrupt(self, prompt_id: Optional[str] = None) -> bool:
        """中断任务"""
        data = {}
        if prompt_id:
            data['prompt_id'] = prompt_id
        
        try:
            await self._request("POST", "/interrupt", data)
            return True
        except:
            return False
    
    # ==================== 工作流执行 ====================
    
    async def execute_workflow(self, workflow: Dict, extra_data: Optional[Dict] = None) -> Dict:
        """执行工作流"""
        data = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        if extra_data:
            data["extra_data"] = extra_data
        
        try:
            response = await self._request("POST", "/prompt", data)
            
            if "error" in response:
                return {
                    "success": False,
                    "error": response["error"],
                    "prompt_id": response.get("prompt_id")
                }
            
            prompt_id = response.get("prompt_id")
            if not prompt_id:
                return {
                    "success": False,
                    "error": "No prompt_id returned"
                }
            
            # 等待完成
            return await self.wait_for_completion(prompt_id)
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def wait_for_completion(self, prompt_id: str, timeout: int = 300, poll_interval: int = 2) -> Dict:
        """等待任务完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                history = await self._request("GET", f"/history/{prompt_id}")
                
                if prompt_id in history:
                    task_data = history[prompt_id]
                    
                    if task_data.get('status', {}).get('completed', False):
                        # 提取输出
                        outputs = task_data.get('outputs', {})
                        images = []
                        videos = []
                        
                        for node_id, node_output in outputs.items():
                            if 'images' in node_output:
                                for img in node_output['images']:
                                    images.append({
                                        'filename': img.get('filename'),
                                        'subfolder': img.get('subfolder', ''),
                                        'type': img.get('type', 'output')
                                    })
                            if 'videos' in node_output:
                                videos.extend(node_output['videos'])
                        
                        return {
                            "success": True,
                            "prompt_id": prompt_id,
                            "status": "completed",
                            "images": images,
                            "videos": videos,
                            "metadata": task_data
                        }
                    elif task_data.get('status', {}).get('status_str') == 'error':
                        return {
                            "success": False,
                            "prompt_id": prompt_id,
                            "status": "error",
                            "error": task_data.get('status', {}).get('message', 'Unknown error')
                        }
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error while waiting: {e}")
                await asyncio.sleep(poll_interval)
        
        return {
            "success": False,
            "prompt_id": prompt_id,
            "status": "timeout",
            "error": f"Timeout after {timeout} seconds"
        }
    
    # ==================== 通用工作流生成 ====================
    
    def create_basic_image_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        config: Optional[GenerationConfig] = None
    ) -> Dict:
        """创建基础图像生成工作流"""
        cfg = config or self.default_config
        
        if cfg.seed == -1:
            import random
            cfg.seed = random.randint(0, 2**32 - 1)
        
        return {
            "checkpoint": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": cfg.model}
            },
            "positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["checkpoint", 1],
                    "text": prompt
                }
            },
            "negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["checkpoint", 1],
                    "text": negative_prompt
                }
            },
            "empty_latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": cfg.width,
                    "height": cfg.height,
                    "batch_size": cfg.batch_size
                }
            },
            "ksampler": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["checkpoint", 0],
                    "positive": ["positive", 0],
                    "negative": ["negative", 0],
                    "latent_image": ["empty_latent", 0],
                    "seed": cfg.seed,
                    "steps": cfg.steps,
                    "cfg": cfg.cfg,
                    "sampler_name": cfg.sampler,
                    "scheduler": cfg.scheduler,
                    "denoise": 1.0
                }
            },
            "vae_decode": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["ksampler", 0],
                    "vae": ["checkpoint", 2]
                }
            },
            "save_image": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["vae_decode", 0],
                    "filename_prefix": f"generated_{int(time.time())}"
                }
            }
        }
    
    # ==================== 人物一致性工作流 ====================
    
    def create_ipadapter_workflow(
        self,
        reference_image: str,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.7,
        config: Optional[GenerationConfig] = None,
        ipadapter_model: str = "ip-adapter-plus-face_sd15.bin"
    ) -> Dict:
        """创建 IP-Adapter 工作流"""
        cfg = config or self.default_config
        
        if cfg.seed == -1:
            import random
            cfg.seed = random.randint(0, 2**32 - 1)
        
        workflow = {
            "load_image": {
                "class_type": "LoadImage",
                "inputs": {"image": reference_image}
            },
            "checkpoint": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": cfg.model}
            },
            "clip_vision": {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": ipadapter_model}
            },
            "ipadapter_encode": {
                "class_type": "IPAdapterEncoder",
                "inputs": {
                    "clip_vision": ["clip_vision", 0],
                    "image": ["load_image", 0],
                    "weight": strength,
                    "weight_type": "original",
                    "start_at": 0.0,
                    "end_at": 1.0,
                    "noise": 0.0
                }
            },
            "positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["checkpoint", 1],
                    "text": prompt
                }
            },
            "negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["checkpoint", 1],
                    "text": negative_prompt
                }
            },
            "empty_latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": cfg.width,
                    "height": cfg.height,
                    "batch_size": cfg.batch_size
                }
            },
            "ksampler": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["checkpoint", 0],
                    "positive": ["positive", 0],
                    "negative": ["negative", 0],
                    "latent_image": ["empty_latent", 0],
                    "seed": cfg.seed,
                    "steps": cfg.steps,
                    "cfg": cfg.cfg,
                    "sampler_name": cfg.sampler,
                    "scheduler": cfg.scheduler,
                    "denoise": 1.0
                }
            },
            "vae_decode": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["ksampler", 0],
                    "vae": ["checkpoint", 2]
                }
            },
            "save_image": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["vae_decode", 0],
                    "filename_prefix": f"ipadapter_{int(time.time())}"
                }
            }
        }
        
        # 添加 IP-Adapter 到 KSampler 的条件输入
        workflow["ksampler"]["inputs"]["positive"] = ["ipadapter_encode", 0]
        
        return workflow
    
    def create_instantid_workflow(
        self,
        reference_image: str,
        prompt: str,
        negative_prompt: str = "",
        control_strength: float = 0.8,
        identity_strength: float = 0.7,
        config: Optional[GenerationConfig] = None
    ) -> Dict:
        """创建 InstantID 工作流"""
        cfg = config or self.default_config
        
        if cfg.seed == -1:
            import random
            cfg.seed = random.randint(0, 2**32 - 1)
        
        workflow = {
            "load_image": {
                "class_type": "LoadImage",
                "inputs": {"image": reference_image}
            },
            "checkpoint": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": cfg.model}
            },
            "instantid_model": {
                "class_type": "InstantIDModelLoader",
                "inputs": {"instantid_name": "instant-id.bin"}
            },
            "insightface": {
                "class_type": "InsightFaceLoader",
                "inputs": {"provider": "CPU"}
            },
            "apply_instantid": {
                "class_type": "ApplyInstantID",
                "inputs": {
                    "image": ["load_image", 0],
                    "instantid": ["instantid_model", 0],
                    "insightface": ["insightface", 0],
                    "control_strength": control_strength,
                    "identity_strength": identity_strength,
                    "ipa_weight": 1.0,
                    "start_at": 0.0,
                    "end_at": 1.0
                }
            },
            "positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["checkpoint", 1],
                    "text": prompt
                }
            },
            "negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["checkpoint", 1],
                    "text": negative_prompt
                }
            },
            "empty_latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": cfg.width,
                    "height": cfg.height,
                    "batch_size": cfg.batch_size
                }
            },
            "ksampler": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["checkpoint", 0],
                    "positive": ["positive", 0],
                    "negative": ["negative", 0],
                    "latent_image": ["empty_latent", 0],
                    "seed": cfg.seed,
                    "steps": 30,  # InstantID 需要更多步数
                    "cfg": 7.0,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0
                }
            },
            "vae_decode": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["ksampler", 0],
                    "vae": ["checkpoint", 2]
                }
            },
            "save_image": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["vae_decode", 0],
                    "filename_prefix": f"instantid_{int(time.time())}"
                }
            }
        }
        
        # 连接 InstantID 到 KSampler
        workflow["ksampler"]["inputs"]["positive"] = ["apply_instantid", 0]
        
        return workflow
    
    def create_facedetailer_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        face_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        face_model: str = "face_yolov8n.pt"
    ) -> Dict:
        """创建 FaceDetailer 工作流（面部修复）"""
        cfg = config or self.default_config
        
        if cfg.seed == -1:
            import random
            cfg.seed = random.randint(0, 2**32 - 1)
        
        workflow = self.create_basic_image_workflow(prompt, negative_prompt, cfg)
        
        # 移除原有的保存节点
        workflow.pop("save_image", None)
        
        # 添加面部检测和修复
        workflow["face_detector"] = {
            "class_type": "FaceDetectorLoader",
            "inputs": {"face_model": face_model}
        }
        
        workflow["face_detailer"] = {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": ["vae_decode", 0],
                "face_detector": ["face_detector", 0],
                "model": ["checkpoint", 0],
                "clip": ["checkpoint", 1],
                "vae": ["checkpoint", 2],
                "face_prompt": face_prompt or prompt,
                "face_negative": "blurry, deformed face, bad anatomy",
                "face_strength": 0.6,
                "face_steps": 15,
                "face_cfg": 7.0,
                "face_sampler": "euler",
                "face_scheduler": "normal"
            }
        }
        
        workflow["save_image"] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["face_detailer", 0],
                "filename_prefix": f"facedetailer_{int(time.time())}"
            }
        }
        
        return workflow
    
    def create_multireference_workflow(
        self,
        reference_images: List[str],
        prompt: str,
        negative_prompt: str = "",
        strengths: List[float] = None,
        config: Optional[GenerationConfig] = None
    ) -> Dict:
        """创建多参考图工作流"""
        if not reference_images:
            raise ValueError("At least one reference image is required")
        
        cfg = config or self.default_config
        
        if cfg.seed == -1:
            import random
            cfg.seed = random.randint(0, 2**32 - 1)
        
        if strengths is None:
            strengths = [0.7] * len(reference_images)
        
        workflow = {
            "checkpoint": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": cfg.model}
            },
            "clip_vision": {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": "ip-adapter.bin"}
            }
        }
        
        # 创建多个图像加载和编码节点
        ipadapter_outputs = []
        
        for i, (img, strength) in enumerate(zip(reference_images, strengths)):
            load_node = f"load_image_{i}"
            encode_node = f"ipadapter_{i}"
            
            workflow[load_node] = {
                "class_type": "LoadImage",
                "inputs": {"image": img}
            }
            
            workflow[encode_node] = {
                "class_type": "IPAdapterEncoder",
                "inputs": {
                    "clip_vision": ["clip_vision", 0],
                    "image": [load_node, 0],
                    "weight": strength,
                    "noise": 0.0
                }
            }
            
            ipadapter_outputs.append([encode_node, 0])
        
        # 合并 IP-Adapter 输出
        workflow["combine_ipadapters"] = {
            "class_type": "IPAdapterCombine",
            "inputs": {}
        }
        
        # 动态添加输入
        for i, output in enumerate(ipadapter_outputs):
            workflow["combine_ipadapters"]["inputs"][f"ipadapter_{i+1}"] = output
        
        # 其余节点
        workflow.update({
            "positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["checkpoint", 1],
                    "text": prompt
                }
            },
            "negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["checkpoint", 1],
                    "text": negative_prompt
                }
            },
            "empty_latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": cfg.width,
                    "height": cfg.height,
                    "batch_size": cfg.batch_size
                }
            },
            "ksampler": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["checkpoint", 0],
                    "positive": ["positive", 0],
                    "negative": ["negative", 0],
                    "latent_image": ["empty_latent", 0],
                    "seed": cfg.seed,
                    "steps": cfg.steps,
                    "cfg": cfg.cfg,
                    "sampler_name": cfg.sampler,
                    "scheduler": cfg.scheduler,
                    "denoise": 1.0
                }
            },
            "vae_decode": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["ksampler", 0],
                    "vae": ["checkpoint", 2]
                }
            },
            "save_image": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["vae_decode", 0],
                    "filename_prefix": f"multiref_{int(time.time())}"
                }
            }
        })
        
        # 连接合并的 IP-Adapter 到 KSampler
        workflow["ksampler"]["inputs"]["positive"] = ["combine_ipadapters", 0]
        
        return workflow
    
    # ==================== 高级生成方法 ====================
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        config: Optional[GenerationConfig] = None,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """生成图像"""
        workflow = self.create_basic_image_workflow(prompt, negative_prompt, config)
        result = await self.execute_workflow(workflow)
        
        if output_dir and result.get("success") and result.get("images"):
            await self.download_results(result, output_dir)
        
        return result
    
    async def generate_character(
        self,
        reference_image: Union[str, Path],
        prompt: str,
        method: CharacterMethod = CharacterMethod.IP_ADAPTER,
        negative_prompt: str = "",
        strength: float = 0.7,
        config: Optional[GenerationConfig] = None,
        output_dir: Optional[Path] = None,
        **kwargs
    ) -> Dict:
        """生成一致性人物"""
        # 上传参考图片
        if isinstance(reference_image, str):
            reference_image = Path(reference_image)
        
        if not reference_image.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_image}")
        
        uploaded_name = await self.upload_image(reference_image)
        
        # 根据方法创建工作流
        if method == CharacterMethod.IP_ADAPTER:
            workflow = self.create_ipadapter_workflow(
                reference_image=uploaded_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                config=config,
                ipadapter_model=kwargs.get('ipadapter_model', 'ip-adapter-plus-face_sd15.bin')
            )
        elif method == CharacterMethod.INSTANT_ID:
            workflow = self.create_instantid_workflow(
                reference_image=uploaded_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_strength=kwargs.get('control_strength', 0.8),
                identity_strength=kwargs.get('identity_strength', 0.7),
                config=config
            )
        elif method == CharacterMethod.FACE_DETAILER:
            workflow = self.create_facedetailer_workflow(
                prompt=prompt,
                negative_prompt=negative_prompt,
                face_prompt=kwargs.get('face_prompt'),
                config=config,
                face_model=kwargs.get('face_model', 'face_yolov8n.pt')
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # 执行工作流
        result = await self.execute_workflow(workflow)
        
        # 下载结果
        if output_dir and result.get("success") and result.get("images"):
            await self.download_results(result, output_dir)
        
        return result
    
    async def generate_character_variations(
        self,
        reference_image: Union[str, Path],
        base_prompt: str,
        variations: List[str],
        method: CharacterMethod = CharacterMethod.IP_ADAPTER,
        output_dir: Optional[Path] = None,
        **kwargs
    ) -> Dict:
        """生成人物变体"""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, variation in enumerate(variations):
            logger.info(f"Generating variation {i+1}/{len(variations)}: {variation}")
            
            full_prompt = f"{base_prompt}, {variation}"
            var_output_dir = output_dir / f"var_{i+1}" if output_dir else None
            
            try:
                result = await self.generate_character(
                    reference_image=reference_image,
                    prompt=full_prompt,
                    method=method,
                    output_dir=var_output_dir,
                    **kwargs
                )
                
                results.append({
                    "variation": variation,
                    "prompt": full_prompt,
                    "success": result.get("success", False),
                    "files": result.get("images", []),
                    "error": result.get("error")
                })
                
            except Exception as e:
                results.append({
                    "variation": variation,
                    "prompt": full_prompt,
                    "success": False,
                    "error": str(e)
                })
            
            # 避免请求过快
            await asyncio.sleep(1)
        
        return {
            "total_variations": len(variations),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results
        }
    
    async def generate_multi_character_scene(
        self,
        characters: List[CharacterReference],
        scene_prompt: str,
        config: Optional[GenerationConfig] = None,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """生成多角色场景"""
        # 上传所有参考图片
        uploaded_images = []
        
        for char in characters:
            image_path = Path(char.image_path)
            if image_path.exists():
                filename = await self.upload_image(image_path)
                uploaded_images.append(filename)
            else:
                logger.warning(f"Character image not found: {image_path}")
        
        if not uploaded_images:
            raise ValueError("No valid character images provided")
        
        # 创建工作流
        strengths = [char.face_weight for char in characters]
        workflow = self.create_multireference_workflow(
            reference_images=uploaded_images,
            prompt=scene_prompt,
            strengths=strengths,
            config=config
        )
        
        # 执行工作流
        result = await self.execute_workflow(workflow)
        
        # 下载结果
        if output_dir and result.get("success") and result.get("images"):
            await self.download_results(result, output_dir)
        
        return result
    
    # ==================== 实用方法 ====================
    
    async def download_results(self, result: Dict, output_dir: Path) -> List[str]:
        """下载生成结果"""
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        
        for img_info in result.get("images", []):
            filename = img_info.get("filename")
            subfolder = img_info.get("subfolder", "")
            img_type = img_info.get("type", "output")
            
            if filename:
                # 构建下载URL
                params = {
                    "filename": filename,
                    "type": img_type
                }
                
                if subfolder:
                    params["subfolder"] = subfolder
                
                query_str = "&".join([f"{k}={v}" for k, v in params.items()])
                url = f"{self.base_url}/view?{query_str}"
                
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            # 保存文件
                            output_path = output_dir / filename
                            with open(output_path, 'wb') as f:
                                f.write(await response.read())
                            
                            downloaded_files.append(str(output_path))
                            logger.info(f"Downloaded: {output_path}")
                        else:
                            logger.error(f"Failed to download {filename}: {response.status}")
                except Exception as e:
                    logger.error(f"Error downloading {filename}: {e}")
        
        return downloaded_files
    
    async def get_image_as_base64(self, filename: str, image_type: str = "output") -> str:
        """获取图片的 base64 编码"""
        try:
            url = f"{self.base_url}/view?filename={filename}&type={image_type}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return base64.b64encode(image_data).decode('utf-8')
                else:
                    raise Exception(f"Failed to get image: {response.status}")
        except Exception as e:
            logger.error(f"Error getting image as base64: {e}")
            return ""
    
    async def check_nodes_available(self) -> Dict[str, bool]:
        """检查关键节点是否可用"""
        nodes = await self.get_all_nodes()
        
        key_nodes = {
            "IPAdapterEncoder": "IP-Adapter 编码器",
            "InstantIDModelLoader": "InstantID 模型加载器",
            "FaceDetailer": "面部修复器",
            "IPAdapterCombine": "IP-Adapter 合并器",
            "InsightFaceLoader": "InsightFace 加载器"
        }
        
        available = {}
        for node_id, node_name in key_nodes.items():
            available[node_name] = node_id in nodes
        
        return available


# ==================== 使用示例 ====================

async def comprehensive_example():
    """完整使用示例"""
    
    client = UnifiedComfyUIClient(host="127.0.0.1", port=8188)
    
    try:
        # 连接服务器
        if not await client.connect():
            print("❌ 连接失败")
            return
        
        print("✅ 已连接到 ComfyUI")
        
        # 1. 检查系统信息
        print("\n1. 检查系统信息")
        system_info = await client.get_system_info()
        print(f"   版本: {system_info.get('system', {}).get('comfyui_version', 'unknown')}")
        print(f"   Python: {system_info.get('system', {}).get('python_version', 'unknown')}")
        
        # 2. 检查可用节点
        print("\n2. 检查关键节点")
        available_nodes = await client.check_nodes_available()
        for node_name, is_available in available_nodes.items():
            status = "✅" if is_available else "❌"
            print(f"   {status} {node_name}")
        
        # 3. 普通图像生成
        print("\n3. 普通图像生成测试")
        basic_result = await client.generate_image(
            prompt="a beautiful sunset over mountains, cinematic, 4k",
            negative_prompt="blurry, low quality",
            config=GenerationConfig(width=512, height=512, steps=20)
        )
        
        if basic_result.get("success"):
            print("   ✅ 普通图像生成成功")
            if basic_result.get("images"):
                print(f"   生成图片数: {len(basic_result['images'])}")
        else:
            print(f"   ❌ 普通图像生成失败: {basic_result.get('error')}")
        
        # 4. 检查是否有参考图片用于一致性测试
        test_ref_image = Path("test_reference.jpg")
        
        if test_ref_image.exists():
            print(f"\n4. 人物一致性测试 (使用 {test_ref_image.name})")
            
            # IP-Adapter 测试
            print("   a) IP-Adapter 测试")
            ip_result = await client.generate_character(
                reference_image=test_ref_image,
                prompt="a portrait of a person wearing sunglasses, cool style",
                method=CharacterMethod.IP_ADAPTER,
                strength=0.7,
                output_dir=Path("output/ipadapter_test")
            )
            
            if ip_result.get("success"):
                print("      ✅ IP-Adapter 成功")
            else:
                print(f"      ❌ IP-Adapter 失败: {ip_result.get('error')}")
            
            # 人物变体测试
            print("   b) 人物变体测试")
            variations = [
                "wearing business suit",
                "in medieval armor",
                "with rainbow hair"
            ]
            
            var_result = await client.generate_character_variations(
                reference_image=test_ref_image,
                base_prompt="a portrait of a person",
                variations=variations,
                method=CharacterMethod.IP_ADAPTER,
                output_dir=Path("output/variations_test")
            )
            
            print(f"      生成 {var_result['successful']}/{var_result['total_variations']} 个变体成功")
            
        else:
            print("\n4. 跳过人物一致性测试 (test_reference.jpg 不存在)")
            print("   请创建一个测试图片或使用自己的参考图片")
        
        # 5. 队列管理
        print("\n5. 队列管理")
        queue = await client.get_queue()
        queue_running = len(queue.get('queue_running', []))
        queue_pending = len(queue.get('queue_pending', []))
        
        print(f"   运行中: {queue_running}, 等待中: {queue_pending}")
        
        if queue_pending > 0:
            print("   正在清空队列...")
            if await client.clear_queue():
                print("   ✅ 队列已清空")
        
        print("\n🎉 所有测试完成")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await client.close()
        print("\n🔌 连接已关闭")


# 快速生成函数
async def quick_generate_character(
    reference_image: str,
    prompt: str,
    output_dir: str = "output",
    method: str = "ip_adapter"
) -> List[str]:
    """
    快速生成一致性人物
    
    Args:
        reference_image: 参考图片路径
        prompt: 提示词
        output_dir: 输出目录
        method: 方法 (ip_adapter, instant_id, face_detailer)
    
    Returns:
        生成的图片路径列表
    """
    method_map = {
        "ip_adapter": CharacterMethod.IP_ADAPTER,
        "instant_id": CharacterMethod.INSTANT_ID,
        "face_detailer": CharacterMethod.FACE_DETAILER
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")
    
    client = UnifiedComfyUIClient()
    
    try:
        await client.connect()
        
        result = await client.generate_character(
            reference_image=reference_image,
            prompt=prompt,
            method=method_map[method],
            output_dir=Path(output_dir)
        )
        
        if result.get("success"):
            downloaded = await client.download_results(result, Path(output_dir))
            return downloaded
        else:
            raise Exception(f"Generation failed: {result.get('error')}")
            
    finally:
        await client.close()


if __name__ == "__main__":
    # 运行示例
    import asyncio
    
    # 检查服务器是否运行
    try:
        response = requests.get("http://127.0.0.1:8188", timeout=5)
        
        if response.status_code == 200:
            print("ComfyUI 服务器正在运行\n")
            asyncio.run(comprehensive_example())
        else:
            print(f"服务器响应: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️ ComfyUI 服务器未运行")
        print("请先启动 ComfyUI: python main.py --port 8188")