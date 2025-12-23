# voice_manager.py - 简化的音色管理器
import os
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VoiceManagerDialog:
    """音色管理器对话框 - 简化版"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.selected_voice = ""
        
    def exec(self):
        """显示对话框并获取选择的音色文件"""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QListWidget, 
            QListWidgetItem, QPushButton, QLabel, QFileDialog,
            QMessageBox
        )
        from PySide6.QtCore import Qt
        
        class SimpleVoiceDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.selected_voice = ""
                self.init_ui()
            
            def init_ui(self):
                self.setWindowTitle("音色管理器")
                self.resize(500, 400)
                
                layout = QVBoxLayout()
                
                # 说明标签
                info_label = QLabel("选择音色文件 (.wav格式)")
                layout.addWidget(info_label)
                
                # 文件列表
                self.file_list = QListWidget()
                layout.addWidget(self.file_list)
                
                # 扫描voices目录
                self.scan_voices()
                
                # 按钮布局
                button_layout = QHBoxLayout()
                
                browse_btn = QPushButton("浏览文件...")
                browse_btn.clicked.connect(self.browse_file)
                button_layout.addWidget(browse_btn)
                
                refresh_btn = QPushButton("刷新")
                refresh_btn.clicked.connect(self.scan_voices)
                button_layout.addWidget(refresh_btn)
                
                button_layout.addStretch()
                
                ok_btn = QPushButton("确定")
                ok_btn.clicked.connect(self.accept)
                button_layout.addWidget(ok_btn)
                
                cancel_btn = QPushButton("取消")
                cancel_btn.clicked.connect(self.reject)
                button_layout.addWidget(cancel_btn)
                
                layout.addLayout(button_layout)
                self.setLayout(layout)
            
            def scan_voices(self):
                """扫描voices目录"""
                self.file_list.clear()
                voices_dir = Path("voices")
                
                if voices_dir.exists():
                    for file in voices_dir.glob("*.wav"):
                        item = QListWidgetItem(str(file))
                        self.file_list.addItem(item)
                    
                    for file in voices_dir.glob("*.mp3"):
                        item = QListWidgetItem(str(file))
                        self.file_list.addItem(item)
                
                # 如果没有文件，显示提示
                if self.file_list.count() == 0:
                    item = QListWidgetItem("voices目录中没有音色文件")
                    item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                    self.file_list.addItem(item)
            
            def browse_file(self):
                """浏览文件"""
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "选择音色文件", "", "音频文件 (*.wav *.mp3);;所有文件 (*.*)"
                )
                
                if file_path:
                    # 复制到voices目录
                    try:
                        voices_dir = Path("voices")
                        voices_dir.mkdir(exist_ok=True)
                        
                        dest_path = voices_dir / Path(file_path).name
                        
                        # 如果是wav文件，直接复制
                        if file_path.lower().endswith('.wav'):
                            import shutil
                            shutil.copy2(file_path, dest_path)
                        else:
                            # 如果是其他格式，转换为wav
                            import soundfile as sf
                            audio, sr = sf.read(file_path)
                            sf.write(dest_path, audio, sr)
                        
                        self.scan_voices()
                        QMessageBox.information(self, "成功", f"音色文件已添加到: {dest_path}")
                        
                    except Exception as e:
                        QMessageBox.warning(self, "错误", f"添加音色文件失败: {e}")
            
            def accept(self):
                """确认选择"""
                selected_items = self.file_list.selectedItems()
                if selected_items:
                    self.selected_voice = selected_items[0].text()
                    if self.selected_voice == "voices目录中没有音色文件":
                        self.selected_voice = ""
                super().accept()
        
        dialog = SimpleVoiceDialog(self.parent)
        result = dialog.exec()
        self.selected_voice = dialog.selected_voice
        return result
    
    def get_selected_voice(self) -> str:
        """获取选择的音色文件路径"""
        return self.selected_voice


class VoiceLibrary:
    """音色库管理"""
    
    def __init__(self, library_path: str = "voice_library.json"):
        self.library_path = Path(library_path)
        self.voices: Dict[str, Dict] = {}
        self.load_library()
    
    def load_library(self):
        """加载音色库"""
        if self.library_path.exists():
            try:
                with open(self.library_path, 'r', encoding='utf-8') as f:
                    self.voices = json.load(f)
                logger.info(f"音色库加载成功，共 {len(self.voices)} 个音色")
            except Exception as e:
                logger.error(f"加载音色库失败: {e}")
                self.voices = {}
        else:
            logger.info("音色库不存在，创建新的音色库")
            self.voices = {}
    
    def save_library(self):
        """保存音色库"""
        try:
            with open(self.library_path, 'w', encoding='utf-8') as f:
                json.dump(self.voices, f, ensure_ascii=False, indent=2)
            logger.info("音色库保存成功")
        except Exception as e:
            logger.error(f"保存音色库失败: {e}")
    
    def add_voice(self, voice_id: str, name: str, file_path: str, 
                  description: str = "", tags: List[str] = None):
        """添加音色"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"音色文件不存在: {file_path}")
                return False
            
            # 获取音频信息
            audio, sr = sf.read(file_path)
            duration = len(audio) / sr
            
            self.voices[voice_id] = {
                "name": name,
                "file_path": file_path,
                "description": description,
                "tags": tags or [],
                "sample_rate": sr,
                "duration": duration,
                "channels": audio.shape[1] if len(audio.shape) > 1 else 1,
                "added_time": datetime.now().isoformat()
            }
            
            self.save_library()
            logger.info(f"音色添加成功: {name}")
            return True
            
        except Exception as e:
            logger.error(f"添加音色失败: {e}")
            return False
    
    def remove_voice(self, voice_id: str):
        """删除音色"""
        if voice_id in self.voices:
            voice_name = self.voices[voice_id]["name"]
            del self.voices[voice_id]
            self.save_library()
            logger.info(f"音色删除成功: {voice_name}")
            return True
        return False
    
    def get_voice(self, voice_id: str) -> Optional[Dict]:
        """获取音色信息"""
        return self.voices.get(voice_id)
    
    def search_voices(self, keyword: str = "", tags: List[str] = None) -> List[Dict]:
        """搜索音色"""
        results = []
        
        for voice_id, voice_info in self.voices.items():
            # 关键词搜索
            if keyword:
                keyword_lower = keyword.lower()
                name_match = keyword_lower in voice_info["name"].lower()
                desc_match = keyword_lower in voice_info["description"].lower()
                if not (name_match or desc_match):
                    continue
            
            # 标签搜索
            if tags:
                voice_tags = set(voice_info.get("tags", []))
                search_tags = set(tags)
                if not search_tags.issubset(voice_tags):
                    continue
            
            results.append({"id": voice_id, **voice_info})
        
        return results
    
    def get_audio_data(self, voice_id: str) -> Optional[Tuple[np.ndarray, int]]:
        """获取音频数据"""
        voice_info = self.get_voice(voice_id)
        if not voice_info:
            return None
        
        try:
            audio, sr = sf.read(voice_info["file_path"])
            return audio, sr
        except Exception as e:
            logger.error(f"读取音色文件失败: {e}")
            return None


def test_voice_file(file_path: str) -> bool:
    """测试音色文件"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False
        
        audio, sr = sf.read(file_path)
        
        # 检查音频参数
        duration = len(audio) / sr
        
        print(f"音色文件测试结果:")
        print(f"  文件路径: {file_path}")
        print(f"  采样率: {sr} Hz")
        print(f"  时长: {duration:.2f} 秒")
        print(f"  声道数: {audio.shape[1] if len(audio.shape) > 1 else 1}")
        print(f"  数据范围: [{audio.min():.4f}, {audio.max():.4f}]")
        
        # 检查静音部分
        rms = np.sqrt(np.mean(audio**2))
        print(f"  音量水平: {rms:.4f}")
        
        if rms < 0.01:
            print("⚠️  警告: 音量过低，可能是静音文件")
        
        return True
        
    except Exception as e:
        logger.error(f"测试音色文件失败: {e}")
        return False


def convert_to_wav(input_path: str, output_path: str = None, 
                   target_sr: int = 24000, normalize: bool = True) -> Optional[str]:
    """转换音频文件为WAV格式"""
    try:
        import soundfile as sf
        import librosa
        
        # 如果没有指定输出路径，自动生成
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(input_file.with_suffix('.wav'))
        
        # 加载音频
        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
        
        # 归一化
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
        
        # 保存为WAV
        sf.write(output_path, audio, target_sr)
        
        logger.info(f"音频转换成功: {input_path} -> {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"音频转换失败: {e}")
        return None


if __name__ == "__main__":
    # 简单的命令行测试
    import sys
    
    if len(sys.argv) > 1:
        test_voice_file(sys.argv[1])
    else:
        print("使用方式: python voice_manager.py <音色文件路径>")