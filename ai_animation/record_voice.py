#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
录音功能模块
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QMessageBox, QGroupBox, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont


class RecordVoiceDialog(QDialog):
    """录音对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎤 录制音色")
        self.setModal(True)
        self.resize(500, 400)
        
        self.is_recording = False
        self.record_start_time = None
        self.recorded_file = None
        self.record_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("🎤 录制新音色")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #4ec9b0; text-align: center;")
        layout.addWidget(title_label)
        
        # 说明
        info_label = QLabel(
            "录制步骤:\n"
            "1. 点击'开始录制'按钮开始录音\n"
            "2. 说出您想要录制的语音内容\n"
            "3. 点击'停止录制'完成录音\n"
            "4. 可以试听录制的效果"
        )
        info_label.setStyleSheet("color: #888; font-size: 13px; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 录音控制
        control_group = QGroupBox("录音控制")
        control_layout = QVBoxLayout()
        
        # 文件名
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("音色名称:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如: 我的音色_001")
        default_name = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.name_edit.setText(default_name)
        name_layout.addWidget(self.name_edit)
        control_layout.addLayout(name_layout)
        
        # 时间显示
        self.time_label = QLabel("准备录音...")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
                padding: 10px;
                background-color: #2d2d2d;
                border-radius: 5px;
            }
        """)
        control_layout.addWidget(self.time_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 60)  # 最多60秒
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🎤 开始录制")
        self.start_btn.clicked.connect(self.start_recording)
        self.start_btn.setMinimumHeight(40)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ 停止录制")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        control_layout.addLayout(button_layout)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 预览控制
        preview_group = QGroupBox("预览")
        preview_layout = QVBoxLayout()
        
        preview_btn_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ 试听")
        self.play_btn.clicked.connect(self.play_recording)
        self.play_btn.setMinimumHeight(40)
        self.play_btn.setEnabled(False)
        preview_btn_layout.addWidget(self.play_btn)
        
        self.save_btn = QPushButton("💾 保存")
        self.save_btn.clicked.connect(self.save_recording)
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setEnabled(False)
        preview_btn_layout.addWidget(self.save_btn)
        
        preview_layout.addLayout(preview_btn_layout)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        
        close_btn = QPushButton("取消")
        close_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(close_btn)
        
        bottom_layout.addStretch()
        
        self.done_btn = QPushButton("完成")
        self.done_btn.clicked.connect(self.accept)
        self.done_btn.setEnabled(False)
        bottom_layout.addWidget(self.done_btn)
        
        layout.addLayout(bottom_layout)
        self.setLayout(layout)
        
        # 定时器更新时间
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
    
    def start_recording(self):
        """开始录音"""
        if self.is_recording:
            return
        
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            
            # 检查录音设备
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            
            self.is_recording = True
            self.record_start_time = time.time()
            
            # 录音参数
            self.sample_rate = 44100
            self.channels = 1
            
            # 创建录音线程
            self.record_thread = threading.Thread(
                target=self._record_audio,
                daemon=True
            )
            self.record_thread.start()
            
            # 更新UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.time_label.setText("00:00")
            self.status_label.setText("正在录音...")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # 开始计时
            self.timer.start(100)  # 每100ms更新一次
            
        except ImportError:
            QMessageBox.critical(
                self, "缺少依赖",
                "需要安装 sounddevice 和 soundfile 库:\n"
                "pip install sounddevice soundfile"
            )
        except Exception as e:
            QMessageBox.critical(self, "录音失败", f"无法开始录音: {e}")
    
    def _record_audio(self):
        """录音线程函数"""
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            
            # 创建临时文件
            temp_dir = Path("temp_recordings")
            temp_dir.mkdir(exist_ok=True)
            
            temp_file = temp_dir / f"recording_{int(time.time())}.wav"
            
            # 录音
            with sf.SoundFile(
                str(temp_file), mode='w',
                samplerate=self.sample_rate,
                channels=self.channels,
                subtype='PCM_16'
            ) as file:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    callback=lambda indata, frames, time, status: file.write(indata)
                ):
                    while self.is_recording:
                        time.sleep(0.1)
            
            self.recorded_file = str(temp_file)
            
        except Exception as e:
            print(f"录音线程错误: {e}")
    
    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.timer.stop()
        
        # 等待录音线程结束
        if self.record_thread:
            self.record_thread.join(timeout=2)
        
        # 更新UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.done_btn.setEnabled(True)
        
        self.status_label.setText("录音完成")
        self.status_label.setStyleSheet("color: #4CAF50;")
        
        # 显示录音时长
        if self.record_start_time:
            duration = time.time() - self.record_start_time
            self.time_label.setText(f"录音时长: {duration:.1f}秒")
    
    def update_timer(self):
        """更新计时器"""
        if self.record_start_time:
            elapsed = time.time() - self.record_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.time_label.setText(f"{minutes:02d}:{seconds:02d}")
            
            # 更新进度条
            self.progress_bar.setValue(min(int(elapsed), 60))
            
            # 如果超过60秒，自动停止
            if elapsed >= 60:
                self.stop_recording()
    
    def play_recording(self):
        """播放录音"""
        if not self.recorded_file or not os.path.exists(self.recorded_file):
            QMessageBox.warning(self, "错误", "没有可播放的录音文件")
            return
        
        try:
            import sounddevice as sd
            import soundfile as sf
            
            data, fs = sf.read(self.recorded_file)
            sd.play(data, fs)
            sd.wait()
            
        except Exception as e:
            QMessageBox.warning(self, "播放失败", f"无法播放录音: {e}")
    
    def save_recording(self):
        """保存录音"""
        if not self.recorded_file:
            return
        
        # 获取文件名
        voice_name = self.name_edit.text().strip()
        if not voice_name:
            voice_name = f"voice_{int(time.time())}"
        
        # 确保是.wav格式
        if not voice_name.endswith('.wav'):
            voice_name += '.wav'
        
        # 目标路径
        voices_dir = Path("voices")
        voices_dir.mkdir(exist_ok=True)
        dst_path = voices_dir / voice_name
        
        # 复制文件
        try:
            import shutil
            shutil.copy2(self.recorded_file, dst_path)
            
            QMessageBox.information(
                self, "保存成功",
                f"音色已保存到:\n{dst_path}"
            )
            
            self.recorded_file = str(dst_path)
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存音色: {e}")
    
    def get_recorded_file(self) -> str:
        """获取录音文件路径"""
        return self.recorded_file