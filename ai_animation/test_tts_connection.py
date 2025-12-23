#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 TTS API 连接
"""

import requests
import base64
import json

def test_tts_api():
    """测试 TTS API 连接"""
    api_url = "http://127.0.0.1:5021/api/tts"
    
    # 1. 测试根路径
    print("1. 测试API根路径...")
    try:
        response = requests.get(api_url.replace('/api/tts', ''), timeout=5)
        print(f"   API根路径状态: {response.status_code}")
        if response.status_code == 404:
            print("   ⚠️ 返回404是正常的，因为API路径是/api/tts")
    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
    
    # 2. 读取一个示例音频文件作为参考音色
    print("\n2. 准备测试数据...")
    try:
        # 使用一个简单的音频文件，如果没有，可以创建一个临时的
        test_audio_path = "test_voice.wav"
        # 如果没有测试文件，可以使用一个简单的替代方法
        import numpy as np
        import soundfile as sf
        
        # 创建一个1秒的静音音频
        sr = 24000
        silence = np.zeros(sr, dtype=np.float32)
        sf.write(test_audio_path, silence, sr)
        
        with open(test_audio_path, 'rb') as f:
            audio_bytes = f.read()
        prompt_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        print("   ✅ 测试音频准备完成")
    except Exception as e:
        print(f"   ❌ 准备测试音频失败: {e}")
        return
    
    # 3. 发送测试请求
    print("\n3. 发送TTS请求...")
    payload = {
        "tts_text": "你好，这是一个测试。Hello, this is a test.",
        "prompt_audio": prompt_audio_b64,
        "emo_control_method": 0,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 30,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 1500,
        "num_beams": 3
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        print(f"   TTS API响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                print("   ✅ TTS API工作正常！")
                print(f"   采样率: {result.get('sample_rate')}")
                print(f"   音频数据长度: {len(result.get('audio', ''))} 字符")
                
                # 保存生成的音频
                audio_bytes = base64.b64decode(result["audio"])
                with open("test_output.wav", "wb") as f:
                    f.write(audio_bytes)
                print("   ✅ 音频已保存到: test_output.wav")
            else:
                print(f"   ❌ API返回错误: {result.get('message')}")
        else:
            print(f"   ❌ HTTP错误: {response.status_code}")
            print(f"   响应内容: {response.text[:500]}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ 请求失败: {e}")
    except Exception as e:
        print(f"   ❌ 其他错误: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("TTS API 连接测试")
    print("=" * 50)
    test_tts_api()