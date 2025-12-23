# smart_tts_launcher.py
# 智能TTS服务启动器

import os
import sys
import time
import requests
import subprocess
from pathlib import Path
import threading
import webbrowser

class SmartTTSLauncher:
    def __init__(self):
        self.indextts_path = Path(r"D:\03pei yin\indexTTS2-Cu128")
        self.port = 5021
        self.api_process = None
        self.service_ready = False
        
    def check_service(self, timeout=3):
        """检查服务是否运行"""
        try:
            response = requests.get(
                f"http://127.0.0.1:{self.port}/api/health",
                timeout=timeout
            )
            return response.status_code == 200
        except:
            return False
    
    def start_indextts_service(self):
        """启动Index-TTS服务"""
        if not self.indextts_path.exists():
            print(f"❌ 整合包路径不存在: {self.indextts_path}")
            return False
        
        # 查找启动脚本
        bat_files = [
            self.indextts_path / "启动API服务_deepspeed加速版(推荐).bat",
            self.indextts_path / "run_api.bat",
            self.indextts_path / "start.bat",
        ]
        
        bat_file = None
        for file in bat_files:
            if file.exists():
                bat_file = file
                break
        
        if not bat_file:
            print(f"❌ 未找到启动脚本")
            return False
        
        print(f"🚀 启动Index-TTS服务: {bat_file.name}")
        
        try:
            # 在新的命令窗口中启动服务
            self.api_process = subprocess.Popen(
                f'start /B "{bat_file}"',
                shell=True,
                cwd=self.indextts_path
            )
            
            print("⏳ 等待服务启动...")
            
            # 等待服务就绪
            for i in range(30):  # 最多等待30秒
                time.sleep(1)
                if self.check_service(timeout=1):
                    print(f"✅ 服务启动成功！({i+1}秒)")
                    self.service_ready = True
                    return True
                
                if i % 5 == 0:
                    print(f"   等待中... {i+1}秒")
            
            print("❌ 服务启动超时")
            return False
            
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return False
    
    def create_tts_proxy(self):
        """创建TTS代理服务（如果Index-TTS服务有问题）"""
        print("🔄 创建TTS代理服务...")
        
        proxy_code = '''
from flask import Flask, request, jsonify
import requests
import base64

app = Flask(__name__)

# Index-TTS服务地址
INDEXTTS_URL = "http://127.0.0.1:5021/api/tts"

@app.route('/api/tts', methods=['POST'])
def tts_proxy():
    """代理请求到Index-TTS"""
    try:
        # 转发请求
        response = requests.post(INDEXTTS_URL, json=request.json, timeout=30)
        
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({
                "status": "error",
                "message": f"Index-TTS返回错误: {response.status_code}"
            }), response.status_code
            
    except requests.ConnectionError:
        return jsonify({
            "status": "error",
            "message": "无法连接到Index-TTS服务"
        }), 503
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"代理错误: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    try:
        # 检查底层服务
        response = requests.get("http://127.0.0.1:5021/api/health", timeout=3)
        return jsonify({
            "status": "healthy",
            "upstream": "Index-TTS",
            "upstream_status": "running" if response.status_code == 200 else "down"
        }), 200
    except:
        return jsonify({
            "status": "degraded",
            "upstream": "Index-TTS",
            "upstream_status": "down",
            "message": "代理运行但底层服务不可用"
        }), 200

if __name__ == '__main__':
    print("🚀 TTS代理服务启动 (端口: 5022)")
    print("   转发到: http://127.0.0.1:5021")
    app.run(host="0.0.0.0", port=5022, debug=False)
'''
        
        # 保存代理服务文件
        proxy_file = Path("tts_proxy.py")
        proxy_file.write_text(proxy_code, encoding="utf-8")
        
        # 启动代理服务
        import subprocess
        self.proxy_process = subprocess.Popen(
            [sys.executable, "tts_proxy.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        time.sleep(2)
        return True
    
    def open_web_interface(self):
        """打开Web界面"""
        urls = [
            f"http://127.0.0.1:{self.port}",
            f"http://127.0.0.1:{self.port}/api/health",
        ]
        
        for url in urls:
            try:
                webbrowser.open(url)
                print(f"🌐 打开Web界面: {url}")
                break
            except:
                pass
    
    def setup_voices_directory(self):
        """设置音色目录"""
        # 在你的AI配音工厂中创建voices目录
        local_voices = Path("voices")
        indextts_voices = self.indextts_path / "voices"
        
        if not local_voices.exists():
            local_voices.mkdir()
            print(f"📁 创建本地音色目录: {local_voices}")
        
        # 检查整合包中的音色
        if indextts_voices.exists():
            print(f"📁 整合包音色目录: {indextts_voices}")
            files = list(indextts_voices.glob("*"))
            if files:
                print(f"   找到 {len(files)} 个音色文件")
                
                # 创建符号链接或复制提示
                print("💡 建议:")
                print(f"   1. 将音色文件复制到: {local_voices}")
                print(f"   2. 或在config.json中设置:")
                print(f'      "voices_dir": "{indextts_voices}"')
        
        return str(local_voices)
    
    def create_config(self):
        """创建配置文件"""
        config = {
            "project_name": "AI配音工厂",
            "version": "2.0.0",
            "tts_api": f"http://127.0.0.1:{self.port}/api/tts",
            "indextts_path": str(self.indextts_path),
            "voices_dir": "voices",
            "whisper_model": "large-v3",
            "sample_rate": 24000,
            "use_gpu": True,
            "character_styles": {
                "哆啦A梦": "李云龙",
                "大雄": "王境泽",
                "静香": "佟湘玉",
                "胖虎": "张飞",
                "小夫": "马保国"
            }
        }
        
        import json
        config_file = Path("config.json")
        config_file.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        print(f"📝 创建配置文件: {config_file}")
        return config
    
    def run(self):
        """运行启动器"""
        print("""
        ╔══════════════════════════════════════════════════╗
        ║          AI配音工厂 - 智能启动器                ║
        ╚══════════════════════════════════════════════════╝
        """)
        
        print(f"📦 Index-TTS整合包: {self.indextts_path}")
        
        # 步骤1: 检查服务
        print("\n1️⃣ 检查TTS服务状态...")
        if self.check_service():
            print("✅ Index-TTS服务已在运行")
            self.service_ready = True
        else:
            print("❌ Index-TTS服务未运行")
            
            # 询问是否启动
            choice = input("   是否启动Index-TTS服务? (y/n): ").lower()
            if choice == 'y':
                if not self.start_indextts_service():
                    print("⚠️  启动失败，使用代理模式")
                    self.create_tts_proxy()
                    self.port = 5022  # 切换到代理端口
        
        # 步骤2: 设置音色目录
        print("\n2️⃣ 设置音色目录...")
        voices_dir = self.setup_voices_directory()
        
        # 步骤3: 创建配置
        print("\n3️⃣ 创建配置文件...")
        config = self.create_config()
        
        # 步骤4: 验证
        print("\n4️⃣ 验证配置...")
        if self.check_service():
            print(f"✅ TTS API可用: {config['tts_api']}")
            
            # 测试请求
            try:
                test_data = {
                    "text": "测试AI配音工厂",
                    "voice_link": "李云龙"
                }
                response = requests.post(
                    config['tts_api'],
                    json=test_data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    print("✅ TTS测试请求成功")
                else:
                    print(f"⚠️  TTS测试请求失败: {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️  TTS测试错误: {e}")
        
        # 步骤5: 启动选项
        print("\n5️⃣ 下一步操作:")
        print("   A. 🎭 启动AI配音工厂GUI")
        print("   B. 🌐 打开TTS Web界面")
        print("   C. 📋 查看配置")
        print("   D. 🚪 退出")
        
        choice = input("\n请选择 (A/B/C/D): ").upper()
        
        if choice == 'A':
            # 启动AI配音工厂
            if Path("main.py").exists():
                print("\n🚀 启动AI配音工厂...")
                subprocess.Popen([sys.executable, "main.py"])
            else:
                print("❌ 未找到main.py")
                
        elif choice == 'B':
            self.open_web_interface()
            
        elif choice == 'C':
            import json
            print("\n📋 当前配置:")
            print(json.dumps(config, ensure_ascii=False, indent=2))
        
        print("\n🎉 设置完成！")
        print(f"\n💡 使用说明:")
        print(f"   1. TTS服务地址: http://127.0.0.1:{self.port}")
        print(f"   2. 音色目录: {voices_dir}")
        print(f"   3. 启动AI配音工厂: python main.py")
        
        input("\n按Enter键退出...")

if __name__ == "__main__":
    launcher = SmartTTSLauncher()
    launcher.run()