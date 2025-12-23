#!/usr/bin/env python3
"""
依赖检查工具 - AI配音工厂
检查系统环境、Python包、服务等是否满足运行条件
"""

import sys
import os
import subprocess
import platform
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# 屏蔽警告
warnings.filterwarnings('ignore')

class DependencyChecker:
    """依赖检查器"""
    
    def __init__(self):
        self.system_info = {}
        self.results = {
            "python": {},
            "packages": {},
            "system": {},
            "services": {},
            "files": {},
            "summary": {"status": "unknown", "issues": []}
        }
        
    def get_system_info(self):
        """获取系统信息"""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
        }
        
        # 内存信息
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory_total_gb"] = mem.total / (1024**3)
            info["memory_available_gb"] = mem.available / (1024**3)
        except:
            info["memory_info"] = "psutil not available"
        
        self.system_info = info
        return info
    
    def check_python_version(self) -> Tuple[bool, str]:
        """检查Python版本"""
        major, minor, _ = sys.version_info[:3]
        required = (3, 8)
        
        if (major, minor) >= required:
            return True, f"Python {major}.{minor} (>=3.8要求)"
        else:
            return False, f"Python {major}.{minor} (需要>=3.8)"
    
    def check_package(self, package_name: str, import_name: str = None, version_check: bool = False) -> Dict:
        """检查Python包"""
        if import_name is None:
            import_name = package_name
        
        result = {
            "name": package_name,
            "status": "unknown",
            "version": "unknown",
            "message": ""
        }
        
        try:
            # 尝试导入
            module = __import__(import_name)
            result["status"] = "installed"
            
            # 获取版本
            if hasattr(module, '__version__'):
                result["version"] = module.__version__
            elif hasattr(module, 'version'):
                result["version"] = module.version
            
            result["message"] = f"已安装 v{result['version']}"
            
            # 版本检查
            if version_check:
                try:
                    import pkg_resources
                    installed_version = pkg_resources.get_distribution(package_name).version
                    result["version"] = installed_version
                    result["message"] = f"已安装 v{installed_version}"
                except:
                    pass
                    
        except ImportError as e:
            result["status"] = "missing"
            result["message"] = f"未安装: {str(e)}"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"检查失败: {str(e)}"
        
        return result
    
    def check_executable(self, cmd: str, args: List[str] = None, name: str = None) -> Dict:
        """检查可执行文件"""
        if name is None:
            name = cmd
        
        result = {
            "name": name,
            "status": "unknown",
            "path": "unknown",
            "version": "unknown",
            "message": ""
        }
        
        try:
            # 查找可执行文件路径
            path = shutil.which(cmd)
            
            if path:
                result["path"] = path
                result["status"] = "found"
                
                # 获取版本
                try:
                    if args is None:
                        args = ["--version"]
                    
                    output = subprocess.run(
                        [cmd] + args,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if output.returncode == 0:
                        version_lines = output.stdout.strip().split('\n')
                        if version_lines:
                            result["version"] = version_lines[0][:50]  # 截断长版本信息
                            result["message"] = f"找到: {path}"
                        else:
                            result["version"] = "unknown"
                            result["message"] = f"找到但无法获取版本: {path}"
                    else:
                        result["message"] = f"找到但版本检查失败: {path}"
                        
                except subprocess.TimeoutExpired:
                    result["message"] = f"找到但版本检查超时: {path}"
                except Exception as e:
                    result["message"] = f"找到但版本检查错误: {e}"
            else:
                result["status"] = "missing"
                result["message"] = f"未找到 {cmd}，请确保已安装并添加到PATH"
                
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"检查失败: {e}"
        
        return result
    
    def check_directory(self, path: str) -> Dict:
        """检查目录"""
        result = {
            "name": path,
            "status": "unknown",
            "exists": False,
            "writable": False,
            "message": ""
        }
        
        try:
            path_obj = Path(path)
            result["exists"] = path_obj.exists()
            result["writable"] = os.access(str(path_obj), os.W_OK)
            
            if result["exists"]:
                result["status"] = "exists"
                if result["writable"]:
                    result["message"] = f"目录存在且可写"
                else:
                    result["status"] = "warning"
                    result["message"] = f"目录存在但不可写"
            else:
                result["status"] = "missing"
                result["message"] = f"目录不存在"
                
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"检查失败: {e}"
        
        return result
    
    def check_file(self, path: str) -> Dict:
        """检查文件"""
        result = {
            "name": os.path.basename(path),
            "path": path,
            "status": "unknown",
            "exists": False,
            "size": 0,
            "message": ""
        }
        
        try:
            if os.path.exists(path):
                result["exists"] = True
                result["size"] = os.path.getsize(path)
                result["status"] = "exists"
                result["message"] = f"文件存在 ({result['size']}字节)"
            else:
                result["status"] = "missing"
                result["message"] = f"文件不存在"
                
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"检查失败: {e}"
        
        return result
    
    def check_service(self, url: str, name: str = None, timeout: int = 5) -> Dict:
        """检查服务是否运行"""
        if name is None:
            name = url
        
        result = {
            "name": name,
            "url": url,
            "status": "unknown",
            "response_time": 0,
            "message": ""
        }
        
        try:
            import requests
            import time
            
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            end_time = time.time()
            
            result["response_time"] = round((end_time - start_time) * 1000, 2)  # 毫秒
            result["status_code"] = response.status_code
            
            if response.status_code == 200:
                result["status"] = "running"
                result["message"] = f"服务正常 (响应时间: {result['response_time']}ms)"
            else:
                result["status"] = "warning"
                result["message"] = f"服务响应异常 HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            result["status"] = "stopped"
            result["message"] = f"无法连接到服务"
        except requests.exceptions.Timeout:
            result["status"] = "timeout"
            result["message"] = f"连接超时 ({timeout}秒)"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"检查失败: {e}"
        
        return result
    
    def check_cuda(self) -> Dict:
        """检查CUDA"""
        result = {
            "name": "CUDA/GPU",
            "status": "unknown",
            "cuda_available": False,
            "gpu_name": "unknown",
            "gpu_memory_gb": 0,
            "message": ""
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                result["cuda_available"] = True
                result["status"] = "available"
                
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    result["gpu_name"] = torch.cuda.get_device_name(0)
                    result["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
                    result["message"] = f"CUDA可用: {result['gpu_name']} ({result['gpu_memory_gb']}GB)"
                else:
                    result["message"] = "CUDA可用但未检测到GPU"
            else:
                result["status"] = "unavailable"
                result["message"] = "CUDA不可用，将使用CPU模式"
                
        except ImportError:
            result["status"] = "error"
            result["message"] = "torch未安装，无法检查CUDA"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"CUDA检查失败: {e}"
        
        return result
    
    def run_all_checks(self):
        """运行所有检查"""
        print("🔍 AI配音工厂 - 系统依赖检查")
        print("=" * 60)
        
        # 1. 系统信息
        print("\n📋 系统信息:")
        sys_info = self.get_system_info()
        for key, value in sys_info.items():
            print(f"  {key}: {value}")
        
        # 2. Python版本
        print("\n🐍 Python检查:")
        py_ok, py_msg = self.check_python_version()
        self.results["python"] = {"status": "ok" if py_ok else "fail", "message": py_msg}
        print(f"  {'✅' if py_ok else '❌'} {py_msg}")
        
        # 3. CUDA检查
        print("\n🎮 GPU/CUDA检查:")
        cuda_result = self.check_cuda()
        self.results["system"]["cuda"] = cuda_result
        print(f"  {'✅' if cuda_result['status'] == 'available' else '⚠️ '} {cuda_result['message']}")
        
        # 4. 检查Python包
        print("\n📦 Python包检查:")
        
        packages = [
            # 核心包
            ("torch", "torch", True),
            ("torchaudio", "torchaudio", True),
            ("numpy", "numpy", True),
            ("whisper", "whisper", True),
            ("librosa", "librosa", True),
            ("soundfile", "soundfile", True),
            ("requests", "requests", True),
            ("aiohttp", "aiohttp", True),
            ("asyncio", "asyncio", True),
            ("opencv-python", "cv2", True),
            ("PySide6", "PySide6", True),
            ("ffmpeg-python", "ffmpeg", True),
            ("scipy", "scipy", True),
            ("pillow", "PIL", True),
            
            # 可选包
            ("openai", "openai", False),
            ("psutil", "psutil", False),
        ]
        
        for pkg_name, import_name, required in packages:
            result = self.check_package(pkg_name, import_name)
            self.results["packages"][pkg_name] = result
            
            icon = "✅" if result["status"] == "installed" else ("❌" if required else "⚠️ ")
            print(f"  {icon} {pkg_name}: {result['message']}")
        
        # 5. 检查系统工具
        print("\n🔧 系统工具检查:")
        
        executables = [
            ("ffmpeg", ["-version"], "FFmpeg"),
            ("ffprobe", ["-version"], "FFprobe"),
            ("python", ["--version"], "Python"),
            ("pip", ["--version"], "Pip"),
        ]
        
        for cmd, args, name in executables:
            result = self.check_executable(cmd, args, name)
            self.results["system"][name] = result
            
            icon = "✅" if result["status"] == "found" else "❌"
            print(f"  {icon} {name}: {result['message']}")
        
        # 6. 检查目录和文件
        print("\n📁 目录和文件检查:")
        
        directories = [
            ".",
            "voices",
            "characters",
            "outputs",
            "temp",
            "cache"
        ]
        
        for dir_path in directories:
            result = self.check_directory(dir_path)
            self.results["files"][dir_path] = result
            
            if result["status"] == "exists":
                icon = "✅" if result["writable"] else "⚠️"
            elif result["status"] == "missing":
                icon = "⚠️"
            else:
                icon = "❌"
            
            print(f"  {icon} {dir_path}: {result['message']}")
        
        # 7. 检查服务（可选）
        print("\n🌐 服务检查 (可选):")
        
        services = [
            ("http://127.0.0.1:11434/api/tags", "Ollama服务"),
            ("http://127.0.0.1:5021/api/tts", "TTS API"),
            ("http://127.0.0.1:8188", "ComfyUI服务"),
        ]
        
        for url, name in services:
            result = self.check_service(url, name)
            self.results["services"][name] = result
            
            if result["status"] == "running":
                icon = "✅"
            elif result["status"] in ["warning", "timeout"]:
                icon = "⚠️"
            else:
                icon = "⚪"  # 可选服务不运行时用中性图标
            
            print(f"  {icon} {name}: {result['message']}")
        
        # 8. 总结
        print("\n" + "=" * 60)
        print("📊 检查总结:")
        
        # 统计问题
        issues = []
        
        # Python版本问题
        if not py_ok:
            issues.append("Python版本过低，需要3.8或更高版本")
        
        # 必需包缺失
        missing_required = []
        for pkg_name, result in self.results["packages"].items():
            # 检查是否为必需包
            for pkg_info in packages:
                if pkg_info[0] == pkg_name and pkg_info[2]:  # required=True
                    if result["status"] != "installed":
                        missing_required.append(pkg_name)
        
        if missing_required:
            issues.append(f"缺少必需Python包: {', '.join(missing_required)}")
        
        # FFmpeg缺失
        if self.results["system"].get("FFmpeg", {}).get("status") != "found":
            issues.append("FFmpeg未找到，请安装并添加到PATH")
        
        # 目录不可写
        unwritable_dirs = []
        for dir_path, result in self.results["files"].items():
            if result.get("exists") and not result.get("writable", False):
                unwritable_dirs.append(dir_path)
        
        if unwritable_dirs:
            issues.append(f"目录不可写: {', '.join(unwritable_dirs)}")
        
        # 更新总结
        if not issues:
            self.results["summary"]["status"] = "ready"
            self.results["summary"]["message"] = "✅ 所有依赖已满足，可以启动AI配音工厂！"
            print("✅ 所有依赖已满足，可以启动AI配音工厂！")
        else:
            self.results["summary"]["status"] = "issues"
            self.results["summary"]["issues"] = issues
            self.results["summary"]["message"] = f"⚠️  发现{len(issues)}个问题需要解决"
            
            print(f"⚠️  发现{len(issues)}个问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            
            print("\n💡 建议:")
            if missing_required:
                print("  1. 安装缺失的Python包:")
                print(f"     pip install {' '.join(missing_required)}")
            
            if "FFmpeg未找到" in str(issues):
                print("  2. 安装FFmpeg:")
                print("     - Windows: 下载并添加bin目录到PATH")
                print("     - Linux: sudo apt install ffmpeg")
                print("     - Mac: brew install ffmpeg")
            
            if unwritable_dirs:
                print("  3. 修复目录权限:")
                for dir_path in unwritable_dirs:
                    print(f"     chmod 755 {dir_path}  # Linux/Mac")
                    print(f"     # 或修改文件夹属性 (Windows)")
        
        # 生成修复脚本
        self.generate_fix_script(issues)
        
        # 保存结果
        self.save_results()
        
        return self.results
    
    def generate_fix_script(self, issues):
        """生成修复脚本"""
        if not issues:
            return
        
        script_content = """#!/usr/bin/env python3
# AI配音工厂依赖修复脚本
import sys
import subprocess
import os

def run_command(cmd, desc):
    print(f"正在{desc}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {desc}成功")
            return True
        else:
            print(f"❌ {desc}失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {desc}异常: {e}")
        return False

print("🔧 AI配音工厂依赖修复脚本")
print("=" * 50)

# 修复目录权限
directories = ["voices", "characters", "outputs", "temp", "cache"]
for dir_name in directories:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ 创建目录: {dir_name}")

print("\\n修复完成！请重新运行check_deps.py检查依赖。")
"""
        
        with open("fix_deps.py", "w", encoding="utf-8") as f:
            f.write(script_content)
        
        print(f"\n📝 已生成修复脚本: fix_deps.py")
        print("  运行: python fix_deps.py")
    
    def save_results(self):
        """保存检查结果"""
        try:
            with open("dependency_check.json", "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 检查结果已保存到: dependency_check.json")
        except Exception as e:
            print(f"⚠️  保存结果失败: {e}")
    
    def quick_fix(self):
        """快速修复常见问题"""
        print("\n🔧 尝试快速修复...")
        
        # 创建必要目录
        for dir_name in ["voices", "characters", "outputs", "temp", "cache"]:
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"✅ 确保目录存在: {dir_name}")
            except Exception as e:
                print(f"⚠️  创建目录失败 {dir_name}: {e}")
        
        print("快速修复完成。")

def main():
    """主函数"""
    try:
        checker = DependencyChecker()
        results = checker.run_all_checks()
        
        # 提供下一步建议
        print("\n" + "=" * 60)
        print("🚀 下一步建议:")
        
        summary = results["summary"]
        
        if summary["status"] == "ready":
            print("1. 直接启动AI配音工厂:")
            print("   python main.py")
            print("\n2. 如有问题，检查服务是否运行:")
            print("   - Ollama: ollama serve")
            print("   - TTS服务: python api.py --port 5021")
            print("   - ComfyUI: python main.py --port 8188")
        else:
            print("1. 先解决上述问题")
            print("2. 运行生成的修复脚本:")
            print("   python fix_deps.py")
            print("3. 或手动安装缺失依赖")
            print("4. 重新运行检查:")
            print("   python check_deps.py")
        
        print("\n🔄 按Enter键重新检查，或直接关闭窗口...")
        input()
        
    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 检查过程发生错误: {e}")
        traceback.print_exc()
        input("\n按Enter键退出...")

if __name__ == "__main__":
    main()