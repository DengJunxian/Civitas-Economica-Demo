import sys
import os
import subprocess
def check_environment():
    """检查运行环境依赖"""
    # 局部导入以避免提前加载
    import os
    import importlib
    import config
    
    cfg = config.GLOBAL_CONFIG
    print(f"[*] 初始化 {cfg.PROJECT_NAME} (v{cfg.VERSION})")
    
    # 检查 API Key
    if not cfg.DEEPSEEK_API_KEY:
        print("\n[!] 警告: 未检测到 DEEPSEEK_API_KEY 环境变量。")
        key = input("请输入 DeepSeek API Key (回车跳过使用默认/Mock): ").strip()
        if key:
            # 设置环境变量
            os.environ["DEEPSEEK_API_KEY"] = key
            
            # 重新加载 Config
            importlib.reload(config)
            cfg = config.GLOBAL_CONFIG
            print("[*] API Key 已配置并重新加载 GLOBAL_CONFIG")
        else:
            print("[!] 继续运行，部分AI功能将不可用。")
    return cfg

def run_ui():
    """启动 Streamlit 界面"""
    print("[*] 正在启动可视化控制台...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "app.py")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] 找不到文件: {file_path}")
        return

    try:
        env = os.environ.copy()
        cmd = [sys.executable, "-m", "streamlit", "run", file_path]
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
    except Exception as e:
        print(f"[!] 启动失败: {e}")

if __name__ == "__main__":
    GLOBAL_CONFIG = check_environment()
    run_ui()