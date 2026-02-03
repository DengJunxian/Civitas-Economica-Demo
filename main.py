import sys
import os
import subprocess
from config import GLOBAL_CONFIG

def check_environment():
    """检查运行环境依赖"""
    print(f"[*] 初始化 {GLOBAL_CONFIG.PROJECT_NAME} (v{GLOBAL_CONFIG.VERSION})")
    print(f"[*] 市场制度: 涨跌幅 {GLOBAL_CONFIG.PRICE_LIMIT*100:.0f}%, "
          f"印花税 {GLOBAL_CONFIG.TAX_RATE_STAMP*10000:.0f}bps, "
          f"T+1: {GLOBAL_CONFIG.T_PLUS_1}")
    
    # 简单检查 API Key
    if not GLOBAL_CONFIG.DEEPSEEK_API_KEY:
        print("[!] 警告: 未检测到 DEEPSEEK_API_KEY 环境变量。请在 UI 中手动输入。")
    else:
        print("[*] API Key: 已配置")

def run_ui():
    """启动 Streamlit 界面"""
    print("[*] 正在启动可视化控制台...")
    file_path = os.path.abspath("app.py")
    try:
        # 使用 subprocess 调用 streamlit
        subprocess.run(["streamlit", "run", file_path], check=True)
    except KeyboardInterrupt:
        print("\n[!] 用户终止程序")
    except Exception as e:
        print(f"[!] 启动失败: {e}")

if __name__ == "__main__":
    check_environment()
    
    print("\n请选择启动模式:")
    print("1. 启动 Streamlit 可视化大屏 (UI)")
    print("2. 运行无头模式基准测试 (Headless Benchmark) - [TODO]")
    
    choice = input("请输入选项 [1]: ").strip()
    if choice == "2":
        print("该功能尚在开发序列 Sequence 2 中...")
    else:
        run_ui()