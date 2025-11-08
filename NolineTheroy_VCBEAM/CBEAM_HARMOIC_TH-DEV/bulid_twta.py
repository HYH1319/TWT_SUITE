#!/usr/bin/env python3
"""
TWT行波管计算器打包工具 - 最终修正版
解决Tkinter和Matplotlib后端问题
"""
import os
import sys
import subprocess
import platform
from pathlib import Path
import matplotlib

def main():
    print("""
    === TWT行波管计算器打包工具 ===
    [最终修正模式]
    """)
    
    # 1. 清理旧构建
    print("\n步骤 1/3: 清理旧构建...")
    for folder in ["build", "dist"]:
        p = Path(folder)
        if p.exists():
            try:
                if platform.system() == "Windows":
                    os.system(f'rmdir /s /q "{folder}"')
                else:
                    os.system(f'rm -rf "{folder}"')
                print(f"已清理: {folder}")
            except Exception as e:
                print(f"清理失败 {folder}: {str(e)}")

    # 2. 准备打包命令
    print("\n步骤 2/3: 准备打包参数...")
    script_file = "PYQT_MAINGUI_SINGLE_FREQUNCY_VerHARMONIC.py"
    
    if not Path(script_file).exists():
        print(f"[错误] 主脚本文件 '{script_file}' 不存在！")
        return

    try:
        mpl_data_path = matplotlib.get_data_path()
        print(f"找到 Matplotlib 数据路径: {mpl_data_path}")
        mpl_include_option = f"--include-data-dir={mpl_data_path}=matplotlib/mpl-data"
    except Exception as e:
        print(f"[错误] 无法获取 Matplotlib 数据路径: {e}")
        return

    # 构建打包命令
    cmd = [
        sys.executable,
        "-m", "nuitka",
        "--onefile",
        "--standalone",
        # 关键修改1：启用tkinter插件
        "--enable-plugin=tk-inter",
        # 关键修改2：使用新的控制台模式参数
        "--windows-console-mode=disable",
        # 其他参数
        "--follow-imports",
        mpl_include_option,
        "--include-module=TWT_CORE_SIMP",
        "--include-module=_TWT_CORE_NOLINE_COMPLEX_VCBEAM_HARMONIC_VF",
        "--mingw64"
    ]
    
    for dir_name in ["config", "Results"]:
        if Path(dir_name).exists():
            cmd.append(f"--include-data-dir={dir_name}={dir_name}")
            print(f"添加数据目录: {dir_name}")
    
    cmd.append(script_file)

    # 3. 执行打包
    print("\n步骤 3/3: 开始Nuitka打包...")
    print("\n执行命令:")
    print(" ".join(f'"{c}"' if ' ' in c else c for c in cmd))
    print("\n请耐心等待打包完成...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("\n" + "="*50)
        print("[错误] 打包失败！")
        print("错误信息:")
        print(e.stdout)
        print(e.stderr)
        print("="*50)
        return

    # 完成提示
    print("\n" + "="*50)
    exe_name = script_file.replace(".py", ".exe")
    print(f"打包成功！")
    print(f"输出文件位置: dist/{exe_name}")
    print("="*50)

if __name__ == '__main__':
    main()
