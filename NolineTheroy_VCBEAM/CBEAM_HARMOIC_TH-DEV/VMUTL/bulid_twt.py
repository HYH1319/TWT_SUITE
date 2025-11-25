#!/usr/bin/env python3
"""
TWT行波管计算器极简打包工具 - 跳过依赖检查版
专为已配置完整环境设计
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def main():
    print("""
    === TWT行波管计算器极简打包工具 ===
    [跳过依赖检查模式]
    """)
    
    # 1. 清理旧构建
    print("\n步骤 1/2: 清理旧构建...")
    for folder in ["build", "dist"]:
        p = Path(folder)
        if p.exists():
            try:
                # Windows/Linux兼容的删除方式
                if platform.system() == "Windows":
                    os.system(f'rmdir /s /q "{folder}"')
                else:
                    os.system(f'rm -rf "{folder}"')
                print(f"已清理: {folder}")
            except Exception as e:
                print(f"清理失败 {folder}: {str(e)}")
    
    # 2. 使用Nuitka直接打包（跳过所有依赖检查）
    print("\n步骤 2/2: 开始Nuitka打包...")
    script_file = "PYQT_NOLine__PVTOPT-DDE_AL_VCBHARMONIC_VFMUTLIFREQ.py"
    
    # 构建打包命令
    cmd = [
        sys.executable,  # 使用当前Python解释器
        "-m", "nuitka",
        "--onefile",
        "--standalone",
        "--follow-imports",
        "--enable-plugin=pyqt5",
        "--include-package-data=matplotlib",
        "--windows-disable-console",
        "--mingw64"
    ]
    
    # 添加可选的目录包含（确保存在）
    config_dir = "config"
    results_dir = "Results"
    
    if Path(config_dir).exists():
        cmd.append(f"--include-data-dir={config_dir}=config")
    if Path(results_dir).exists():
        cmd.append(f"--include-data-dir={results_dir}=Results")
    
    cmd.append(script_file)
    
    # 执行打包命令
    print("\n执行命令:")
    print(" ".join(cmd))
    print("\n请耐心等待打包完成...")
    
    result = subprocess.run(cmd)
    
    # 完成提示
    print("\n" + "="*50)
    if result.returncode == 0:
        exe_name = script_file.replace(".py", ".exe")
        print(f"打包成功！输出文件位置: dist/{exe_name}")
    else:
        print("打包失败，请检查错误信息")
    print("="*50)

if __name__ == '__main__':
    main()