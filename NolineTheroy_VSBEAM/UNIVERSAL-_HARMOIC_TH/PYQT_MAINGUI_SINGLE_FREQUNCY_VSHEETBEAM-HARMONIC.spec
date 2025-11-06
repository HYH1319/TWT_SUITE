# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['E:\\Downloads\\TWT_PreRelase\\TWT_HF_TOOLS_PreRlease\\NolineTheroy_VSBEAM\\UNIVERSAL-_HARMOIC_TH\\PYQT_MAINGUI_SINGLE_FREQUNCY_VSHEETBEAM-HARMONIC.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PYQT_MAINGUI_SINGLE_FREQUNCY_VSHEETBEAM-HARMONIC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
