# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Agrinova.py'],
    pathex=[],
    binaries=[],
    datas=[('agrinova_model_1.pkl', '.'), ('agrinova_model_2.pkl', '.'), ('agrinova_model_3.pkl', '.')],
    hiddenimports=['sklearn', 'sklearn.ensemble', 'sklearn.metrics', 'seaborn', 'matplotlib', 'matplotlib.backends.backend_tkagg', 'pandas', 'numpy'],
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
    name='Agrinova',
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
    icon=['ML.ico'],
)
