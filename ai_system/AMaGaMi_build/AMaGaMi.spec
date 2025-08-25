# -*- mode: python ; coding: utf-8 -*-

# Step 1: ターミナルで以下のコマンドを実行して、'terminal-notifier.app'へのパスを調べてコピーしてください
# conda activate amagami_env
# python -c "import pync, os; print(os.path.join(os.path.dirname(pync.__file__), 'vendor', 'terminal-notifier.app'))"

a = Analysis(
    ['fatigue_detection_app.py'],  # あなたのメインスクリプト名
    pathex=[],
    binaries=[],
    datas=[('resources', 'resources')],  # resourcesフォルダ全体を同梱
    hiddenimports=[
        'sip',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'torch.hub',
        'torchvision.models',
    ],  # 隠れた依存関係をすべて追加
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5'],  # PyQt5との競合を避けるため除外
    windowed=True,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# 実行ファイル(EXE)を作成
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AMaGaMi',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# EXEと各種リソースをまとめる
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AMaGaMi',
)

app = BUNDLE(
    coll,
    name='AMaGaMi.app',
    icon='resources/icon/app_icon.icns',
    bundle_identifier='com.yourdomain.amigami',
    info_plist={
        'NSCameraUsageDescription': '疲労度を検出するためにカメラへのアクセスが必要です。'
    } # ← この info_plist 辞書を追加
)