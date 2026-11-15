# -*- mode: python ; coding: utf-8 -*-
"""MoodDetector build: mood_ui.py + fused 8-state detector.

collect_all for mediapipe (its native C extension mediapipe.tasks.c is
loaded via importlib.resources, invisible to static analysis) and
openvino (native runtime DLLs + plugin modules). torch/torchvision are
excluded: the detector only needs rules + OpenVINO at runtime.
"""
from PyInstaller.utils.hooks import collect_all

mp_bins, mp_datas, mp_hidden = collect_all("mediapipe")
ov_bins, ov_datas, ov_hidden = collect_all("openvino")

a = Analysis(
    ['mood_ui.py'],
    pathex=[],
    binaries=mp_bins + ov_bins,
    datas=[('model_ir', 'model_ir'), ('assets/face_landmarker.task', 'assets')]
    + mp_datas + ov_datas,
    hiddenimports=['mediapipe.tasks.c'] + mp_hidden + ov_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision'],
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
    name='MoodDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
