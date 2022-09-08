# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata

datas = []
datas += copy_metadata('torchvision')
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('importlib_metadata')


block_cipher = None


a = Analysis(
    ['F:\\stable-diffusion-main StableDiffusionUI.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['huggingface_hub.repository', 'ldm.models.diffusion.ddpm', 'ldm.modules.encoders', 'ldm.modules.encoders.modules', 'ldm.modules.diffusionmodules.openaimodel'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='stable-diffusion-main StableDiffusionUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='F:\\stable-diffusion-main\\UI\\icons\\StableDifusion.ico',
)
