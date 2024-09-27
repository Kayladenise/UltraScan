# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None

# Collect necessary data files
datas = collect_data_files('ttkbootstrap') + \
        collect_data_files('PIL') + \
        collect_data_files('openpyxl') + \
        collect_data_files('tensorflow') + \
        collect_data_files('keras') + \
        copy_metadata('ttkbootstrap') + \
        copy_metadata('openpyxl') + \
        copy_metadata('tensorflow') + \
        copy_metadata('keras') + [
            ('best_model.h5', '.'),  # Include your model file
        ]

# Manually add TensorFlow hidden imports
hiddenimports = collect_submodules('tensorflow') + [
    'tensorflow.python.platform',
    'tensorflow.python._pywrap_tensorflow_internal',
    'tensorflow.python.eager.context',
    'tensorflow.python.framework.ops',
    'tensorflow.python.profiler',
    'tensorflow.python.framework.load_library',
    'tensorflow.python.keras.utils',
    'tensorflow.python.keras.engine',
    'tensorflow.python.training.tracking.data_structures',
    'tensorflow.python.saved_model',
]

a = Analysis(
    ['main.py'],
    pathex=['C:\\Users\\kayla\\OneDrive\\Desktop\\Tk Project'],
    binaries=[],
    datas=datas + [('images', 'images')],  # Include your images folder
    hiddenimports=hiddenimports,
    hookspath=[],  # You can specify custom hooks here if needed
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='UltraScan',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # This will include a command prompt when the exe runs
    icon='images/main_icon.png'  # Specify the path to your icon file if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='UltraScan',
)
