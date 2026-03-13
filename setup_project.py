import os

folders = [
    'data',
    'notebooks',
    'src',
    'src/models',
    'src/preprocessing',
    'src/utils'
]

files = [
    'src/__init__.py',
    'src/data_loader.py',
    'src/preprocessor.py',
    'src/model.py',
    'src/advisor.py',
    'requirements.txt',
    'README.md',
    '.gitignore'
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    with open(file, 'w') as f:
        pass

print("✅ Project structure created successfully!")