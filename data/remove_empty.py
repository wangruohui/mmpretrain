from pathlib import Path

def remove_empty_folder(root='res'):
    root = Path(root)
    for p in root.iterdir():
        if p.is_dir():
            if not list(p.iterdir()):
                p.rmdir()

remove_empty_folder()