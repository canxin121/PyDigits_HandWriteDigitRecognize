from pathlib import Path
from PIL import Image

path = Path.cwd()
path = path / 'old'
path = path.iterdir()
for p1 in path:
    print(str(p1))
    i = 0
    for eachpath in p1.iterdir():
        img = Image.open(eachpath)
        img.save(eachpath, 'PNG')
        img.close()
        eachpath.rename(p1 / f'{i}.png')
        i += 1
