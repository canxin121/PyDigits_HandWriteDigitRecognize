import cv2
from pathlib import Path
from prepic import image_preprocessing

newpath = Path(Path.cwd()/'new')
Path.mkdir(newpath,exist_ok=True)
for i in range(0,10):
    Path.mkdir(newpath/str(i),exist_ok=True)

root = Path.cwd()/'old'
dirs = root.iterdir()
for eachdir in dirs:
    for eachpic in eachdir.iterdir():
        imglist = image_preprocessing(eachpic)
        new_eachpic = str(eachpic).replace("old", "new")
        cv2.imwrite(new_eachpic, imglist)
        print(new_eachpic)
print('ok')
