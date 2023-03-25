import cv2
import numpy as np
from pathlib import Path
import struct

root = Path.cwd()
Names = [['./train', './data/train'], ['./test', './data/test']]
for name in Names:
    # 获取train和test的文件夹
    Dir = root / name[0]
    # 获取imgpaths(所有图片路径)
    imgpaths = []
    # labels(每个图片对应的标签)
    labels = []
    Dirs = Dir.iterdir()
    for eachdir in Dirs:
        for eachimgpath in eachdir.iterdir():
            labels.append(list(str(eachdir)[-1]))
            imgpaths.append(eachimgpath)

    # 定义图片数组，储存所有灰度后的图片，并且之后转为numpy数组
    images = []

    # 遍历当前子文件夹中的每张图片
    for image_path in imgpaths:
        # 传入每张图片
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        # 将当前图片添加到数组中
        images.append(image)

    # 将图片数组转换为numpy数组，并归一化到[0,1]范围内
    images = np.array(images) / 255.0

    # 打印图片和标签数组的形状
    print('Images shape:', images.shape)
    print('Labels len:', len(labels))

    # 定义输出文件的路径
    image_file = name[1] + '-images-idx3-ubyte'
    label_file = name[1] + '-labels-idx1-ubyte'
    Path(image_file).parent.mkdir(parents=True, exist_ok=True)
    Path(label_file).parent.mkdir(parents=True, exist_ok=True)

    # 定义魔数(magic number)
    image_magic = 2051
    label_magic = 2049

    # 获取图片的数量和形状，标签的数量
    num_images, rows, cols = images.shape

    num_labels = len(labels)

    # 打开输出文件并写入魔数、维度数和每个维度的大小
    with open(image_file, 'wb') as f:
        f.write(image_magic.to_bytes(4, byteorder='big'))
        f.write(num_images.to_bytes(4, byteorder='big'))
        f.write(rows.to_bytes(4, byteorder='big'))
        f.write(cols.to_bytes(4, byteorder='big'))
    with open(label_file, 'wb') as f:
        f.write(label_magic.to_bytes(4, byteorder='big'))
        f.write(num_labels.to_bytes(4, byteorder='big'))

    # 遍历每张图片和每个标签，并写入数据本身
    for i in range(num_images):
        # 获取当前图片和标签
        image = images[i]
        label = labels[i]
        # 将当前图片转换为一维数组，并乘以255还原为整数值
        image = image.flatten() * 255.0
        # 将当前图片和标签写入对应的文件中，使用无符号字节类型（'B'）
        with open(image_file, 'ab') as f:
            f.write(struct.pack('>' + 'B' * rows * cols, *[int(x) for x in image]))
        with open(label_file, 'ab') as f:
            f.write(struct.pack('>B', int(label[0])))
