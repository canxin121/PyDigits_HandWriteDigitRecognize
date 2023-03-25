import cv2
import numpy as np


def image_preprocessing(imgpath):
    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), -1)
    # 对图片边界进行扩展，防止后续裁剪时因边界值造成越界
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(200, 200, 200))
    # 图像灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像高斯降噪
    img = cv2.GaussianBlur(img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    # 检测图像变化值最大的边缘
    img_edge1 = cv2.Canny(img, 100, 200)

    # 检测有效图像四面边界
    rows, cols = np.where(img_edge1 == 255)
    # 上
    min_row = np.min(rows)
    # 下
    max_row = np.max(rows)
    # 左
    min_col = np.min(cols)
    # 右
    max_col = np.max(cols)
    rows = max_row - min_row
    cols = max_col - min_col
    # 方形切片截取有效图片
    img = img[min_row:max_row + 1, min_col:max_col + 1]

    # 将图像转变成纯黑白
    img = np.where(img < 150, 255, 0)

    height, width = img.shape[:2]

    # 计算图像的最大边长
    max_side = max(height, width)

    # 创建一个正方形的白色背景
    square = np.full((max_side, max_side), 0, dtype=np.uint8)

    # 计算将图像放置在正方形中心的坐标
    x_pos = (max_side - width) // 2
    y_pos = (max_side - height) // 2

    # 将图像放置在正方形中心
    square[y_pos:y_pos + height, x_pos:x_pos + width] = img
    img = cv2.copyMakeBorder(square, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    newimg = cv2.resize(img, (28, 28), interpolation=cv2.INTER_NEAREST)

    return newimg
