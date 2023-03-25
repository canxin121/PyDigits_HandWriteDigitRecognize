import cv2 as cv
import numpy as np

def sort_contour(contours,height,num_row):
    contours = sorted(contours, key=lambda c: cv.minAreaRect(c)[0][1])
    # 根据轮廓中心y坐标将轮廓分成多行
    rows = []
    current_row = []
    current_y = None
    for contour in contours:
        center_x, center_y = cv.minAreaRect(contour)[0]
        if current_y is None:
            current_y = center_y
        if num_row == 1:
            if abs(center_y - current_y) > height:  # 根据需要调整阈值
                rows.append(sorted(current_row, key=lambda c: cv.minAreaRect(c)[0][0]))
                current_row = []
                current_y = center_y
            current_row.append(contour)
        else:
            if abs(center_y - current_y) > height / num_row * 0.4:  # 根据需要调整阈值
                rows.append(sorted(current_row, key=lambda c: cv.minAreaRect(c)[0][0]))
                current_row = []
                current_y = center_y
            current_row.append(contour)
    rows.append(sorted(current_row, key=lambda c: cv.minAreaRect(c)[0][0]))
    # 将所有轮廓连接成一个列表
    newcontours = [contour for row in rows for contour in row]
    return newcontours

def crop_image(img,num_row):

    # 将图像转换为numpy数组
    img = np.array(img)
    # cv.imwrite('aaa.png', img)
    # 计算图像的高度和宽度
    h_min, w_min = img.shape[:2]
    # 将图像转换为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 对图像进行阈值处理进行二值化
    ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
    # 为形态学操作定义一个内核
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))
    # 执行形态学闭合
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # 在图像中查找轮廓
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 用书写习惯的排序算法排序轮廓
    contours = sort_contour(contours,h_min,num_row)
    # 创建一个列表来存储裁剪后的图像
    newimgs = []
    i=0
    for none, cnt in enumerate(contours):
        # 获取外接矩形
        x, y, w, h = cv.boundingRect(cnt)
        # 绘制矩形框
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 提取数字并保存为新的图片，如果满足条件
        digit = thresh[y:y + h, x:x + w]
        # 判断面积是否满足是数字的大小
        wd = digit.shape[1]
        hd = digit.shape[0]
        if hd > (h_min * 0.2 / num_row) or (num_row == 1 and hd > h_min * 0.1):
            # 保存图片
            if wd / hd > 2:
                continue
            height, width = digit.shape[:2]
            # 计算图像的最大边长
            max_side = max(height, width)
            # 创建一个正方形的白色背景
            square = np.full((max_side, max_side), 0, dtype=np.uint8)
            # 计算将图像放置在正方形中心的坐标
            x_pos = (max_side - width) // 2
            y_pos = (max_side - height) // 2
            # 将图像放置在正方形中心
            square[y_pos:y_pos + height, x_pos:x_pos + width] = digit
            # 将图像放置在正方形中心后，将其放置在一个黑色背景上
            img = cv.copyMakeBorder(square, 15, 15, 15, 15, cv.BORDER_CONSTANT, value=0)
            # 将图像缩放为28x28像素
            newimg = cv.resize(img, (28, 28), interpolation=cv.INTER_NEAREST)
            # 将新图像添加到列表中
            newimgs.append(newimg)
    # 返回裁剪后的图像列表
    return newimgs

