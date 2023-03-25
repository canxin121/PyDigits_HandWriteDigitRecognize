import os
from pathlib import Path
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from mlxtend.data import loadlocal_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import EarlyStopping
from PIL import Image
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 调用本项目中数据集
dataroot = Path.cwd() / 'data/'
train_images, train_labels = loadlocal_mnist(images_path=str(dataroot / 'train-images-idx3-ubyte'),
                                             labels_path=str(dataroot / 'train-labels-idx1-ubyte'))
test_images, test_labels = loadlocal_mnist(images_path=str(dataroot / 'test-images-idx3-ubyte'),
                                           labels_path=str(dataroot / 'test-labels-idx1-ubyte'))

# 将加载的本地数据集 转换成四维向量，并转换格式为float32并进行归一化
# 将二维的数据转化成思维，前三个维度表示图像的高度、宽度和通道数（例如RGB图像有3个通道），最后一个维度表示样本数量
x_train4D = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
x_test4D = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
# 归一化
x_train = x_train4D / 255  # 标准化
x_test = x_test4D / 255

# 对数据的标签进行单独热处理
y_train = np_utils.to_categorical(train_labels)
y_test = np_utils.to_categorical(test_labels)

# 乱序
np.random.seed(255)
np.random.shuffle(x_train)
np.random.seed(255)
np.random.shuffle(y_train)

# 构建CNN卷积神经网络模型
model = Sequential() # 创建一个顺序模型
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # 添加一个卷积层，使用32个3x3的过滤器，激活函数为relu，输入形状为28x28x1
model.add(MaxPooling2D((2, 2))) # 添加一个最大池化层，使用2x2的窗口
model.add(Conv2D(64, (3, 3), activation='relu')) # 添加另一个卷积层，使用64个3x3的过滤器，激活函数为relu
model.add(MaxPooling2D((2, 2))) # 添加另一个最大池化层，使用2x2的窗口
model.add(Conv2D(64, (3, 3), activation='relu')) # 添加最后一个卷积层，使用64个3x3的过滤器，激活函数为relu
model.add(Flatten()) # 添加一个展平层，将三维张量转换为一维向量
model.add(Dense(64, activation='relu')) # 添加一个全连接层，使用64个神经元，激活函数为relu
model.add(Dense(10, activation='softmax')) # 添加最后一个全连接层，使用10个神经元，激活函数为softmax，输出10个类别的概率

# 编译模型，指定优化器，损失函数和评估指标
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 创建一个图像数据生成器，用于对训练数据进行数据增强
datagen = ImageDataGenerator(
    rotation_range=10, # 随机旋转角度范围
    width_shift_range=0.1, # 随机水平平移范围
    height_shift_range=0.1, # 随机垂直平移范围
    zoom_range=0.1 # 随机缩放范围
)

# 创建一个早停回调函数，用于在验证损失不再下降时停止训练，并恢复最佳模型参数
early_stopping = EarlyStopping(
    monitor='accuracy', # 监控验证损失
    patience=20, # 容忍多少个轮次没有改善
    restore_best_weights=True)# 恢复最佳模型参数
# 训练模型，指定批量大小，迭代次数，验证数据集和回调函数
model.fit(datagen.flow(x_train, y_train, batch_size=128), # 使用数据生成器对训练数据进行数据增强
          epochs=500, # 迭代次数
          validation_data=(x_test, y_test), # 验证数据集
          callbacks=[early_stopping]) # 回调函数
model.save('mine.h5')
# 在测试数据集上评估模型的性能
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)