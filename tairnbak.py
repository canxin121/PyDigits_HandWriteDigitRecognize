# 导入Keras和其他必要的库
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator # 导入图像数据生成器
from keras.callbacks import EarlyStopping # 导入早停回调函数
import numpy as np # 导入numpy库

# 加载MNIST数据集并进行筛选，只保留标签为1或7的样本
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[(y_train == 1) | (y_train == 7)]
y_train = y_train[(y_train == 1) | (y_train == 7)]
x_test = x_test[(y_test == 1) | (y_test == 7)]
y_test = y_test[(y_test == 1) | (y_test == 7)]

# 对标签进行二值化，将1映射为0，将7映射为1
y_train = np.where(y_train == 1, 0, 1)
y_test = np.where(y_test == 1, 0, 1)

# 对数据进行预处理和one-hot编码
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((len(x_test), 28, 28, 1))
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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

# 加载已经训练好的模型参数
model.load_weights('best.h5')

# 删除最后一个全连接层
model.pop()

# 添加一个新的全连接层，使用2个神经元，激活函数为softmax，输出2个类别的概率
model.add(Dense(2, activation='softmax'))

# 添加一个新的全连接层，使用2个神经元，激活函数为softmax，输出2个类别的概率
model.add(Dense(2, activation='softmax'))

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
    monitor='val_loss', # 监控验证损失
    patience=10, # 容忍多少个轮次没有改善
    restore_best_weights=True # 恢复最佳模型参数
)

# 继续训练模型，指定批量大小，迭代次数，验证数据集和回调函数
model.fit(datagen.flow(x_train, y_train, batch_size=128), # 使用数据生成器对训练数据进行数据增强
          epochs=50, # 迭代次数
          validation_data=(x_test, y_test), # 验证数据集
          callbacks=[early_stopping]) # 回调函数

# 在测试数据集上评估模型的性能
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)