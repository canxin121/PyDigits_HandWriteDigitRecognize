# PyDigits_HandWriteDigitRecognize

本项目包括  
1.mnist数据集的创建：首先将各个数字的照片分好类放在old中，通过prepic.py可以将其处理成可以用于转化为ubyte类型数据集的黑底白字28*28图  
进而分配为训练集train和测试集test两部分，然后通过pictoubyte.py转化为mnist同格式的训练集和测试集。  
2.recognize.py可以提供一个tk窗口来进行识别，注意先输入行数（默认1），此py依赖more.py进行数字分割。   
3.另有nonebot2插件接入cqhttp，马上发布。  

![image](https://github.com/canxin121/PyDigits_HandWriteDigitRecognize/blob/main/envdav/show%20(1).png)  
![image](https://github.com/canxin121/PyDigits_HandWriteDigitRecognize/blob/main/envdav/show%20(3).png)  
![image](https://github.com/canxin121/PyDigits_HandWriteDigitRecognize/blob/main/envdav/show%20(2).png)  
