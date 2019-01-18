项目名称：手写数字识别

[![](https://travis-ci.org/Alamofire/Alamofire.svg?branch=master)](https://travis-ci.org/Alamofire/Alamofire)![](https://img.shields.io/badge/python-2.7-orange.svg)

### 开发人员:

项目开发：[Xiang Jinhu](https://github.com/chirsxjh)，[Chen Wei](https://github.com/Cris0525)，[Fu Tongyong](https://github.com/CANYOUFINDIT)

项目指导：[Li Xipeng](https://github.com/hahaps)


### 需求概述
-----

该项目是MNIST数据集实现的手写数字图片识别工具,基于tensorflow的框架,通过建立卷积神经网络实现的模型结构,使用选择优化器进行模型训练,将需要好的变量进行保存,在识别手写数字图片的过程中使用原本已经保存好的数据,实现识别手写数字图片的效果。
通过opencv库来处理图片,进行灰度处理,二值化,降噪,以及剪裁图片,完成我们对图片要求的处理。
最终将项目进行封装,封装成库ImgProcess，提供简单易用的验证码识别API，从而方便调用。

### 项目运行环境及技术相关

----

语言：`python2.7`

运行环境:  `ubuntu16.04`

深度学习框架:  `Tensorflow1.4.0`

图片处理技术:  `python-opencv3.4.0`

代码封装:  `setuptools(Python2.7.9后自带setuptools无须另外安装)`

图片显示: `Image`

### 项目功能描述

-----

> * 识别手写数字图片
>
> 下载该文件,通过安装ImgProcess库,只需调用该模块,输入手写数字图片的地址信息,即可识别该图片

### 安装和卸载

-----------

将项目克隆

`sudo python setup.py install`

使用pip卸载

`sudo pip uninstall ImgProcess`

### API使用

---


```python
from ImgProcess import process_img
result = process_img('./img/red4.jpg')
print(result)
```



### 注意事项

----

训练集使用MNIST手写数字的图片，介于东西方手写数字风格可能存在差异，模型在数字5和6上可能会识别错误，该问题在识别机器生成的图片上同样存在

### 相关技术

----

> 1.建立卷积神经网络层并进行训练模型

 * 导入MNIST数据集,定义变量x和y,并进行设置形状,控制图片大小为28*28

   ```python
   mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)  
     
   sess = tf.InteractiveSession()  
     
   #训练数据  
   x = tf.placeholder("float", shape=[None, 784])  
   #训练标签数据  
   y_ = tf.placeholder("float", shape=[None, 10])  
   #把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白  
   x_image = tf.reshape(x, [-1,28,28,1])  
   ```

 * 第一层:开始建立卷积层,定义Weights和Biases,建立过滤器大小为5*5,当前深度为1,过滤器深度为32,并建立卷积函数,再通过relu进行一次激活函数,其作用是去线性化

   ```python
   #第一层：卷积层  
   conv1_weights = tf.get_variable("conv1_weights", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1)) #过滤器大小为5*5, 当前层深度为1， 过滤器的深度为32  
   conv1_biases = tf.get_variable("conv1_biases", [32], initializer=tf.constant_initializer(0.0))  
   conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME') #移动步长为1, 使用全0填充  
   relu1 = tf.nn.relu( tf.nn.bias_add(conv1, conv1_biases) ) #激励函数Relu去线性化  
   ```
   ![](http://imgsrc.baidu.com/forum/pic/item/5164b9014a90f6038a3b8cba3412b31bb251edaf.jpg)



   提取特征值

 * 第二层:建立最大池化层

   ```python
   #第二层：最大池化层  
   #池化层过滤器的大小为2*2, 移动步长为2，使用全0填充  
   pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
   ```

   ![](http://imgsrc.baidu.com/forum/pic/item/ce2836d3d539b60075581f4ae450352ac45cb79b.jpg)

   max_pool的作用:

   1. invariance(不变性)，这种不变性包括translation(平移)，rotation(旋转)，scale(尺度) 

   2. 保留主要的特征同时减少参数(降维，效果类似PCA)和计算量，防止过拟合，提高模型泛化能力

 * 第三层;再一次建立卷积层进行收集数据集

   ```python
   #第三层：卷积层  
   conv2_weights = tf.get_variable("conv2_weights", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1)) #过滤器大小为5*5, 当前层深度为32， 过滤器的深度为64  
   conv2_biases = tf.get_variable("conv2_biases", [64], initializer=tf.constant_initializer(0.0))  
   conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') #移动步长为1, 使用全0填充  
   relu2 = tf.nn.relu( tf.nn.bias_add(conv2, conv2_biases) )  
   ```

 * 第四层:再次建立最大池化层

   ```python
   #第四层：最大池化层  
   #池化层过滤器的大小为2*2, 移动步长为2，使用全0填充  
   pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
   ```

 * Dropout层处理

   ```python
   #为了减少过拟合，加入Dropout层  
   keep_prob = tf.placeholder(tf.float32)  
   fc1_dropout = tf.nn.dropout(fc1, keep_prob)  
   ```

 * 第五层:建立全连接层,把前一层的输出变成特征向量

   ```python
   #第五层：全连接层  
   fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1)) #7*7*64=3136把前一层的输出变成特征向量  
   fc1_baises = tf.get_variable("fc1_baises", [1024], initializer=tf.constant_initializer(0.1))  
   pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])  
   fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises) 
   ```

 * 第六层:搭建全连接层

   ```python
   #第六层：全连接层  
   fc2_weights = tf.get_variable("fc2_weights", [1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.1)) #神经元节点数1024, 分类节点10  
   fc2_biases = tf.get_variable("fc2_biases", [10], initializer=tf.constant_initializer(0.1))  
   fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases  
   ```

 * 第七层:输出层,对网络最后一层的输出做一个softmax,这一步通常是求取输出属于某一类的概率,reduce_mean函数用来降低损失值

   ```python
   #第七层：输出层  
   # softmax  
   y_conv = tf.nn.softmax(fc2)  
     
   #定义交叉熵损失函数  
   cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) 
   ```

 * 再使用选择优化器进行梯度计算并进行训练

   ```python
   #选择优化器，并让优化器最小化损失函数/收敛, 反向传播  
   train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
     
   # tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真实值  
   # 判断预测值y和真实值y_中最大数的索引是否一致，y的值为1-10概率  
   correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  
     
   # 用平均值来统计测试准确率  
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
   ```

 * 将训练好的数据集存储在文件中

   ```python
   saver_path = saver.save(sess, "/home/cris/AI/MNIST/save/model.ckpt")  # 将模型保存到save/model.ckpt文件
   print("Model saved in file:", saver_path)
   ```

------

>  2.安装opencv第三方库,依次将图片进行灰度处理,降噪,二值化,最后进行剪裁图片,在建立神经网络模型的时候我们已控制图片的大小为28*28,通过opencv的剪裁功能进行控制图片的大小,,并进行保存
* 原图片:

![](http://imgsrc.baidu.com/forum/pic/item/5a2d39dbb6fd52666fd33fe9a618972bd60736bd.jpg)



```python
#灰度处理
def cvt_Color(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

# 中值滤波,降噪
def median_Blur(img, m=3):
    blur = cv2.medianBlur(img, m) #模板大小3*3
    return blur

# 二值化处理
def thresh_old(img, l=100, h=255):
    ret,im_fixed=cv2.threshold(img,l,h,cv2.THRESH_BINARY)
    return im_fixed

```

进行剪裁图片,并通过opencv库的findContours()函数自动检测物体的轮廓

```python
# 裁剪字符
def cut(img):
    # 查找检测物体的轮廓
    image, contours, hierarchy = cv2.findContours(img, 2, 2)
    #cv2.imshow('image',image)
    # 计数器
    flag = 1
    for cnt in contours:
        # 最小的外接矩形, cnt是一个二值图, x,y是矩阵左上点的坐标, w,h是矩阵的宽和高
        x, y, w, h = cv2.boundingRect(cnt)
        if x != 0 and y != 0 and w*h >= 500:
            #print((x,y,w,h))
            # 调整图像尺寸为28*28
            resize_img = cv2.resize(img[y:y+h, x:x+w], (28,28))
            # 再次二值化
            im_fixed = thresh_old(resize_img, 200, 255)
            # 显示图片
            new_img = get_Side(im_fixed)
            image = Image.fromarray(new_img)
            flag += 1
    return image
```

处理后:

![](http://imgsrc.baidu.com/forum/pic/item/77492edda3cc7cd9d78c39243401213fba0e9194.jpg)

-------

> 3.调用第一步训练好的模型,再通过Image库打开剪裁后的图片,并建立卷积神经网络层,进行手写数字识别功能

```python
file_name='/home/cris/AI/MNIST/image/hei8.png'#导入自己的图片地址
    #进行灰度处理
    im = Image.open(file_name).convert('L')
    #将图片进行剪裁成28*28像素大小
    img = im.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
```

```python
#导入训练好的模型
sess = tf.InteractiveSession()
saver = tf.train.Saver()
tf.global_variables_initializer().run()
saver.restore(sess, "/home/cris/AI/MNIST/save/model.ckpt")
```

>  4.将完成的代码进行打包封装成模块,提供简单易用的验证码识别API，从而方便调用。

      setup.py

```python
from setuptools import setup, find_packages

setup(
    name = 'ImgProcess',
    version = '0.0.1',
    author = 'Fly',
    packages = find_packages(),
    install_requires = ['numpy', 'tensorflow', 'matplotlib', 'opencv-python', 'Pillow'],
    package_data = {'ImgProcess': ['model/*']},
    include_package_data = True,
)
```



### 项目目录

-----

```tree
├── getNumber
│   ├── build
│   │   └── lib.linux-x86_64-2.7
│   │       └── ImgProcess
│   │           ├── __init__.py
│   │           └── model
│   │               ├── checkpoint
│   │               ├── model.ckpt.data-00000-of-00001
│   │               ├── model.ckpt.index
│   │               └── model.ckpt.meta
│   ├── dist
│   │   └── ImgProcess-0.0.1-py2.7.egg
│   ├── image
│   │   ├── char.png
│   │   ├── gary.png
│   │   ├── img.png
│   │   ├── resize.png
│   │   └── thresh.png
│   ├── ImgProcess
│   │   ├── img
│   │   ├── __init__.py
│   │   └── model
│   │       ├── checkpoint
│   │       ├── model.ckpt.data-00000-of-00001
│   │       ├── model.ckpt.index
│   │       └── model.ckpt.meta
│   ├── ImgProcess.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── requires.txt
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── README.md
│   └── setup.py
└── README.md

```

### 更新链接

[github](https://github.com/WeAreHus/NumRecognition)

### 运行效果展示

![](http://imgsrc.baidu.com/forum/pic/item/09fd0ef41bd5ad6e67b164178ccb39dbb4fd3ced.jpg)

![](http://imgsrc.baidu.com/forum/pic/item/45b70fb30f2442a71234fe58dc43ad4bd013027f.jpg)





