# 基于MNIST数据集实现的手写数字图片识别工具

## 概述

基于tensorflow建立的CNN模型，使用MNIST手写数字数据集训练的数字识别工具，封装成库ImgProcess，提供简单易用的验证码识别API，从而方便调用。

## 安装和卸载

将项目克隆后

`sudo python setup.py install`即可

使用pip卸载

`sudo pip uninstall ImgProcess`

## demo

```python
from ImgProcess import process_img
result = process_img('./img/red4.jpg')
print result
```

## 注意事项

训练集使用MNIST手写数字的图片，介于东西方手写数字风格可能存在差异，模型在数字5和6上可能会识别错误，该问题在识别机器生成的图片上同样存在

## 代码实现

该工具使用MNIST 28*28像素数字图片数据集训练，在识别数字前使用opencv对图片进行处理，使之符合数据集特征方便识别。

### 使用opencv对图片预处理

- 将图片处理成灰度图
  ```python
  def cvt_Color(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
  ```

- 如果图片有边框则去除边框，255表示白色，0表示黑色
  ```python
  # 去边框/加边框
  def get_Side(img):
    # 将图片的边缘变为白色
    height, width = img.shape
    for i in range(width):
        img[0, i] = 255
        img[height-1, i] = 255
    for j in range(height):
        img[j, 0] = 255
        img[j, width-1] = 255
    return img
    ```

- 对图片进行滤波降噪去除噪点
  ```python
  # 中值滤波,降噪
  def median_Blur(img, m=3):
    blur = cv2.medianBlur(img, m) #模板大小3*3
    return blur
    ```

- 二值化处理，将图片转化成黑白图片
  ```python
  # 二值化处理
  def thresh_old(img, l=100, h=255):
    ret,im_fixed=cv2.threshold(img,l,h,cv2.THRESH_BINARY)
    return im_fixed
    ```

- 裁剪字符，将图片空白部分去除，并将图片压缩为28*28像素
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
        # 设置图片像素块最小大小为500
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
    该函数可以将一张图片中的多个数字分隔开，即满足多位数字的验证码图片的处理

- 处理图片的流程
  ```python
      #读入原始图像
    img=cv2.imread(filedir)
    # 灰度图
    gary = cvt_Color(img)
    # 降噪
    blur = median_Blur(gary)
    # 二值化
    thresh = thresh_old(blur)
    # 裁剪
    image = cut(thresh)
    ```

    **处理效果：**

    ![原图](image/img.png "原图")
    ![灰度图](image/gary.png "灰度图")
    ![二值化](image/thresh.png "二值化")
    ![裁剪](image/char.png "裁剪")
    ![压缩图](image/resize.png "压缩")
    
    可以根据实际情况调整处理顺序，对于特殊的图片可能需要对应地调整参数以达到最佳效果

### CNN模型的搭建

**卷积神经网络层次原理图：**
![](https://cdn-pri.nlark.com/yuque/0/2018/png/176236/1543911746802-04495b23-0596-483e-92fa-32de356433c2.png)

1. 导入MNIST数据集,定义变量x和y,并进行设置形状,控制图片大小为28*28
   ```python
   mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True) sess = tf.InteractiveSession()  
   #训练数据  
   x = tf.placeholder("float", shape=[None, 784])  
   #训练标签数据  
   y_ = tf.placeholder("float", shape=[None, 10])  
   #把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白  
   x_image = tf.reshape(x, [-1,28,28,1])  
   ```

2. 第一层:开始建立卷积层,定义Weights和Biases,建立过滤器大小为5*5,当前深度为1,过滤器深度为32,并建立卷积函数,再通过relu进行一次激活函数,其作用是去线性化
   ```python
   #第一层：卷积层  
   conv1_weights = tf.get_variable("conv1_weights", [5, 5, 1, 32]initializer=tf.truncated_normal_initializer(stddev=0.1)) #过滤器大小为5*5, 当前层深度为1， 过滤器的深度为32  
   conv1_biases = tf.get_variable("conv1_biases", [32], initializer=tf.constant_initializer(0.0))  
   conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME') #移动步长为1, 使用全0填充  
   relu1 = tf.nn.relu( tf.nn.bias_add(conv1, conv1_biases) ) #激励函数Relu去线性化  
   ```
   ![提取特征值](https://cdn-pri.nlark.com/yuque/0/2018/png/176236/1543911661122-b8756f48-abf2-4203-ae8e-b02309aac848.png)

3. 第二层:建立最大池化层
   ```python
   #第二层：最大池化层  
   #池化层过滤器的大小为2*2, 移动步长为2，使用全0填充  
   pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
   ```
   ![](https://cdn-pri.nlark.com/yuque/0/2018/jpeg/176236/1543911677998-66c5f44f-67f4-41a1-80a3-037ee611c8ab.jpeg)
   max_pool的作用:
   1. invariance(不变性)，这种不变性包括translation(平移)，rotation(旋转)，scale(尺度)
   2. 保留主要的特征同时减少参数(降维，效果类似PCA)和计算量，防止过拟合，提高模型泛化能力

4. 第三层:再一次建立卷积层进行收集数据集
   ```python
   #第三层：卷积层  
   conv2_weights = tf.get_variable("conv2_weights", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1)) #过滤器大小为5*5, 当前层深度为32， 过滤器的深度为64  
   conv2_biases = tf.get_variable("conv2_biases", [64], initializer=tf.constant_initializer(0.0))  
   conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') #移动步长为1, 使用全0填充  
   relu2 = tf.nn.relu( tf.nn.bias_add(conv2, conv2_biases) )  
   ```

5. 第四层:再次建立最大池化层
   ```python
   #第四层：最大池化层  
   #池化层过滤器的大小为2*2, 移动步长为2，使用全0填充  
   pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
   ```

   Dropout层处理

   ```python
   #为了减少过拟合，加入Dropout层  
   keep_prob = tf.placeholder(tf.float32)  
   fc1_dropout = tf.nn.dropout(fc1, keep_prob)  
   ```

6. 第五层:建立全连接层,把前一层的输出变成特征向量
   ```python
   #第五层：全连接层  
   fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1)) #7*7*64=3136把前一层的输出变成特征向量  
   fc1_baises = tf.get_variable("fc1_baises", [1024], initializer=tf.constant_initializer(0.1))  
   pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])  
   fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises) 
   ```

7. 第六层:搭建全连接层
   ```python
   #第六层：全连接层  
   fc2_weights = tf.get_variable("fc2_weights", [1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.1)) #神经元节点数1024, 分类节点10  
   fc2_biases = tf.get_variable("fc2_biases", [10], initializer=tf.constant_initializer(0.1))  
   fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases  
   ```

8. 第七层:输出层,对网络最后一层的输出做一个softmax,这一步通常是求取输出属于某一类的概率,reduce_mean函数用来降低损失值
   ```python
   #第七层：输出层  
   # softmax  
   y_conv = tf.nn.softmax(fc2)  
   #定义交叉熵损失函数  
   cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) 
   ```

9. 导入训练模型
   ```python
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    # 找到文件父级目录
    model_path = os.path.dirname(os.path.abspath(__file__)) + "/model/model.ckpt"
    # print model_path
    saver.restore(sess, model_path)
    prediction=tf.argmax(y_conv,1)
    predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
    ```
    最终的识别结果保存在predint[0]中。
