# coding:utf-8

import copy
import sys
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter


#灰度化处理
def cvt_Color(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img


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


# 中值滤波,降噪
def median_Blur(img, m=3):
    blur = cv2.medianBlur(img, m) #模板大小3*3
    return blur


# 二值化处理
def thresh_old(img, l=100, h=255):
    ret,im_fixed=cv2.threshold(img,l,h,cv2.THRESH_BINARY)
    #im_fixed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
    return im_fixed


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


def image_prepare(img):
    #导入自己的图片地址
    #file_name='img/red6.jpg'
    #file_name = sys.argv[1]
    #img_process.process_img(file_name)
    #img = Image.open('out_img/resize1.png')
    data = list(img.getdata())
    result = [ (255-x)*1.0/255.0 for x in data] 
    x = tf.placeholder("float", shape=[None, 784])  
    #训练标签数据  
    y_ = tf.placeholder("float", shape=[None, 10])  
    x_image = tf.reshape(x, [-1,28,28,1])  

    conv1_weights = tf.get_variable("conv1_weights", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
    conv1_biases = tf.get_variable("conv1_biases", [32], initializer=tf.constant_initializer(0.0))  
    conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu( tf.nn.bias_add(conv1, conv1_biases) )    
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    
    conv2_weights = tf.get_variable("conv2_weights", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1)) #过滤器大小为5*5, 当前层深度为32， 过滤器的深度为64  
    conv2_biases = tf.get_variable("conv2_biases", [64], initializer=tf.constant_initializer(0.0))  
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') #移动步长为1, 使用全0填充  
    relu2 = tf.nn.relu( tf.nn.bias_add(conv2, conv2_biases) )  

    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    
    fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1)) #7*7*64=3136把前一层的输出变成特征向量  
    fc1_baises = tf.get_variable("fc1_baises", [1024], initializer=tf.constant_initializer(0.1))  
    pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])  
    fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises)  
    
    #为了减少过拟合，加入Dropout层  
    keep_prob = tf.placeholder(tf.float32)  
    fc1_dropout = tf.nn.dropout(fc1, keep_prob)  
    
    #第六层：全连接层  
    fc2_weights = tf.get_variable("fc2_weights", [1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.1)) #神经元节点数1024, 分类节点10  
    fc2_biases = tf.get_variable("fc2_biases", [10], initializer=tf.constant_initializer(0.1))  
    fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases  
    
    #第七层：输出层  
    # softmax  
    y_conv = tf.nn.softmax(fc2)  
    
    #定义交叉熵损失函数  
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))  
    
    #选择优化器，并让优化器最小化损失函数/收敛, 反向传播  
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
    
    # tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真实值  
    # 判断预测值y和真实值y_中最大数的索引是否一致，y的值为1-10概率  
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  
    
    # 用平均值来统计测试准确率  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    
    
    #导入训练好的模型
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    # 找到文件父级目录
    model_path = os.path.dirname(os.path.abspath(__file__)) + "/model/model.ckpt"
    # print model_path
    saver.restore(sess, model_path)
    #print("W1:", sess.run(conv1_weights)) # 打印v1、v2的值一会读取之后对比
    #print("W2:", sess.run(conv1_biases))
    prediction=tf.argmax(y_conv,1)
    predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)

    #print('recognize result:')
    #print(predint[0])
    return predint[0]


def process_img(filedir):
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
    result = image_prepare(image)
    return result


if __name__ == "__main__":
    print process_img('ImgProcess/img/white8.jpg')
