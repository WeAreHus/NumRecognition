# 基于MNIST数据集实现的手写数字图片识别工具

### 概述

基于tensorflow建立的CNN模型，使用MNIST手写数字数据集训练的数字识别工具，封装成库ImgProcess，提供简单易用的验证码识别API，从而方便调用。

### 安装和卸载

将项目克隆后

`sudo python setup.py install`即可

使用pip卸载

`sudo pip uninstall ImgProcess`

### 例子

```python
from ImgProcess import process_img
result = process_img('./img/red4.jpg')
print result
```

### 注意事项

训练集使用MNIST手写数字的图片，介于东西方手写数字风格可能存在差异，模型在数字5和6上可能会识别错误，该问题在识别机器生成的图片上同样存在