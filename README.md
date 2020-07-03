# VarGFaceNet

文档结构如下：

```bash
├── CASIA
│   ├── CASIA.7z
│   ├── CASIA-WebFace-112X96
│   └── CASIA-WebFace-112X96.txt
├── config.py
├── dataload
│   ├── CASIA_Face_loader.py
│   └── LFW_loader.py
├── eval_flw.py
├── LFW
│   ├── lfw-112X96
│   ├── lfw.7z
│   └── pairs.txt
├── model.py (VarGFaceNet模型文件)
├── __pycache__
│   └── config.cpython-37.pyc
├── README.md
├── run.sh
├── save_model
└── train.py

```


## 运行环境

* Python 3.8
* pytorch 1.3+
* GPU or CPU

## 使用方法

### 1、下载数据集

[Align-CASIA-WebFace@BaiduDrive](https://pan.baidu.com/s/1k3Cel2wSHQxHO9NkNi3rkg) and [Align-LFW@BaiduDrive](https://pan.baidu.com/s/1r6BQxzlFza8FM8Z8C_OCBg).

### 2、训练

改变**CAISIA_DATA_DIR** and **LFW_DATA_DAR** (在`config.py`文件中)
  
运行指令训练

```
python train_my.py
```
      
### 3、测试

在LFW数据集上测试
    
      
```
sh ./run.sh
```



## 参考资料

  * [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
  * [SphereFace](https://github.com/wy1iu/sphereface)
  * [Insightface](https://github.com/deepinsight/insightface)
  * [MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)
  
  
  
  
