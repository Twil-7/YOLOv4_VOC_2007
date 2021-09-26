# YOLOv4_VOC_2007

# 环境配置：

python == 3.6

tensorflow == 2.3.0

opencv == 3.4.2


# 文件说明：

（1）Annotations、JEPGImages文件夹：存放VOC2007数据集。

下载路径：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

（2）Logs文件夹：存放训练过程中的权重文件。

（3）model_data：存放划分的数据集path、原始YOLOv4权重、anchors系数、类别信息。

（4）nets文件夹：存放YOLOv4模型.py文件。

（5）utils文件夹：存放YOLOv4辅助功能.py文件。

（6）权重文件：

yolo4_weight.h5：原始载入的YOLOv4权重文件。

best_ep036-loss2.248-val_loss2.266.h5：自己训练的权重文件。

下载路径：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

以下4个py文件需要单独运行：

第1步：运行annotation.py文件：将VOC数据集进行划分，生成.txt路径文件存储到model_data文件夹中。

第2步：运行k_means_calculate.py文件：计算生成anchors数值，存储到model_data文件夹中。

第3步：运行train.py文件：加载原始权重，训练YOLOv4模型，并将每轮训练的结果存储进Logs文件夹中。

第4步：运行yolo_predict.py文件：载入训练好的YOLOv4权重，对测试集数据进行检测，检测结果存放入demo文件夹中。


# 算法效果：

YOLOv4每张图片检测耗时373.5ms，精度较高，训练30 epoch左右后val loss降低至2.3附近，比较满意。

相比YOLOv1和YOLOv3，YOLOv4训练起来更为轻松，val loss能轻易降到比较低的程度，权重的检测效果较佳。
