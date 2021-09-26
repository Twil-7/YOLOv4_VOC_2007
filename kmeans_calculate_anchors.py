import os
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

# k_means虽然会对数据集中的框进行聚类，但是很多数据集由于框的大小相近，聚类出来的9个框相差不大，这样的框反而不利于模型的训练。
# 因为不同的特征层适合不同大小的先验框，越浅的特征层适合越大的先验框。
# 原始网络的先验框已经按大中小比例分配好了，不进行聚类也会有非常好的效果。


class_dictionary = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
                    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
                    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
                    'sofa': 17, 'train': 18, 'tvmonitor': 19}
class_list = list(class_dictionary.keys())


def cas_iou(box, cluster):

    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):

    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def k_means(box, k):

    # 取出一共有多少框
    row = box.shape[0]
    
    # 每个框各个点的位置
    distance = np.empty((row, k))
    
    # 最后的聚类位置
    last_clu = np.zeros((row,))

    np.random.seed(10)

    # 随机选5个当聚类中心
    cluster = box[np.random.choice(row, k, replace=False)]

    while True:

        # 计算每一行距离9个点的iou情况
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)
        
        # 取出最小点
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break
        
        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(box[near == j], axis=0)

        last_clu = near

    return cluster


if __name__ == '__main__':

    SIZE = 416
    anchors_num = 9

    data = []
    filename = os.listdir('Annotations')
    filename.sort()

    for name in filename:

        obj1 = name.split('.')
        obj2 = obj1[0]
        img_path = 'JPEGImages/' + obj2 + '.jpg'

        img = Image.open(img_path)
        width = img.width
        height = img.height

        if height <= 0 or width <= 0:
            continue

        tree = ET.parse('Annotations/' + name)
        root = tree.getroot()

        for obj in root.iter('object'):

            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in class_list or int(difficult) == 1:
                continue

            cls_id = class_list.index(cls)
            xml_box = obj.find('bndbox')
            x_min = int(xml_box.find('xmin').text) / width
            y_min = int(xml_box.find('ymin').text) / height
            x_max = int(xml_box.find('xmax').text) / width
            y_max = int(xml_box.find('ymax').text) / height

            data.append([x_max - x_min, y_max - y_min])

    data = np.array(data)

    # 使用k聚类算法
    out = k_means(data, anchors_num)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    print('anchors :', out*SIZE)

    anchors = out*SIZE
    f = open("./model_data/yolo_anchors.txt", 'w')
    r = np.shape(anchors)[0]
    for i in range(r):
        if i == 0:
            x_y = "%d,%d" % (anchors[i][0], anchors[i][1])
        else:
            x_y = ", %d,%d" % (anchors[i][0], anchors[i][1])
        f.write(x_y)
    f.close()
