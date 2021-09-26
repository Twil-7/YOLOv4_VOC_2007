import os
import random
import numpy as np
import xml.etree.ElementTree as ET


class_dictionary = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
                    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
                    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
                    'sofa': 17, 'train': 18, 'tvmonitor': 19}
class_list = list(class_dictionary.keys())


def write_each_txt(file_list, txt):

    file_txt = open("./model_data/" + txt, "w")

    for name in file_list:
        file_txt.write(name + "\n")

    file_txt.close()


def write_txt():

    filename = os.listdir('Annotations')
    filename.sort()
    annotation_list = []

    for name in filename:

        obj1 = name.split('.')
        obj2 = obj1[0]
        obj3 = 'JPEGImages/' + obj2 + '.jpg' + ' '

        annotation = obj3

        tree = ET.parse('Annotations/' + name)
        root = tree.getroot()

        for obj in root.iter('object'):

            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in class_list or int(difficult) == 1:
                continue

            cls_id = class_list.index(cls)
            xml_box = obj.find('bndbox')
            x_min = int(xml_box.find('xmin').text)
            y_min = int(xml_box.find('ymin').text)
            x_max = int(xml_box.find('xmax').text)
            y_max = int(xml_box.find('ymax').text)

            loc = (str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max) + ',' + str(cls_id) + ' ')

            annotation = annotation + loc

        annotation_list.append(annotation)

    index = list(np.arange(0, len(annotation_list), 1))
    random.seed(10)
    random.shuffle(index)
    annotation_list = [annotation_list[k] for k in index]

    train_file = annotation_list[:8000]
    val_file = annotation_list[8000:9000]
    test_file = annotation_list[9000:]

    write_each_txt(train_file, '2021_train.txt')
    write_each_txt(val_file, '2021_val.txt')
    write_each_txt(test_file, '2021_test.txt')


write_txt()
