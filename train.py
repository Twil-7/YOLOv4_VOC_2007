from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from nets.loss import yolo_loss
from nets.yolov4_model import yolo_body
from data_generate import data_generator


def get_classes(path):

    with open(path) as f:
        cls_names = f.readlines()
    cls_names = [c.strip() for c in cls_names]
    return cls_names


def get_anchors(path):

    with open(path) as f:
        anchor = f.readline()
    anchor = [float(x) for x in anchor.split(',')]

    return np.array(anchor).reshape(-1, 2)


if __name__ == "__main__":

    train_path = 'model_data/2021_train.txt'
    val_path = 'model_data/2021_val.txt'
    test_path = 'model_data/2021_test.txt'

    log_dir = 'Logs/'
    classes_path = 'model_data/new_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'

    weights_path = 'model_data/yolo4_weight.h5'
    label_smoothing = 0
    input_shape = (416, 416)

    class_names = get_classes(classes_path)    # ['car']
    anchors = get_anchors(anchors_path)
    # [[114.  53.]
    #  [139.  64.]
    #  [147.  79.]
    #  [164.  71.]
    #  [170.  95.]
    #  [189.  82.]
    #  [197. 108.]
    #  [221.  94.]
    #  [243. 119.]]

    num_classes = len(class_names)    # 1
    num_anchors = len(anchors)        # 9

    image_input = Input(shape=(416, 416, 3))
    h, w = input_shape

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    model_body.summary()
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    #   搭建损失函数层，将网络的输出结果传入loss函数，把整个模型的输出作为loss
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    loss_input = [*model_body.output, *y_true]

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes,
                                   'ignore_thresh': 0.5, 'label_smoothing': label_smoothing})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)

    checkpoint = ModelCheckpoint(log_dir + "/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 save_weights_only=True, save_best_only=False, period=1)

    with open(train_path) as f:
        train_lines = f.readlines()

    with open(val_path) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)    # 2900
    num_val = len(val_lines)        # 100

    freeze_layers = 249
    for i in range(freeze_layers):
        model_body.layers[i].trainable = False

    Init_epoch = 3
    Freeze_epoch = 250
    batch_size = 5
    learning_rate_base = 1e-3

    epoch_size_train = num_train // batch_size    # 580
    epoch_size_val = num_val // batch_size        # 20

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pre: y_pre})

    model.fit(data_generator(train_lines, batch_size, input_shape, anchors, num_classes, random=True),
              steps_per_epoch=epoch_size_train,
              validation_data=data_generator(val_lines, batch_size, input_shape, anchors, num_classes, random=False),
              validation_steps=epoch_size_val,
              epochs=Freeze_epoch,
              initial_epoch=Init_epoch,
              callbacks=[checkpoint, reduce_lr])

    # Epoch 1/250
    # 1600/1600 [==============================] - 4970s 3s/step - loss: 36.166 - val_loss: 6.7241
    # Epoch 5/250
    # 1600/1600 [==============================] - 4952s 3s/step - loss: 6.2326 - val_loss: 5.1956
    # Epoch 6/250
    # 1600/1600 [==============================] - 4906s 3s/step - loss: 5.3712 - val_loss: 4.7031
    # Epoch 4/250
    # 1600/1600 [==============================] - 4970s 3s/step - loss: 5.0922 - val_loss: 4.4992
    # Epoch 5/250
    # 1600/1600 [==============================] - 4952s 3s/step - loss: 4.5979 - val_loss: 3.9611
    # Epoch 6/250
    # 1600/1600 [==============================] - 4906s 3s/step - loss: 4.3305 - val_loss: 3.8423
    # Epoch 7/250
    # 1600/1600 [==============================] - 4973s 3s/step - loss: 4.1514 - val_loss: 3.4939
    # Epoch 8/250
    # 1600/1600 [==============================] - 4943s 3s/step - loss: 3.9295 - val_loss: 3.2480
    # Epoch 9/250
    # 1600/1600 [==============================] - 4867s 3s/step - loss: 3.8196 - val_loss: 3.2813
    # Epoch 10/250
    # 1600/1600 [==============================] - 4880s 3s/step - loss: 3.7062 - val_loss: 3.2720
    # Epoch 11/250
    # 1600/1600 [==============================] - 4869s 3s/step - loss: 3.5978 - val_loss: 3.2084
    # Epoch 12/250
    # 1600/1600 [==============================] - 4884s 3s/step - loss: 3.5349 - val_loss: 3.2265
    # Epoch 13/250
    # 1600/1600 [==============================] - 4895s 3s/step - loss: 3.4705 - val_loss: 2.9859
    # Epoch 14/250
    # 1600/1600 [==============================] - 4891s 3s/step - loss: 3.3872 - val_loss: 3.0937
    # Epoch 15/250
    # 1600/1600 [==============================] - 4866s 3s/step - loss: 3.3051 - val_loss: 2.9948
    # Epoch 16/250
    # 1600/1600 [==============================] - 4861s 3s/step - loss: 3.2831 - val_loss: 2.9360
    # Epoch 17/250
    # 1600/1600 [==============================] - 4868s 3s/step - loss: 3.1858 - val_loss: 3.0259
    # Epoch 18/250
    # 1600/1600 [==============================] - 4952s 3s/step - loss: 3.1701 - val_loss: 2.9079
    # Epoch 19/250
    # 1600/1600 [==============================] - 4963s 3s/step - loss: 3.0896 - val_loss: 2.8858
    # Epoch 20/250
    # 1600/1600 [==============================] - 4903s 3s/step - loss: 3.1119 - val_loss: 2.7995
    # Epoch 21/250
    # 1600/1600 [==============================] - 5116s 3s/step - loss: 3.0445 - val_loss: 2.7876
    # Epoch 22/250
    # 1600/1600 [==============================] - 5237s 3s/step - loss: 2.9627 - val_loss: 2.8871
    # Epoch 23/250
    # 1600/1600 [==============================] - 4938s 3s/step - loss: 2.9607 - val_loss: 2.7918
    # Epoch 24/250
    # 1600/1600 [==============================] - ETA: 0s - loss: 2.9712
    # Epoch 00024: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
    # 1600/1600 [==============================] - 4946s 3s/step - loss: 2.9712 - val_loss: 2.8582
    # Epoch 25/250
    # 1600/1600 [==============================] - 4973s 3s/step - loss: 2.6839 - val_loss: 2.4534
    # Epoch 26/250
    # 1600/1600 [==============================] - 5301s 3s/step - loss: 2.5865 - val_loss: 2.5393
    # Epoch 27/250
    # 1600/1600 [==============================] - 4844s 3s/step - loss: 2.5873 - val_loss: 2.4939
    # Epoch 28/250
    # 1600/1600 [==============================] - 4853s 3s/step - loss: 2.5283 - val_loss: 2.4220
    # Epoch 29/250
    # 1600/1600 [==============================] - 4852s 3s/step - loss: 2.5004 - val_loss: 2.3658
    # Epoch 30/250
    # 1600/1600 [==============================] - 4855s 3s/step - loss: 2.5108 - val_loss: 2.4580
    # Epoch 31/250
    # 1600/1600 [==============================] - 4849s 3s/step - loss: 2.4780 - val_loss: 2.3377
    # Epoch 32/250
    # 1600/1600 [==============================] - 4851s 3s/step - loss: 2.4481 - val_loss: 2.3765
    # Epoch 33/250
    # 1600/1600 [==============================] - 4856s 3s/step - loss: 2.4441 - val_loss: 2.4612
    # Epoch 34/250
    # 1600/1600 [==============================] - ETA: 0s - loss: 2.4383
    # Epoch 00034: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
    # 1600/1600 [==============================] - 4844s 3s/step - loss: 2.4383 - val_loss: 2.3839
    # Epoch 35/250
    # 1600/1600 [==============================] - 4848s 3s/step - loss: 2.3406 - val_loss: 2.3051
    # Epoch 36/250
    # 1600/1600 [==============================] - 4843s 3s/step - loss: 2.2477 - val_loss: 2.2659
    # Epoch 37/250
    # 1600/1600 [==============================] - 4840s 3s/step - loss: 2.2216 - val_loss: 2.3333
    # Epoch 38/250
    # 1600/1600 [==============================] - 4843s 3s/step - loss: 2.2568 - val_loss: 2.3369
