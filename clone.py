import csv
import cv2
import numpy as np
import os

lines = []
print('Start to read csv file...')
with open('.\DrivingData\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        # print(line)

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import sklearn


def preprocess(image):
    image = image[60:-25, :, :]
    image = cv2.resize(image, (200, 66), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image


def augument(image, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    if np.random.rand() < 0.1:
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))

    if np.random.rand() < 0.1:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image, steering_angle


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            #
            # images = []
            # angles = []
            # for batch_sample in batch_samples:
            #     name = './IMG/'+batch_sample[0].split('/')[-1]
            #     center_image = cv2.imread(name)
            #     center_angle = float(batch_sample[3])
            #     images.append(center_image)
            #     angles.append(center_angle)
            images = []
            measurements = []
            for line in batch_samples:
                # if i < img_limit:
                source_path = line[0]
                filename = source_path.split('/')[-1]
                current_path = filename
                # print('Image file is %s' %current_path)
                image = cv2.imread(current_path)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                measurement = float(line[3])
                image, measurement = augument(image, measurement)
                image = preprocess(image)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(image)
                images.append(image)

                measurements.append(measurement)

                # images.append(np.fliplr(image))
                # measurements.append(-measurement)
                # i+= 1
                # else:s
                #     break
                source_path = line[1]
                filename = source_path.split('/')[-1]
                current_path = filename
                # print('Image file is %s' %current_path)
                image = cv2.imread(current_path)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                measurement = float(line[3]) + 0.2
                image, measurement = augument(image, measurement)

                image = preprocess(image)
                # print(image)

                images.append(image)

                measurements.append(measurement)
                # images.append(np.fliplr(image))
                # measurements.append(-measurement - 0.25)

                source_path = line[2]
                filename = source_path.split('/')[-1]
                current_path = filename
                # print('Image file is %s' % current_path)
                image = cv2.imread(current_path)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                measurement = float(line[3]) - 0.2
                image, measurement = augument(image, measurement)
                image = preprocess(image)
                images.append(image)
                measurements.append(measurement)
                # images.append(np.fliplr(image))
                # measurements.append(-measurement + 0.25)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=50)
validation_generator = generator(validation_samples, batch_size=50)

ch, row, col = 3, 160, 320  # Trimmed image format

#
# images = []
# measurements = []
# print('Start to read images...')
# # img_limit = 1e2
# # i = 0
# for line in lines:
#     # if i < img_limit:
#     source_path = line[0]
#     filename = source_path.split('/')[-1]
#     current_path = filename
#     # print('Image file is %s' %current_path)
#     image = cv2.imread(current_path)
#     # print(image)
#     images.append(image)
#     measurement = float(line[3])
#     measurements.append(measurement)
#     images.append(np.fliplr(image))
#     measurements.append(-measurement)
#     # i+= 1
#     # else:s
#     #     break
#     source_path = line[1]
#     filename = source_path.split('/')[-1]
#     current_path = filename
#     # print('Image file is %s' %current_path)
#     image = cv2.imread(current_path)
#     # print(image)
#     images.append(image)
#     measurement = float(line[3]) + 0.25
#     measurements.append(measurement)
#     images.append(np.fliplr(image))
#     measurements.append(-measurement - 0.25)
#
#     source_path = line[2]
#     filename = source_path.split('/')[-1]
#     current_path = filename
#     print('Image file is %s' %current_path)
#     image = cv2.imread(current_path)
#     # print(image)
#     images.append(image)
#     measurement = float(line[3]) - 0.25
#     measurements.append(measurement)
#     images.append(np.fliplr(image))
#     measurements.append(-measurement + 0.25)
#
# X_train = np.array(images)
# print(X_train.shape)
# y_train = np.array(measurements)
# print('Start to import learning model')
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras.preprocessing import image
from keras import optimizers
print('Start to build learning model')
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
# model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam)

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print('Start the training...')
model.fit_generator(train_generator, samples_per_epoch=50000, validation_data=validation_generator,
                    nb_val_samples=10000, nb_epoch=50)

model.save('model_50Iter_3rdRun.h5')
