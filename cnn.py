import numpy as np
import json
import keras
import keras.optimizers
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn import preprocessing

# read xList
f = open("xList_1.txt")
xList = []
for line in f.readlines():
    xList.append(json.loads(line))
    for i in range(60):
        xList[-1][i][4] = xList[-1][i][4] / 100000000.0
        xList[-1][i].append((xList[-1][i][3] - xList[-1][i][0])/(xList[-1][i][0]+1e-6))
        xList[-1][i].append((xList[-1][i][1] - xList[-1][i][2]) / (xList[-1][i][2]+1e-6) )
f.close()

# read yList
f = open("yList_1.txt")
yList = []
for line in f.readlines():
    yList.append(json.loads(line))
f.close()

z = list(zip(xList, yList))
random.shuffle(z)
xList[:], yList[:] = zip(*z)

alpha = 0.75
trainNum = int(len(xList) * alpha)

xListTrain = xList[0:trainNum]
yListTrain = yList[0:trainNum]
xListTest = xList[trainNum + 1:]
yListTest = yList[trainNum + 1:]

x_train = np.array(xListTrain).reshape((len(xListTrain), 420))
x_train = preprocessing.scale(x_train).reshape((len(xListTrain), 60, 7, 1))
x_test = np.array(xListTest).reshape((len(xListTest), 420))
x_test = preprocessing.scale(x_test).reshape((len(xListTest), 60, 7, 1))

y_train = keras.utils.to_categorical(np.array(yListTrain), num_classes=2)
y_test = keras.utils.to_categorical(np.array(yListTest), num_classes=2)

model = Sequential()
# input: 60*5 images -> (60, 5) tensors.
# this applies 32 convolution filters of size 3x3 each.

# model.add(Conv2D(64, (3,3), activation='relu', input_shape = (60,7,1),strides=1))
model.add(Conv2D(64, (3, 3), input_shape=(60, 7, 1), strides=1))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.25))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2, activation='softmax', name="Dense_1"))

# opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.99, beta_2=0.98, epsilon=1e-08, schedule_decay=0.004)
opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.99, beta_2=0.98, epsilon=1e-08)
# opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=512, epochs=50, verbose=1, validation_split=0.25)

score = model.evaluate(x_test, y_test, verbose=1)

for i in range(len(score)):
    print(model.metrics_names[i], ": ", score[i])
