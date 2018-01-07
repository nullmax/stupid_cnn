import numpy as np
import json
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn import preprocessing

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

x_test = np.array(xList).reshape((len(xList), 420))
x_test = preprocessing.scale(x_test).reshape((len(xList), 60, 7, 1))

y_test = keras.utils.to_categorical(np.array(yList), num_classes=2)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(60, 7, 1), strides=1))
model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.25))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2, activation='softmax', name="Dense_1"))

opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.99, beta_2=0.98, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.load_weights("cnn.h5")

score = model.evaluate(x_test, y_test, verbose=1)

for i in range(len(score)):
    print(model.metrics_names[i], ": ", score[i])