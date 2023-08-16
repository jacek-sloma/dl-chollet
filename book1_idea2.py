# Pomysł jest taki żeby pokazać obrazki które nie zostały prawidłowo rozpoznane przez sieć

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images_reshaped = train_images.reshape((60000, 28 * 28))
train_images_reshaped_casted_as_float = train_images_reshaped.astype('float32') / 255

test_images_reshaped = test_images.reshape((10000, 28 * 28))
test_images_reshaped_casted_as_float = test_images_reshaped.astype('float32') / 255

from keras.utils import to_categorical

train_labels_casted_to_categorical = to_categorical(train_labels)
test_labels_casted_to_categorical = to_categorical(test_labels)

network.fit(train_images_reshaped_casted_as_float,
            train_labels_casted_to_categorical,
            batch_size=256,
            epochs=10
            )

debug = ''

import matplotlib.pyplot as plt

errors = []
labels = []
counter = 0
predictions = network.predict(test_images_reshaped_casted_as_float)

for index, p in enumerate(predictions):
    prediction_result_as_int =  p.argmax()
    real_label = test_labels[index]
    if prediction_result_as_int != real_label:
        errors.append(test_images[index])
        labels.append(f'{prediction_result_as_int}/{real_label}')

fig = plt.figure(figsize=(10, 15))

rows = 15
columns = 14

for i in range(len(errors)):
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title(labels[i])
    plt.imshow(errors[i], cmap='binary')

plt.show()

