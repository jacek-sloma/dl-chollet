# Pomysł na to jak pokazać cyfry ze zbioru na ekranie

from keras.datasets import mnist
import matplotlib.pyplot as plt
# https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

numbers = []
numbers_amount = 900

counter = 0
while len(numbers) < numbers_amount:
    if train_labels[counter] == 7:
        numbers.append(train_images[counter])
    counter += 1

fig = plt.figure(figsize=(10, 10))
# setting values to rows and column variables
rows = 30
columns = 30

for i in range(numbers_amount):
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.imshow(numbers[i], cmap='binary')

plt.show()

