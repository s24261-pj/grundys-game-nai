import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from utils import normalize_data, plot_sample_images, plot_training_history, display_confusion_matrix

train_dir = './files/train'
test_dir = './files/test'

class_names = ['apple', 'banana', 'beetroot']

train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=(64, 64),
    batch_size=64,
    label_mode='int',
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=(64, 64),
    batch_size=64,
    label_mode='int',
    shuffle=False
)

x_train, y_train = [], []
for images, labels in train_dataset:
    x_train.append(images.numpy())
    y_train.append(labels.numpy())

x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

x_test, y_test = [], []
for images, labels in test_dataset:
    x_test.append(images.numpy())
    y_test.append(labels.numpy())

x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

x_train, x_test = normalize_data(x_train, x_test)

plot_sample_images(x_train, y_train, class_names)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.2f}")

plot_training_history(history)

y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)
display_confusion_matrix(y_test, y_pred_class, class_names)
