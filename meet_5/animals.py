import tensorflow as tf
import numpy as np
from utils import filter_classes, normalize_data, plot_sample_images, plot_training_history, display_confusion_matrix

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

selected_classes = [2, 3, 4, 5, 6, 7]

x_train, y_train = filter_classes(x_train, y_train, selected_classes)
x_test, y_test = filter_classes(x_test, y_test, selected_classes)

x_train, x_test = normalize_data(x_train, x_test)

plot_sample_images(x_train, y_train, [class_names[i] for i in selected_classes])

model = tf.keras.models.Sequential([
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(6, activation='softmax')
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
display_confusion_matrix(y_test, y_pred_class, [class_names[i] for i in selected_classes])
