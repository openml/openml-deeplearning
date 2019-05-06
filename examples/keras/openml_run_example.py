import openml
import openml.extensions.keras as keras_extension

import keras

model = keras.models.Sequential([
    keras.layers.BatchNormalization(),
    keras.layers.Reshape(target_shape=(32, 32, 3)),
    keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=keras.activations.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=1024, activation=keras.activations.relu),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(units=10, activation=keras.activations.softmax),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

task = openml.tasks.get_task(167124)

run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
