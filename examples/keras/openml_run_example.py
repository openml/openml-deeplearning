import openml
import openml.extensions.keras

import keras

model = keras.models.Sequential([
    keras.layers.BatchNormalization(),
    keras.layers.Dense(units=256, activation=keras.activations.relu),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(units=2, activation=keras.activations.softmax),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

task = openml.tasks.get_task(1793)

run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
