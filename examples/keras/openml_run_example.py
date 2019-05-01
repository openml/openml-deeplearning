"""
OpenML Run Example
==================

An example of an automated machine learning experiment.
"""
import openml
import openml.extensions.keras as keras_extension

import keras

model = keras.models.Sequential([
    keras.layers.BatchNormalization(),
    keras.layers.Dense(units=128, activation=keras.activations.relu),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(units=10, activation=keras.activations.softmax),
])

task = openml.tasks.get_task(3573)

run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
