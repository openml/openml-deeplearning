"""
Keras regression example
==================

An example of a sequential network used to solve a regression task.
"""

import keras

import openml.extensions.keras

############################################################################
# Define a sequential Keras model.
model = keras.models.Sequential([
    keras.layers.BatchNormalization(),
    keras.layers.Dense(units=1024, activation=keras.activations.relu),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(units=1, activation=keras.activations.linear),
])

# We will compile using the Adam optimizer while targeting mean absolute error.
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['mae', 'mse'])
############################################################################

############################################################################
# Download the OpenML task for the Friedman dataset.
task = openml.tasks.get_task(4958)
############################################################################
# Run the Keras model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
# Publish the experiment on OpenML (optional, requires an API key).
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))

############################################################################
