import openml
import openml.extensions.keras

import keras
from keras.layers import *
from keras.models import Model

openml.config.start_using_configuration_for_example()
openml.config.stop_using_configuration_for_example()

task = openml.tasks.get_task(1)

output_length = len(task.class_labels)

model = keras.models.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.Dense(units=1024, activation=keras.activations.relu),
        keras.layers.Dropout(rate=0.4),
        keras.layers.Dense(units=output_length, activation=keras.activations.softmax),
])
# inputs = Input(shape=(15,))

# a layer instance is callable on a tensor, and returns a tensor
# x = BatchNormalization()(inputs)
# x = Dense(1024, activation=keras.activations.relu)(x)
# x = Dropout(rate=0.4)(x)
# predictions = Dense(2, activation=keras.activations.softmax)(x)

inputs = Input(shape=(10, )) # TODO run task to find shape
x = Dense(10, activation=keras.activations.relu)(inputs)
predictions = Dense(1, activation=keras.activations.relu)(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


task = openml.tasks.get_task(189051)

run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
run.publish()



print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
