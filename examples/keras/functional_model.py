"""
Keras functional model example
==================

An example of an functional (non-sequential) network used as an OpenML flow.
"""

import openml
import openml.extensions.keras

import keras

############################################################################
# Define an input layer for the network. In this example we are using the
# credit-g dataset, which contains 20 features, and as such the input shape
# will be (20,).
inp = keras.layers.Input(shape=(20,))

# Normalize the input data in order to speed up the training process
normalized = keras.layers.BatchNormalization()(inp)

# Fork the input data into two parallel dense layers
# which use ReLU activation.
dense1 = keras.layers.Dense(units=64, activation='relu')(normalized)
dense2 = keras.layers.Dense(units=64, activation='relu')(normalized)

# Merge the results of the two parallel layers into one merge layer.
merged = keras.layers.concatenate([dense1, dense2])

# Introduce an additional Dense layer in combination to a dropout layer.
dense3 = keras.layers.Dense(units=64, activation='sigmoid')(merged)
dropout1 = keras.layers.Dropout(rate=0.25)(dense3)
# Finally, output the probabiltiies in the final dense layer using
# softmax activation.
dense4 = keras.layers.Dense(units=2, activation='softmax')(dropout1)

# We can now construct this model which uses our functional neural network
# with one input and one output.
model = keras.models.Model(inputs=[inp], outputs=[dense4])

# We will compile using the Adam optimizer while targeting accuracy.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
############################################################################

############################################################################
# Download the OpenML task for the german credit card dataset.
task = openml.tasks.get_task(31)
############################################################################
# Run the Keras model on the task.
run = openml.runs.run_model_on_task(model, task)
# Publish the experiment on OpenML (optional, requires an API key).
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))

############################################################################
