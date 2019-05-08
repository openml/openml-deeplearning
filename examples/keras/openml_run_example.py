import openml
import openml.extensions.keras as kerasext

import keras

model = keras.models.Sequential([
    keras.layers.BatchNormalization(),
    keras.layers.Dense(units=1024, activation=keras.activations.relu),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(units=2, activation=keras.activations.softmax),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

old_flow = kerasext.KerasExtension().model_to_flow(model)

result = old_flow.publish()

new_flow = openml.flows.get_flow(old_flow.flow_id, True)

task = openml.tasks.get_task(1793)

run = openml.runs.run_flow_on_task(new_flow, task, avoid_duplicate_runs=False)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
