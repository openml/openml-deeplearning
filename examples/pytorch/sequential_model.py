import torch.nn

import openml
import openml.extensions.pytorch

model = torch.nn.Sequential(
    torch.nn.LayerNorm(20),
    torch.nn.Linear(20, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(64, 2),
    torch.nn.Softmax(dim=0)
)

task = openml.tasks.get_task(31)

run = openml.runs.run_model_on_task(model, task)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
