import openml
import openml.extensions.mxnet

flow = openml.flows.get_flow(flow_id=12380, reinstantiate=True)

task = openml.tasks.get_task(31)

run = openml.runs.run_flow_on_task(flow, task, avoid_duplicate_runs=False)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
