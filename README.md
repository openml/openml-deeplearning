[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

An OpenML extension for deep learning frameworks.

Currently supported:
* Keras  
* PyTorch  
* MXNet  
* ONNX  

You can find usage examples in the [documentation](https://openml.github.io/openml-deeplearning/master/examples/index.html).

### OpenML Support Squad
Adrian-Stefan Mares, Andrei Danila, Bogdan Enache, Gergana Goncheva, Loïc Alexander Hijl, Veselin Minev, Thanh-Dat Nguyen, Antoine Labasse, Lutou Nijhuis, Claudiu-Teodor Nohai, Dragos Mihai Serban, Tsvetan Zahariev, Sonya Zarkova

### Code Structure
<pre>
.
├── appveyor - DON'T CHECK - AUTOMATICALLY GENERATED
│   └── run_with_env.cmd
├── appveyor.yml
├── ci_scripts - DON'T CHECK - 3RD PARTY SOURCES
│   ├── create_doc.sh
│   ├── flake8_diff.sh
│   ├── install.sh
│   ├── success.sh
│   └── test.sh
├── CONTRIBUTING.md - DON'T CHECK - 3RD PARTY SOURCES
├── doc - DON'T CHECK - 3RD PARTY SOURCES
│   ├── api.rst
│   ├── conf.py
│   ├── contributing.rst
│   ├── index.rst
│   ├── Makefile
│   ├── progress.rst
│   ├── _static
│   │   └── codehighlightstyle.css
│   ├── _templates
│   │   ├── class.rst
│   │   ├── class_without_init.rst
│   │   ├── function.rst
│   │   └── layout.html
│   └── usage.rst
├── examples
│   ├── create_upload_tutorial.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── datasets_tutorial.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── fetch_evaluations_tutorial.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── flows_and_runs_tutorial.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── introduction_tutorial.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── keras - CHECK - OUR CODE
│   │   ├── keras_functional_classification_model.py
│   │   ├── keras_sequential_classification_model.py
│   │   ├── keras_sequential_regression_model.py
│   │   └── README.txt
│   ├── mxnet - CHECK - OUR CODE
│   │   ├── mxnet_mnist_visdom.py
│   │   ├── mxnet_sequential_classification_model.py
│   │   ├── mxnet_sequential_regression_model.py
│   │   └── README.txt
│   ├── onnx - CHECK - OUR CODE
│   │   ├── onnx_mxnet_model.py
│   │   └── README.txt
│   ├── pytorch - CHECK - OUR CODE
│   │   ├── pytorch_mnist_visdom.py
│   │   ├── pytorch_sequential_classification_model.py
│   │   ├── pytorch_sequential_regression_model.py
│   │   └── README.txt
│   ├── README.txt - DON'T CHECK - 3RD PARTY SOURCES
│   ├── run_setup_tutorial.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── sklearn - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── openml_run_example.py
│   │   └── README.txt
│   └── tasks_tutorial.py - DON'T CHECK - 3RD PARTY SOURCES
├── LICENSE - DON'T CHECK - 3RD PARTY SOURCES
├── Makefile - DON'T CHECK - 3RD PARTY SOURCES
├── MANIFEST.in - DON'T CHECK - 3RD PARTY SOURCES
├── openml
│   ├── _api_calls.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── config.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── datasets - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── data_feature.py
│   │   ├── dataset.py
│   │   ├── functions.py
│   │   └── __init__.py
│   ├── evaluations - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── evaluation.py
│   │   ├── functions.py
│   │   └── __init__.py
│   ├── exceptions.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── extensions
│   │   ├── extension_interface.py - DON'T CHECK - 3RD PARTY SOURCES (IMPLEMENTED BY OUR EXTENSIONS)
│   │   ├── functions.py - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── __init__.py
│   │   ├── keras - CHECK - OUR CODE
│   │   │   ├── extension.py
│   │   │   └── __init__.py
│   │   ├── mxnet - CHECK - OUR CODE
│   │   │   ├── config.py
│   │   │   ├── extension.py
│   │   │   └── __init__.py
│   │   ├── onnx - CHECK - OUR CODE
│   │   │   ├── config.py
│   │   │   ├── extension.py
│   │   │   └── __init__.py
│   │   ├── pytorch - CHECK - OUR CODE
│   │   │   ├── config.py
│   │   │   ├── extension.py
│   │   │   ├── __init__.py
│   │   │   └── layers
│   │   │       ├── functional.py
│   │   │       └── __init__.py
│   │   └── sklearn - DON'T CHECK - 3RD PARTY SOURCES
│   │       ├── extension.py
│   │       └── __init__.py
│   ├── flows - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── flow.py
│   │   ├── functions.py
│   │   └── __init__.py
│   ├── __init__.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── runs - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── functions.py
│   │   ├── __init__.py
│   │   ├── run.py
│   │   └── trace.py
│   ├── setups - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── functions.py
│   │   ├── __init__.py
│   │   └── setup.py
│   ├── study - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── functions.py
│   │   ├── __init__.py
│   │   └── study.py
│   ├── tasks - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── functions.py
│   │   ├── __init__.py
│   │   ├── split.py
│   │   └── task.py
│   ├── testing.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── utils.py - DON'T CHECK - 3RD PARTY SOURCES
│   └── __version__.py - DON'T CHECK - 3RD PARTY SOURCES
├── README.md - DON'T CHECK - 3RD PARTY SOURCES
├── setup.cfg - DON'T CHECK - 3RD PARTY SOURCES
├── setup.py - DON'T CHECK - 3RD PARTY SOURCES
├── tests
│   ├── files - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── misc
│   │   │   └── trace.arff
│   │   ├── models
│   │   │   ├── model_task_10101.onnx
│   │   │   ├── model_task_4823.onnx
│   │   │   ├── model_task_52948.onnx
│   │   │   └── model_task_9914.onnx
│   │   └── org
│   │       └── openml
│   │           └── test
│   │               ├── datasets
│   │               │   ├── -1
│   │               │   │   ├── dataset.arff
│   │               │   │   ├── description.xml
│   │               │   │   ├── features.xml
│   │               │   │   └── qualities.xml
│   │               │   └── 2
│   │               │       ├── dataset.arff
│   │               │       ├── description.xml
│   │               │       ├── features.xml
│   │               │       └── qualities.xml
│   │               ├── runs
│   │               │   └── 1
│   │               │       └── description.xml
│   │               ├── setups
│   │               │   └── 1
│   │               │       └── description.xml
│   │               └── tasks
│   │                   ├── 1
│   │                   │   ├── datasplits.arff
│   │                   │   └── task.xml
│   │                   ├── 1882
│   │                   │   ├── datasplits.arff
│   │                   │   └── task.xml
│   │                   └── 3
│   │                       ├── datasplits.arff
│   │                       └── task.xml
│   ├── __init__.py - DON'T CHECK - 3RD PARTY SOURCES
│   ├── test_datasets - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── __init__.py
│   │   ├── test_dataset_functions.py
│   │   └── test_dataset.py
│   ├── test_evaluations - DON'T CHECK - 3RD PARTY SOURCES
│   │   └── test_evaluation_functions.py
│   ├── test_examples - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── __init__.py
│   │   └── test_OpenMLDemo.py
│   ├── test_extensions
│   │   ├── __init__.py - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── test_functions.py - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── test_keras_extension - CHECK - OUR CODE
│   │   │   ├── __init__.py
│   │   │   ├── test_keras_additional_functions.py
│   │   │   ├── test_keras_deserialization.py
│   │   │   ├── test_keras_run_model.py
│   │   │   └── test_keras_serialization.py
│   │   ├── test_mxnet_extension - CHECK - OUR CODE
│   │   │   ├── __init__.py
│   │   │   ├── test_mxnet_additional_functions.py
│   │   │   ├── test_mxnet_deserialization.py
│   │   │   ├── test_mxnet_run_model.py
│   │   │   └── test_mxnet_serialization.py
│   │   ├── test_onnx_extension - CHECK - OUR CODE
│   │   │   ├── __init__.py
│   │   │   ├── onnx_model_utils.py
│   │   │   ├── test_onnx_additional_functions.py
│   │   │   ├── test_onnx_deserialization.py
│   │   │   ├── test_onnx_run_model.py
│   │   │   └── test_onnx_serialization.py
│   │   ├── test_pytorch_extension - CHECK - OUR CODE
│   │   │   ├── __init__.py
│   │   │   ├── test_pytorch_additional_functions.py
│   │   │   ├── test_pytorch_deserialization.py
│   │   │   ├── test_pytorch_run_model.py
│   │   │   └── test_pytorch_serialization.py
│   │   └── test_sklearn_extension - DON'T CHECK - 3RD PARTY SOURCES
│   │       ├── __init__.py
│   │       └── test_sklearn_extension.py
│   ├── test_flows - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── dummy_learn
│   │   │   ├── dummy_forest.py
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   ├── test_flow_functions.py
│   │   └── test_flow.py
│   ├── test_openml - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   └── test_openml.py
│   ├── test_runs - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── __init__.py
│   │   ├── test_run_functions.py
│   │   ├── test_run.py
│   │   └── test_trace.py
│   ├── test_setups - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── __init__.py
│   │   └── test_setup_functions.py
│   ├── test_study - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── test_study_examples.py
│   │   └── test_study_functions.py
│   ├── test_tasks - DON'T CHECK - 3RD PARTY SOURCES
│   │   ├── __init__.py
│   │   ├── test_classification_task.py
│   │   ├── test_clustering_task.py
│   │   ├── test_learning_curve_task.py
│   │   ├── test_regression_task.py
│   │   ├── test_split.py
│   │   ├── test_supervised_task.py
│   │   ├── test_task_functions.py
│   │   ├── test_task_methods.py
│   │   └── test_task.py
│   └── test_utils - DON'T CHECK - 3RD PARTY SOURCES
│       ├── __init__.py
│       └── test_utils.py
└── visualization - CHECK - OUR CODE
    ├── constants.py
    ├── __init__.py
    ├── tests
    │   ├── base.py
    │   ├── __init__.py
    │   ├── test_visualization_flow.py
    │   ├── test_visualization_run.py
    │   ├── test_visualization_utils.py
    │   └── utils.py
    ├── utils.py
    └── visualizer.py
</pre>
