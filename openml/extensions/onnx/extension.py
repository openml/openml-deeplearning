import copy
import importlib
import json
import logging
import re
import sys
import time
import warnings
import zlib
import math
from collections import OrderedDict  # noqa: F401
from distutils.version import LooseVersion
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import onnx
import numpy as np
import pandas as pd
import scipy.sparse
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
from mxnet import nd, gluon, autograd
from google.protobuf import json_format

import openml
from openml.exceptions import PyOpenMLError
from openml.extensions import Extension, register_extension
from openml.flows import OpenMLFlow
from openml.runs.trace import OpenMLRunTrace, OpenMLTraceIteration
from openml.tasks import (
    OpenMLTask,
    OpenMLSupervisedTask,
    OpenMLClassificationTask,
    OpenMLRegressionTask,
)

if sys.version_info >= (3, 5):
    from json.decoder import JSONDecodeError
else:
    JSONDecodeError = ValueError

DEPENDENCIES_PATTERN = re.compile(
    r'^(?P<name>[\w\-]+)((?P<operation>==|>=|>)'
    r'(?P<version>(\d+\.)?(\d+\.)?(\d+)?(dev)?[0-9]*))?$'
)

SIMPLE_NUMPY_TYPES = [nptype for type_cat, nptypes in np.sctypes.items()
                      for nptype in nptypes if type_cat != 'others']
SIMPLE_TYPES = tuple([bool, int, float, str] + SIMPLE_NUMPY_TYPES)


LAYER_PATTERN = re.compile(r'layer\d+\_(.*)')


class OnnxExtension(Extension):
    """Connect ONNX to OpenML-Python."""

    ################################################################################################
    # General setup

    @classmethod
    def can_handle_flow(cls, flow: 'OpenMLFlow') -> bool:
        """Check whether a given flow describes an ONNX specification.

        This is done by parsing the ``external_version`` field.

        Parameters
        ----------
        flow : OpenMLFlow

        Returns
        -------
        bool
        """
        return cls._is_onnx_flow(flow)

    @classmethod
    def can_handle_model(cls, model: Any) -> bool:
        """Check whether a model is an instance of ``onnx.ModelProto``.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        return isinstance(model, onnx.ModelProto)

    ################################################################################################
    # Methods for flow serialization and de-serialization

    def flow_to_model(self, flow: 'OpenMLFlow', initialize_with_defaults: bool = False) -> Any:
        """Initializes an ONNX model representation based on a flow.

        Parameters
        ----------
        flow : mixed
            the object to deserialize (can be flow object, or any serialized
            parameter value that is accepted by)

        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        Returns
        -------
        mixed
        """
        return self._deserialize_onnx(flow, initialize_with_defaults=initialize_with_defaults)

    def _deserialize_onnx(
            self,
            flow: 'OpenMlFlow',
            components: Optional[Dict] = None,
            initialize_with_defaults: bool = False,
            recursion_depth: int = 0,
    ) -> Any:
        """Recursive function to deserialize a onnx flow.

        This function delegates all work to the respective functions to deserialize special data
        structures etc.

        Parameters
        ----------
        o : mixed
            the object to deserialize (can be flow object, or any serialized
            parameter value that is accepted by)

        components : dict


        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        recursion_depth : int
            The depth at which this flow is called, mostly for debugging
            purposes

        Returns
        -------
        mixed
        """

        logging.info('-%s deserialize %s' % ('-' * recursion_depth, flow.name))
        self._check_dependencies(flow.dependencies)

        parameters = flow.parameters
        graph = parameters['graph']

        model_dic = {}
        model_dic['graph'] = graph

        for key, value in parameters['backend'].items():
            model_dic[key] = value

        model = onnx.ModelProto()

        json_format.ParseDict(model_dic, model)

        return model

    def model_to_flow(self, model: Any) -> 'OpenMLFlow':
        """Transform an ONNX model representation to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """
        # Necessary to make pypy not complain about all the different possible return types
        return self._serialize_onnx(model)

    def _serialize_onnx(self, model: Any) -> OpenMLFlow:
        """Create an OpenMLFlow.

        Serializes the protobuf to a python dictionary to create OpenMLFlow

        Parameters
        ----------
        model : ONNX representation of deep learning model

        Returns
        -------
        OpenMLFlow

        """
        # Convert the protobuf to python dictionary
        model_dic = json_format.MessageToDict(model)

        parameters = {}
        parameters_meta_info = {}
        parameters['backend'] = {}
        parameters_meta_info['backend'] = OrderedDict((('description', None), ('data_type', None)))

        # Add graph information to parameters dictionary
        # for key, value in model_dic['graph'].items():
        for key, value in sorted(model_dic['graph'].items(), key=lambda t: t[0]):
            if isinstance(value, list):
                for (index, val) in enumerate(value):
                    k = '{}_{}_{}'.format(key, str(index), val['name'])
                    v = val
                    if key == 'initializer':
                        del v['floatData']
                    parameters[k] = json.dumps(v)
                    parameters_meta_info[k] = OrderedDict((('description', None),
                                                           ('data_type', None)))
            else:
                parameters[key] = value
                parameters_meta_info[key] = OrderedDict((('description', None),
                                                         ('data_type', None)))

        # Add backend information to parameters dictionary
        for key, value in model_dic.items():
            parameters['backend'][key] = value

        # Remove redundant graph information
        del parameters['backend']['graph']

        parameters['backend'] = json.dumps(parameters['backend'])

        parameters = OrderedDict(sorted(parameters.items(), key=lambda x: x[0]))
        parameters_meta_info = OrderedDict(sorted(parameters_meta_info.items(), key=lambda x: x[0]))

        # Create a flow name, which contains a hash of the parameters as part of the name
        # This is done in order to ensure that we are not exceeding the 1024 character limit
        # of the API, since NNs can become quite large
        class_name = model.__module__ + "." + model.__class__.__name__
        class_name += '.' + format(
            zlib.crc32(json.dumps(parameters, sort_keys=True).encode('utf8')),
            'x'
        )

        # Get the external versions of all sub-components
        external_version = self._get_external_version_string(model, OrderedDict())

        dependencies = '\n'.join([
            self._format_external_version(
                'onnx',
                onnx.__version__,
            ),
            self._format_external_version(
                'mxnet',
                mx.__version__,
            ),
            'numpy>=1.6.1,<=1.14.6',
            'scipy>=0.9,<=1.2.1',
        ])

        name = class_name

        # For ONNX, components and parameters_meta_info are empty so they are initialized with
        # empty ordered dictionaries
        components = OrderedDict()

        onnx_version = self._format_external_version('onnx', onnx.__version__)
        onnx_version_formatted = onnx_version.replace('==', '_')
        flow = OpenMLFlow(name=name,
                          class_name=class_name,
                          description='Automatically created ONNX flow.',
                          model=model,
                          components=components,
                          parameters=parameters,
                          parameters_meta_info=parameters_meta_info,
                          external_version=external_version,
                          tags=['openml-python', 'onnx',
                                'python', onnx_version_formatted,

                                ],
                          language='English',
                          # TODO fill in dependencies!
                          dependencies=dependencies)

        return flow

    def get_version_information(self) -> List[str]:
        """List versions of libraries required by the flow.

        Libraries listed are ``Python``, ``onnx``, ``numpy`` and ``scipy``.

        Returns
        -------
        List
        """

        # This can possibly be done by a package such as pyxb, but I could not get
        # it to work properly.
        import onnx
        import scipy
        import numpy
        import mxnet

        major, minor, micro, _, _ = sys.version_info
        python_version = 'Python_{}.'.format(
            ".".join([str(major), str(minor), str(micro)]))
        onnx_version = 'Onnx_{}.'.format(onnx.__version__)
        mxnet_version = 'MXNet_{}.'.format(mxnet.__version__)
        numpy_version = 'NumPy_{}.'.format(numpy.__version__)
        scipy_version = 'SciPy_{}.'.format(scipy.__version__)

        return [python_version, onnx_version, mxnet_version, numpy_version, scipy_version]

    def create_setup_string(self, model: Any) -> str:
        """Create a string which can be used to reinstantiate the given model.

        Parameters
        ----------
        model : Any

        Returns
        -------
        str
        """
        run_environment = " ".join(self.get_version_information())
        # fixme str(model) might contain (...)
        return run_environment + " " + str(model)

    @classmethod
    def _is_onnx_flow(cls, flow: OpenMLFlow) -> bool:
        return (flow.external_version.startswith('onnx==')
                or ',onnx==' in flow.external_version)

    def _get_external_version_string(
            self,
            model: Any,
            sub_components: Dict[str, OpenMLFlow],
    ) -> str:
        # Create external version string for a flow, given the model and the
        # already parsed dictionary of sub_components. Retrieves the external
        # version of all subcomponents, which themselves already contain all
        # requirements for their subcomponents. The external version string is a
        # sorted concatenation of all modules which are present in this run.
        model_package_name = model.__module__.split('_')[0]
        module = importlib.import_module(model_package_name)
        model_package_version_number = module.__version__  # type: ignore
        external_version = self._format_external_version(
            model_package_name, model_package_version_number,
        )
        openml_version = self._format_external_version('openml', openml.__version__)
        external_versions = set()
        external_versions.add(external_version)
        external_versions.add(openml_version)
        for visitee in sub_components.values():
            for external_version in visitee.external_version.split(','):
                external_versions.add(external_version)
        return ','.join(list(sorted(external_versions)))

    def _from_parameters(self, parameters: 'OrderedDict[str, Any]') -> Any:
        # Get an ONNX model representation from flow parameters
        # Create a dict and recursively fill it with model components
        # First do this for non-layer items, then layer items.
        config = {}
        # Add the expected configuration parameters back to the configuration dictionary,
        # as long as they are not layers, since they need to be deserialized separately
        for k, v in parameters.items():
            if not LAYER_PATTERN.match(k):
                config[k] = self._deserialize_onnx(v)

        # Recreate the layers list and start to deserialize them back to the correct location
        config['config']['layers'] = []
        for k, v in parameters.items():
            if LAYER_PATTERN.match(k):
                v = self._deserialize_onnx(v)
                config['config']['layers'].append(v)

        # Deserialize the model from the configuration dictionary
        model = keras.layers.deserialize(config)

        # Attempt to recompile the model if compilation parameters were present
        # during serialization
        if 'optimizer' in parameters:
            training_config = self._deserialize_onnx(parameters['optimizer'])
            optimizer_config = training_config['optimizer_config']
            optimizer = keras.optimizers.deserialize(optimizer_config)

            # Recover loss functions and metrics
            loss = training_config['loss']
            metrics = training_config['metrics']
            sample_weight_mode = training_config['sample_weight_mode']
            loss_weights = training_config['loss_weights']

            # Compile model
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics,
                          loss_weights=loss_weights,
                          sample_weight_mode=sample_weight_mode)
        else:
            warnings.warn('No training configuration found inside the flow: '
                          'the model was *not* compiled. '
                          'Compile it manually.')

        return model

    def _get_parameters(self, model: Any) -> 'OrderedDict[str, Optional[str]]':
        # Get the parameters from a model in an OrderedDict
        parameters = OrderedDict()  # type: OrderedDict[str, Any]

        # Construct the configuration dictionary in the same manner as
        # keras.engine.Network.to_json does
        model_config = {
            'class_name': model.__class__.__name__,
            'config': model.get_config()
        }

        # Remove the layers from the configuration in order to allow them to be
        # pretty printed as model parameters
        layers = model_config['config']['layers']
        del model_config['config']['layers']

        # Add the rest of the model configuration entries to the parameter list
        for k, v in model_config.items():
            parameters[k] = self._serialize_onnx(v, model)

        # Compute the format of the layer numbering. This pads the layer numbers with 0s in
        # order to ensure that the layers are printed in a human-friendly order, instead of
        # having weird orderings
        max_len = int(np.ceil(np.log10(len(layers))))
        len_format = '{0:0>' + str(max_len) + '}'

        # Add the layers as hyper-parameters
        for i, v in enumerate(layers):
            layer = v['config']
            k = 'layer' + len_format.format(i) + "_" + layer['name']
            parameters[k] = self._serialize_onnx(v, model)

        # Introduce the optimizer settings as hyper-parameters, if the model has been compiled
        if model.optimizer:
            parameters['optimizer'] = self._serialize_onnx({
                'optimizer_config': {
                    'class_name': model.optimizer.__class__.__name__,
                    'config': model.optimizer.get_config()
                },
                'loss': model.loss,
                'metrics': model.metrics,
                'weighted_metrics': model.weighted_metrics,
                'sample_weight_mode': model.sample_weight_mode,
                'loss_weights': model.loss_weights,
            }, model)

        return parameters

    def _check_dependencies(self, dependencies: str) -> None:
        """
        Checks whether the dependencies required for the deserialization of an OpenMLFlow are met

        Parameters
        ----------
        dependencies : str
                       a string representing the required dependencies

        Returns
        -------
        None
        """
        if not dependencies:
            return

        dependencies_list = dependencies.split('\n')
        for dependency_string in dependencies_list:
            match = DEPENDENCIES_PATTERN.match(dependency_string)
            if not match:
                raise ValueError('Cannot parse dependency %s' % dependency_string)

            dependency_name = match.group('name')
            operation = match.group('operation')
            version = match.group('version')

            module = importlib.import_module(dependency_name)
            required_version = LooseVersion(version)
            installed_version = LooseVersion(module.__version__)  # type: ignore

            if operation == '==':
                check = required_version == installed_version
            elif operation == '>':
                check = installed_version > required_version
            elif operation == '>=':
                check = (installed_version > required_version
                         or installed_version == required_version)
            else:
                raise NotImplementedError(
                    'operation \'%s\' is not supported' % operation)
            if not check:
                raise ValueError('Trying to deserialize a model with dependency '
                                 '%s not satisfied.' % dependency_string)

    def _format_external_version(
            self,
            model_package_name: str,
            model_package_version_number: str,
    ) -> str:
        """
        Returns a formatted string representing the required dependencies for a flow

        Parameters
        ----------
        model_package_name : str
                           the name of the required package
        model_package_version_number : str
                           the version of the required package
        Returns
        -------
        str
        """
        return '%s==%s' % (model_package_name, model_package_version_number)

    def _can_measure_cputime(self, model: Any) -> bool:
        """
        Returns True if the parameter settings of model are chosen s.t. the model
        will run on a single core (if so, openml-python can measure cpu-times)

        Parameters:
        -----------
        model:
            The model that will be fitted

        Returns:
        --------
        bool:
            False
        """

        return False

    def _can_measure_wallclocktime(self, model: Any) -> bool:
        """
        Returns True if the parameter settings of model are chosen s.t. the model
        will run on a preset number of cores (if so, openml-python can measure wall-clock time)

        Parameters:
        -----------
        model:
            The model that will be fitted

        Returns:
        --------
        bool:
            False
        """

        return False

    ################################################################################################
    # Methods for performing runs with extension modules

    def is_estimator(self, model: Any) -> bool:
        """Check whether the given model is an ONNX model representation.

        This function is only required for backwards compatibility and will be removed in the
        near future.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        return isinstance(model, onnx.ModelProto)

    def seed_model(self, model: Any, seed: Optional[int] = None) -> Any:
        """
        # TODO: Does this still apply?
        Not applied for ONNX, since there are no random states in ONNX.

        Parameters
        ----------
        model : ONNX model representation
            The model to be seeded
        seed : int
            The seed to initialize the RandomState with. Unseeded subcomponents
            will be seeded with a random number from the RandomState.

        Returns
        -------
        Any
        """

        return model

    def _run_model_on_fold(
            self,
            model: Any,
            task: 'OpenMLTask',
            X_train: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame],
            rep_no: int,
            fold_no: int,
            y_train: Optional[np.ndarray] = None,
            X_test: Optional[Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, 'OrderedDict[str, float]', Optional[OpenMLRunTrace]]:
        """Run a model on a repeat,fold,subsample triplet of the task and return prediction
        information.

        Furthermore, it will measure run time measures in case multi-core behaviour allows this.
        * exact user cpu time will be measured if the number of cores is set (recursive throughout
        the model) exactly to 1
        * wall clock time will be measured if the number of cores is set (recursive throughout the
        model) to any given number (but not when it is set to -1)

        Returns the data that is necessary to construct the OpenML Run object. Is used by
        run_task_get_arff_content. Do not use this function unless you know what you are doing.

        Parameters
        ----------
        model : Any
            The UNTRAINED model to run. The model instance will be copied and not altered.
        task : OpenMLTask
            The task to run the model on.
        X_train : array-like
            Training data for the given repetition and fold.
        rep_no : int
            The repeat of the experiment (0-based; in case of 1 time CV, always 0)
        fold_no : int
            The fold nr of the experiment (0-based; in case of holdout, always 0)
        y_train : Optional[np.ndarray] (default=None)
            Target attributes for supervised tasks. In case of classification, these are integer
            indices to the potential classes specified by dataset.
        X_test : Optional, array-like (default=None)
            Test attributes to test for generalization in supervised tasks.

        Returns
        -------
        arff_datacontent : List[List]
            Arff representation (list of lists) of the predictions that were
            generated by this fold (required to populate predictions.arff)
        arff_tracecontent :  List[List]
            Arff representation (list of lists) of the trace data that was generated by this
            fold
            (will be used to populate trace.arff, leave it empty if the model did not perform
            any
            hyperparameter optimization).
        user_defined_measures : OrderedDict[str, float]
            User defined measures that were generated on this fold
        model : Any
            The model trained on this repeat,fold,subsample triple. Will be used to generate
            trace
            information later on (in ``obtain_arff_trace``).
        """

        def _prediction_to_probabilities(y: np.ndarray, classes: List[Any]) -> np.ndarray:
            """Transforms predicted probabilities to match with OpenML class indices.

            Parameters
            ----------
            y : np.ndarray
                Predicted probabilities (possibly omitting classes if they were not present in the
                training data).
            classes : list
                List of classes known_predicted by the model, ordered by their index.

            Returns
            -------
            np.ndarray
            """
            # y: list or numpy array of predictions
            # model_classes: keras classifier mapping from original array id to
            # prediction index id
            if not isinstance(classes, list):
                raise ValueError('please convert model classes to list prior to '
                                 'calling this fn')
            result = np.zeros((len(y), len(classes)), dtype=np.float32)
            for obs, prediction_idx in enumerate(y):
                result[obs][prediction_idx] = 1.0
            return result

        if isinstance(task, OpenMLSupervisedTask):
            if y_train is None:
                raise TypeError('argument y_train must not be of type None')
            if X_test is None:
                raise TypeError('argument X_test must not be of type None')

        # TODO: if possible, give a warning if model is already fitted (acceptable
        # in case of custom experimentation,
        # but not desirable if we want to upload to OpenML).

        # Save model to file and import it as MXNet model
        # TODO: Constant CPU context?
        onnx.save(model, 'model.onnx')
        model_mx = onnx_mxnet.import_to_gluon('model.onnx', ctx=mx.cpu())

        # Reinitialize weights and bias
        # TODO: Find way to initialize using Xavier
        model_mx.initialize(init=mx.init.Uniform(), force_reinit=True)

        # Sanitize train and test data
        X_train[np.isnan(X_train)] = 1.0e-12
        X_test[np.isnan(X_test)] = 1.0e-12

        # Runtime can be measured if the model is run sequentially
        can_measure_cputime = self._can_measure_cputime(model_mx)
        can_measure_wallclocktime = self._can_measure_wallclocktime(model_mx)

        user_defined_measures = OrderedDict()  # type: 'OrderedDict[str, float]'

        try:
            # for measuring runtime. Only available since Python 3.3
            modelfit_start_cputime = time.process_time()
            modelfit_start_walltime = time.time()

            if isinstance(task, OpenMLSupervisedTask):
                # TODO: Extract to different function?
                if isinstance(task, OpenMLClassificationTask):
                    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
                elif isinstance(task, OpenMLRegressionTask):
                    loss_fn = gluon.loss.L2Loss()
                else:
                    raise TypeError('Task not supported')

                # Define trainer
                # TODO: Enable user to configure those
                trainer = gluon.Trainer(model_mx.collect_params(), 'sgd')
                batch_size = 32
                epochs = 20
                nr_of_batches = math.ceil(X_train.shape[0] / batch_size)

                for j in range(epochs):
                    for i in range(nr_of_batches):
                        input = nd.array(X_train[i*batch_size:(i+1)*batch_size])
                        labels = nd.array(y_train[i*batch_size:(i+1)*batch_size])

                        # Train the model
                        with autograd.record():
                            output = model_mx(input)
                            loss = loss_fn(output, labels)

                        loss.backward()
                        trainer.step(input.shape[0])

            modelfit_dur_cputime = (time.process_time() - modelfit_start_cputime) * 1000
            if can_measure_cputime:
                user_defined_measures['usercpu_time_millis_training'] = modelfit_dur_cputime

            modelfit_dur_walltime = (time.time() - modelfit_start_walltime) * 1000
            if can_measure_wallclocktime:
                user_defined_measures['wall_clock_time_millis_training'] = modelfit_dur_walltime

        except AttributeError as e:
            # typically happens when training a regressor on classification task
            raise PyOpenMLError(str(e))

        if isinstance(task, OpenMLClassificationTask):
            # TODO: Check if needs to be changed
            model_classes = mx.nd.argmax(nd.array(y_train), axis=-1)

        modelpredict_start_cputime = time.process_time()
        modelpredict_start_walltime = time.time()

        # In supervised learning this returns the predictions for Y, in clustering
        # it returns the clusters
        if isinstance(task, OpenMLSupervisedTask):
            pred_y = model_mx(nd.array(X_test))
            if isinstance(task, OpenMLClassificationTask):
                pred_y = mx.nd.argmax(pred_y, -1)
                pred_y = (pred_y.asnumpy()).astype(np.int64)
            if isinstance(task, OpenMLRegressionTask):
                pred_y = pred_y.asnumpy()
                pred_y = pred_y.reshape((-1))
        else:
            raise ValueError(task)

        if can_measure_cputime:
            modelpredict_duration_cputime = (time.process_time()
                                             - modelpredict_start_cputime) * 1000
            user_defined_measures['usercpu_time_millis_testing'] = modelpredict_duration_cputime
            user_defined_measures['usercpu_time_millis'] = (modelfit_dur_cputime
                                                            + modelpredict_duration_cputime)
        if can_measure_wallclocktime:
            modelpredict_duration_walltime = (time.time() - modelpredict_start_walltime) * 1000
            user_defined_measures['wall_clock_time_millis_testing'] = modelpredict_duration_walltime
            user_defined_measures['wall_clock_time_millis'] = (modelfit_dur_walltime
                                                               + modelpredict_duration_walltime)

        if isinstance(task, OpenMLClassificationTask):

            try:
                proba_y = model_mx(nd.array(X_test)).asnumpy()
            except AttributeError:
                if task.class_labels is not None:
                    proba_y = _prediction_to_probabilities(pred_y, list(task.class_labels))
                else:
                    raise ValueError('The task has no class labels')
            if task.class_labels is not None:
                if proba_y.shape[1] != len(task.class_labels):
                    # Remap the probabilities in case there was a class missing at training time
                    # By default, the classification targets are mapped to be zero-based indices
                    # to the actual classes. Therefore, the model_classes contain the correct
                    # indices to the correct probability array. Example:
                    # classes in the dataset: 0, 1, 2, 3, 4, 5
                    # classes in the training set: 0, 1, 2, 4, 5
                    # then we need to add a column full of zeros into the probabilities for class 3
                    # (because the rest of the library expects that the probabilities are ordered
                    # the same way as the classes are ordered).
                    proba_y_new = np.zeros((proba_y.shape[0], len(task.class_labels)))
                    for idx, model_class in enumerate(model_classes):
                        proba_y_new[:, model_class] = proba_y[:, idx]
                    proba_y = proba_y_new

                if proba_y.shape[1] != len(task.class_labels):
                    message = "Estimator only predicted for {}/{} classes!".format(
                        proba_y.shape[1], len(task.class_labels),
                    )
                    warnings.warn(message)
                    openml.config.logger.warn(message)

        elif isinstance(task, OpenMLRegressionTask):
            proba_y = None
        else:
            raise TypeError(type(task))

        trace = None

        return pred_y, proba_y, user_defined_measures, trace

    def obtain_parameter_values(
            self,
            flow: 'OpenMLFlow',
            model: Any = None,
    ) -> List[Dict[str, Any]]:
        """Extracts all parameter settings required for the flow from the model.

        If no explicit model is provided, the parameters will be extracted from `flow.model`
        instead.

        Parameters
        ----------
        flow : OpenMLFlow
            OpenMLFlow object (containing flow ids, i.e., it has to be downloaded from the server)

        model: Any, optional (default=None)
            The model from which to obtain the parameter values. Must match the flow signature.
            If None, use the model specified in ``OpenMLFlow.model``.

        Returns
        -------
        list
            A list of dicts, where each dict has the following entries:
            - ``oml:name`` : str: The OpenML parameter name
            - ``oml:value`` : mixed: A representation of the parameter value
            - ``oml:component`` : int: flow id to which the parameter belongs
        """
        openml.flows.functions._check_flow_for_server_id(flow)

        def get_flow_dict(_flow):
            flow_map = {_flow.name: _flow.flow_id}
            for subflow in _flow.components:
                flow_map.update(get_flow_dict(_flow.components[subflow]))
            return flow_map

        def extract_parameters(_flow, _flow_dict, component_model,
                               _main_call=False, main_id=None):
            def is_subcomponent_specification(values):
                # checks whether the current value can be a specification of
                # subcomponents, as for example the value for steps parameter.
                # These are always lists/tuples of lists/tuples, size bigger
                # than 2 and an OpenMLFlow item involved.
                if not isinstance(values, (tuple, list)):
                    return False
                for item in values:
                    if not isinstance(item, (tuple, list)):
                        return False
                    if len(item) < 2:
                        return False
                    if not isinstance(item[1], openml.flows.OpenMLFlow):
                        return False
                return True

            # _flow is openml flow object, _param dict maps from flow name to flow
            # id for the main call, the param dict can be overridden (useful for
            # unit tests / sentinels) this way, for flows without subflows we do
            # not have to rely on _flow_dict
            exp_parameters = set(_flow.parameters)
            exp_components = set(_flow.components)

            _model_parameters = self._get_parameters(component_model)

            model_parameters = set(_model_parameters.keys())
            if len((exp_parameters | exp_components) ^ model_parameters) != 0:
                flow_params = sorted(exp_parameters | exp_components)
                model_params = sorted(model_parameters)
                raise ValueError('Parameters of the model do not match the '
                                 'parameters expected by the '
                                 'flow:\nexpected flow parameters: '
                                 '%s\nmodel parameters: %s' % (flow_params,
                                                               model_params))

            _params = []
            for _param_name in _flow.parameters:
                _current = OrderedDict()
                _current['oml:name'] = _param_name

                current_param_values = self.model_to_flow(_model_parameters[_param_name])

                # Try to filter out components (a.k.a. subflows) which are
                # handled further down in the code (by recursively calling
                # this function)!
                if isinstance(current_param_values, openml.flows.OpenMLFlow):
                    continue

                if is_subcomponent_specification(current_param_values):
                    # complex parameter value, with subcomponents
                    parsed_values = list()
                    for subcomponent in current_param_values:
                        # keras stores usually tuples in the form
                        # (name (str), subcomponent (mixed), argument
                        # (mixed)). OpenML replaces the subcomponent by an
                        # OpenMLFlow object.
                        if len(subcomponent) < 2 or len(subcomponent) > 3:
                            raise ValueError('Component reference should be '
                                             'size {2,3}. ')

                        subcomponent_identifier = subcomponent[0]
                        subcomponent_flow = subcomponent[1]
                        if not isinstance(subcomponent_identifier, str):
                            raise TypeError('Subcomponent identifier should be '
                                            'string')
                        if not isinstance(subcomponent_flow,
                                          openml.flows.OpenMLFlow):
                            raise TypeError('Subcomponent flow should be string')

                        current = {
                            "oml-python:serialized_object": "component_reference",
                            "value": {
                                "key": subcomponent_identifier,
                                "step_name": subcomponent_identifier
                            }
                        }
                        if len(subcomponent) == 3:
                            if not isinstance(subcomponent[2], list):
                                raise TypeError('Subcomponent argument should be'
                                                'list')
                            current['value']['argument_1'] = subcomponent[2]
                        parsed_values.append(current)
                    parsed_values = json.dumps(parsed_values)
                else:
                    # vanilla parameter value
                    parsed_values = json.dumps(current_param_values)

                _current['oml:value'] = parsed_values
                if _main_call:
                    _current['oml:component'] = main_id
                else:
                    _current['oml:component'] = _flow_dict[_flow.name]
                _params.append(_current)

            for _identifier in _flow.components:
                subcomponent_model = self._get_parameters(component_model)[_identifier]
                _params.extend(extract_parameters(_flow.components[_identifier],
                                                  _flow_dict, subcomponent_model))
            return _params

        flow_dict = get_flow_dict(flow)
        model = model if model is not None else flow.model
        parameters = extract_parameters(flow, flow_dict, model, True, flow.flow_id)

        return parameters

    def instantiate_model_from_hpo_class(
            self,
            model: Any,
            trace_iteration: OpenMLTraceIteration,
    ) -> Any:
        """Instantiate a ``base_estimator`` which can be searched over by the hyperparameter
        optimization model.

        Parameters
        ----------
        model : Any
            A hyperparameter optimization model which defines the model to be instantiated.
        trace_iteration : OpenMLTraceIteration
            Describing the hyperparameter settings to instantiate.

        Returns
        -------
        Any
        """

        return model


register_extension(OnnxExtension)
