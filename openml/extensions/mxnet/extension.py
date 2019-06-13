from collections import OrderedDict  # noqa: F401
import copy
from distutils.version import LooseVersion
import importlib
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings
import zlib

import numpy as np
import pandas as pd
import scipy.sparse
import mxnet
import mxnet.autograd

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


class MXNetExtension(Extension):
    """Connect Apache MXNet to OpenML-Python."""

    ################################################################################################
    # General setup

    @classmethod
    def can_handle_flow(cls, flow: 'OpenMLFlow') -> bool:
        """Check whether a given describes a MXNet block.

        This is done by parsing the ``external_version`` field.

        Parameters
        ----------
        flow : OpenMLFlow

        Returns
        -------
        bool
        """
        return cls._is_mxnet_flow(flow)

    @classmethod
    def can_handle_model(cls, model: Any) -> bool:
        """Check whether a model is an instance of ``mxnet.gluon.nn.HybridBlock``.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        from mxnet.gluon.nn import HybridBlock
        return isinstance(model, HybridBlock)

    ################################################################################################
    # Methods for flow serialization and de-serialization

    def flow_to_model(self, flow: 'OpenMLFlow', initialize_with_defaults: bool = False) -> Any:
        """Initializes a MXNet model based on a flow.

        Parameters
        ----------
        flow : OpenMLFlow
            the object to deserialize (can be flow object, or any serialized
            parameter value that is accepted by)

        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        Returns
        -------
        Any
        """
        if not self._is_mxnet_flow(flow):
            raise ValueError('Only mxnet flows can be reinstantiated')
        return self._deserialize_model(
            flow=flow,
            keep_defaults=initialize_with_defaults,
        )

    def model_to_flow(self, model: Any) -> 'OpenMLFlow':
        """Transform a MXNet model to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """
        # Necessary to make pypy not complain about all the different possible return types
        return self._serialize_mxnet(model)

    def _serialize_mxnet(self, o: Any, parent_model: Optional[Any] = None) -> Any:
        rval = None  # type: Any

        if self.is_estimator(o):
            # is the main model or a submodel
            rval = self._serialize_model(o)
        elif isinstance(o, (list, tuple)):
            rval = [self._serialize_mxnet(element, parent_model) for element in o]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, SIMPLE_TYPES) or o is None:
            if isinstance(o, tuple(SIMPLE_NUMPY_TYPES)):
                o = o.item()
            # base parameter values
            rval = o
        elif isinstance(o, dict):
            if not isinstance(o, OrderedDict):
                o = OrderedDict([(key, value) for key, value in sorted(o.items())])

            rval = OrderedDict()
            for key, value in o.items():
                if not isinstance(key, str):
                    raise TypeError('Can only use string as keys, you passed '
                                    'type %s for value %s.' %
                                    (type(key), str(key)))
                key = self._serialize_mxnet(key, parent_model)
                value = self._serialize_mxnet(value, parent_model)
                rval[key] = value
        else:
            raise TypeError(o, type(o))

        return rval

    def get_version_information(self) -> List[str]:
        """List versions of libraries required by the flow.

        Libraries listed are ``Python``, ``mxnet``, ``numpy`` and ``scipy``.

        Returns
        -------
        List
        """

        # This can possibly be done by a package such as pyxb, but I could not get
        # it to work properly.
        import mxnet
        import scipy
        import numpy

        major, minor, micro, _, _ = sys.version_info
        python_version = 'Python_{}.'.format(
            ".".join([str(major), str(minor), str(micro)]))
        mxnet_version = 'MXNet_{}.'.format(mxnet.__version__)
        numpy_version = 'NumPy_{}.'.format(numpy.__version__)
        scipy_version = 'SciPy_{}.'.format(scipy.__version__)

        return [python_version, mxnet_version, numpy_version, scipy_version]

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
    def _is_mxnet_flow(cls, flow: OpenMLFlow) -> bool:
        return (
            flow.external_version.startswith('mxnet==')
            or ',mxnet==' in flow.external_version
        )

    def _serialize_model(self, model: Any) -> OpenMLFlow:
        """Create an OpenMLFlow.

        Calls `mxnet_to_flow` recursively to properly serialize the
        parameters to strings and the components (other models) to OpenMLFlows.

        Parameters
        ----------
        model : mxnet network

        Returns
        -------
        OpenMLFlow

        """

        # Get all necessary information about the model objects itself
        parameters, parameters_meta_info, subcomponents, subcomponents_explicit = \
            self._extract_information_from_model(model)

        # Create a flow name, which contains a hash of the parameters as part of the name
        # This is done in order to ensure that we are not exceeding the 1024 character limit
        # of the API, since NNs can become quite large
        class_name = model.__module__ + "." + model.__class__.__name__
        class_name += '.' + format(
            zlib.crc32(json.dumps(parameters, sort_keys=True).encode('utf8')),
            'x'
        )

        name = class_name

        # Get the external versions of all sub-components
        external_version = self._get_external_version_string(model, subcomponents)

        dependencies = '\n'.join([
            self._format_external_version(
                'mxnet',
                mxnet.__version__,
            ),
            'numpy>=1.6.1',
            'scipy>=0.9',
        ])

        mxnet_version = self._format_external_version('mxnet', mxnet.__version__)
        mxnet_version_formatted = mxnet_version.replace('==', '_')
        flow = OpenMLFlow(name=name,
                          class_name=class_name,
                          description='Automatically created MXNet flow.',
                          model=model,
                          components=subcomponents,
                          parameters=parameters,
                          parameters_meta_info=parameters_meta_info,
                          external_version=external_version,
                          tags=['openml-python', 'mxnet',
                                'python', mxnet_version_formatted],
                          language='English',
                          dependencies=dependencies)

        return flow

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
        model_package_name = model.__module__.split('.')[0]
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

    def _get_symbolic_configuration(self, model: mxnet.gluon.HybridBlock) \
            -> 'OrderedDict[str, Any]':
        # The dictionary containing the configuration of the model.
        configuration = OrderedDict()  # type: OrderedDict[str, Any]

        # The placeholder input variable, representing an unknown size variable input.
        placeholder_input = mxnet.sym.var('data')
        # The resulting symbolic "formula", after applying the model to the variable.
        symbolic_model = model(placeholder_input)
        # The normal symbolic representation dictionary.
        symbolic_dict = json.loads(symbolic_model.tojson())

        # We extract the nodes in order to represent them as separate parameters.
        nodes = symbolic_dict['nodes']
        del symbolic_dict['nodes']

        # Compute the format of the layer numbering. This pads the layer numbers with 0s in
        # order to ensure that the layers are printed in a human-friendly order, instead of
        # having weird orderings.
        max_len = int(np.ceil(np.log10(len(nodes))))
        len_format = '{0:0>' + str(max_len) + '}'

        # Add the order identifier to each node, in order to maintain the order
        # during reconstruction.
        for idx, node in enumerate(nodes):
            configuration[len_format.format(idx) + "_" + node['name']] = node

        # Add the remaining settings to the resulting dictionary.
        configuration['misc'] = symbolic_dict

        return configuration

    def _from_symbolic_configuration(self, configuration: Dict[str, Any]) \
            -> mxnet.gluon.HybridBlock:
        # Retrieve the main settings from the configuration dictionary.
        symbolic_dict = configuration['misc']
        del configuration['misc']

        # Retrieve the remaining nodes from the dictionary.
        nodes = []
        for k, v in configuration.items():
            nodes.append(v)
        symbolic_dict['nodes'] = nodes

        # Recreate the symbolic representation.
        symbolic_model = mxnet.symbol.load_json(json.dumps(symbolic_dict))

        # Inflate the model as a symbol block.
        return mxnet.gluon.SymbolBlock(
            inputs=mxnet.sym.var('data'),
            outputs=symbolic_model
        )

    def _extract_information_from_model(
        self,
        model: Any,
    ) -> Tuple[
        'OrderedDict[str, Optional[str]]',
        'OrderedDict[str, Optional[Dict]]',
        'OrderedDict[str, OpenMLFlow]',
        Set,
    ]:
        # This function contains four "global" states and is quite long and
        # complicated. If it gets to complicated to ensure it's correctness,
        # it would be best to make it a class with the four "global" states being
        # the class attributes and the if/elif/else in the for-loop calls to
        # separate class methods

        # stores all entities that should become subcomponents
        sub_components = OrderedDict()  # type: OrderedDict[str, OpenMLFlow]
        # stores the keys of all subcomponents that should become
        sub_components_explicit = set()  # type: Set
        parameters = OrderedDict()  # type: OrderedDict[str, Optional[str]]
        parameters_meta_info = OrderedDict()  # type: OrderedDict[str, Optional[Dict]]

        model_parameters = self._get_symbolic_configuration(model)
        for k, v in sorted(model_parameters.items(), key=lambda t: t[0]):
            rval = self._serialize_mxnet(v, model)

            parameters[k] = json.dumps(rval)
            parameters_meta_info[k] = OrderedDict((('description', None), ('data_type', None)))

        return parameters, parameters_meta_info, sub_components, sub_components_explicit

    def _deserialize_model(
        self,
        flow: OpenMLFlow,
        keep_defaults: bool,
    ) -> Any:
        self._check_dependencies(flow.dependencies)

        parameters = flow.parameters
        parameter_dict = OrderedDict()  # type: Dict[str, Any]

        for name in parameters:
            value = parameters.get(name)
            parameter_dict[name] = json.loads(value)

        return self._from_symbolic_configuration(parameter_dict)

    def _check_dependencies(self, dependencies: str) -> None:
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
            True
        """
        return True

    ################################################################################################
    # Methods for performing runs with extension modules

    def is_estimator(self, model: Any) -> bool:
        """Check whether the given model is a MXNet estimator.

        This function is only required for backwards compatibility and will be removed in the
        near future.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        return self.can_handle_model(model)

    def seed_model(self, model: Any, seed: Optional[int] = None) -> Any:
        """Set the random state of all the unseeded components of a model and return the seeded
        model.

        Required so that all seed information can be uploaded to OpenML for reproducible results.

        Models that are already seeded will maintain the seed. In this case,
        only integer seeds are allowed (An exception is raised when a RandomState was used as
        seed).

        Parameters
        ----------
        model : mxnet model
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
            model_classes : list
                List of classes known_predicted by the model, ordered by their index.

            Returns
            -------
            np.ndarray
            """
            # y: list or numpy array of predictions
            # model_classes: classifier mapping from original array id to
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

        model_copy = copy.deepcopy(model)

        from .config import active

        initializer = active.initializer_gen(task)
        if initializer is not None:
            model_copy.collect_params().initialize(initializer)

        # Runtime can be measured if the model is run sequentially
        can_measure_cputime = self._can_measure_cputime(model_copy)
        can_measure_wallclocktime = self._can_measure_wallclocktime(model_copy)

        user_defined_measures = OrderedDict()  # type: 'OrderedDict[str, float]'

        try:
            # for measuring runtime. Only available since Python 3.3
            modelfit_start_cputime = time.process_time()
            modelfit_start_walltime = time.time()

            if isinstance(task, OpenMLSupervisedTask):
                X_train_mxnet = mxnet.nd.array(X_train)
                X_train_mxnet = active.sanitize(X_train_mxnet)
                y_train_mxnet = mxnet.nd.array(y_train)

                train_dataset = mxnet.gluon.data.ArrayDataset(X_train_mxnet, y_train_mxnet)
                train_data = mxnet.gluon.data.DataLoader(train_dataset,
                                                         batch_size=active.batch_size,
                                                         shuffle=True)

                criterion = active.criterion_gen(task)
                lr_scheduler = active.scheduler_gen(task)
                trainer = mxnet.gluon.Trainer(params=model_copy.collect_params(),
                                              optimizer=active.optimizer_gen(lr_scheduler, task))

                iteration_metric = active.metric_gen(task)

                for epoch in range(active.epoch_count):
                    iteration_metric.reset()

                    for i, (data, label) in enumerate(train_data):
                        with mxnet.autograd.record():
                            output = model_copy(data)
                            loss = criterion(output, label)
                            iteration_metric.update(labels=label, preds=output)

                        loss.backward()
                        trainer.step(active.batch_size)

                        active.progress_callback(fold_no, rep_no, epoch, i, loss, iteration_metric)

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
            model_classes = np.amax(y_train)

        modelpredict_start_cputime = time.process_time()
        modelpredict_start_walltime = time.time()

        # In supervised learning this returns the predictions for Y
        if isinstance(task, OpenMLSupervisedTask):
            X_test_mxnet = mxnet.nd.array(X_test)
            X_test_mxnet = active.sanitize(X_test_mxnet)

            pred_y = model_copy(X_test_mxnet)
            pred_y = active.predict(pred_y, task)
            pred_y = pred_y.asnumpy()
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
                X_test_mxnet = mxnet.nd.array(X_test)
                X_test_mxnet = active.sanitize(X_test_mxnet)

                proba_y = model_copy(X_test_mxnet)
                proba_y = active.predict_proba(proba_y)
                proba_y = proba_y.asnumpy()
            except AttributeError:
                if task.class_labels is not None:
                    proba_y = _prediction_to_probabilities(pred_y, list(task.class_labels))
                else:
                    raise ValueError('The task has no class labels')

            if task.class_labels is not None:
                if proba_y.shape[1] != len(task.class_labels):
                    # Remap the probabilities in case there was a class missing
                    # at training time. By default, the classification targets
                    # are mapped to be zero-based indices to the actual classes.
                    # Therefore, the model_classes contain the correct indices to
                    # the correct probability array. Example:
                    # classes in the dataset: 0, 1, 2, 3, 4, 5
                    # classes in the training set: 0, 1, 2, 4, 5
                    # then we need to add a column full of zeros into the probabilities
                    # for class 3 because the rest of the library expects that the
                    # probabilities are ordered the same way as the classes are ordered).
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
            else:
                raise ValueError('The task has no class labels')

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
            # _flow is openml flow object, _param dict maps from flow name to flow
            # id for the main call, the param dict can be overridden (useful for
            # unit tests / sentinels) this way, for flows without subflows we do
            # not have to rely on _flow_dict
            exp_parameters = set(_flow.parameters)
            exp_components = set(_flow.components)
            model_parameters = set([mp for mp in self._get_symbolic_configuration(component_model)
                                    if '__' not in mp])
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

                current_param_values = self.model_to_flow(
                    self._get_symbolic_configuration(component_model)[_param_name]
                )

                # Try to filter out components (a.k.a. subflows) which are
                # handled further down in the code (by recursively calling
                # this function)!
                if isinstance(current_param_values, openml.flows.OpenMLFlow):
                    continue

                # vanilla parameter value
                parsed_values = json.dumps(current_param_values)

                _current['oml:value'] = parsed_values
                if _main_call:
                    _current['oml:component'] = main_id
                else:
                    _current['oml:component'] = _flow_dict[_flow.name]
                _params.append(_current)

            for _identifier in _flow.components:
                subcomponent_model = self._get_symbolic_configuration(component_model)[_identifier]
                _params.extend(extract_parameters(_flow.components[_identifier],
                                                  _flow_dict, subcomponent_model))
            return _params

        flow_dict = get_flow_dict(flow)
        model = model if model is not None else flow.model
        parameters = extract_parameters(flow, flow_dict, model, True, flow.flow_id)

        return parameters

    ################################################################################################
    # Methods for hyperparameter optimization

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


register_extension(MXNetExtension)
