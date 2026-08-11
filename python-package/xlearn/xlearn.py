# Copyright (c) 2018 by contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
from typing import Any, Optional, Union

from numpy import ndarray

from ._core import Model, hello
from .data import DMatrix

__all__ = ['XLearn', 'DMatrix', 'create_linear', 'create_fm', 'create_ffm',
           'hello']

# Every hyper-parameter the 'param' dict accepts is a property of the same
# name on the core model, except for the one whose name Python reserves.
_PARAM_KEYS = frozenset((
    'task', 'metric', 'opt', 'log', 'lr', 'k', 'lambda', 'init', 'epoch',
    'fold', 'alpha', 'beta', 'lambda_1', 'lambda_2', 'nthread', 'block_size',
    'stop_window', 'seed',
))

_PARAM_ALIASES = {'lambda': 'regu_lambda'}

class XLearn(object):
    """XLearn is the core interface used by python API."""

    def __init__(self, handle: Model) -> None:
        """Initalizes a new XLearn

        Parameters
        ----------
        handle : Model
            'XLearn' handle of the core module.
        """
        assert isinstance(handle, Model)
        self.handle = handle

    def _set_Param(self, param: dict[str, Any]) -> None:
        """Set hyper-parameter for xlearn handle

        Parameters
        ----------
        param : dict
            xlearn hyper-parameters
        """
        for (key, value) in param.items():
            if key not in _PARAM_KEYS:
                raise Exception("Invalid key!", key)
            setattr(self.handle, _PARAM_ALIASES.get(key, key), value)

    def show(self) -> None:
        """Show model information
        """
        self.handle.show()

    def setTrain(self, train_path: Union[str, DMatrix]) -> None:
        """Set file path of training data.

        Parameters
        ----------
        train_path : str
           the path of training data
        """
        if isinstance(train_path, str):
            self.handle.set_train_file(train_path)
        elif isinstance(train_path, DMatrix):
            self.handle.set_train_data(train_path.handle)
        else:
            raise Exception("Invalid train.Can be test file path or xLearn DMatrix", type(train_path))

    def setTest(self, test_path: Union[str, DMatrix]) -> None:
        """Set file path of test data.

        Parameters
        ----------
        test_path : str
           the path of test data.
        """
        if isinstance(test_path, str):
            self.handle.set_test_file(test_path)
        elif isinstance(test_path, DMatrix):
            self.handle.set_test_data(test_path.handle)
        else:
            raise Exception("Invalid test.Can be test file path or xLearn DMatrix", type(test_path))

    def setPreModel(self, pre_model_path: str) -> None:
        """ Set file path of pre-trained model.

        Parameters
        ----------
        pre_model_path : str
           the path of pre-trained model.
        """
        self.handle.set_pre_model(pre_model_path)

    def setValidate(self, val_path: Union[str, DMatrix]) -> None:
        """Set file path of validation data.

        Parameters
        ----------
        val_path : str
           the path of validation data.
        """
        if isinstance(val_path, str):
            self.handle.set_validate_file(val_path)
        elif isinstance(val_path, DMatrix):
            self.handle.set_validate_data(val_path.handle)
        else:
            raise Exception("Invalid validation.Can be test file path or xLearn DMatrix", type(val_path))

    def setTXTModel(self, model_path: str) -> None:
        """Set the path of TXT model file.

        Parameters
        ----------
        model_path : str
            the path of the TXT model file.
        """
        self.handle.set_txt_model(model_path)

    def setQuiet(self) -> None:
        """Set xlearn to quiet model"""
        self.handle.quiet = True

    def setOnDisk(self) -> None:
        """Set xlearn to use on-disk training"""
        self.handle.on_disk = True

    def setNoBin(self) -> None:
        """Do not generate bin file"""
        self.handle.bin_out = False

    def disableNorm(self) -> None:
        """Disable instance-wise normalization"""
        self.handle.norm = False

    def disableLockFree(self) -> None:
        """Disable lock free training"""
        self.handle.lock_free = False

    def disableEarlyStop(self) -> None:
        """Disable early-stopping"""
        self.handle.early_stop = False

    def setSign(self) -> None:
        """Convert output to 0 and 1"""
        self.handle.sign = True

    def setSigmoid(self) -> None:
        """Convert output by using sigmoid"""
        self.handle.sigmoid = True

    def fit(self, param: dict[str, Any], model_path: str) -> None:
        """Check hyper-parameters, train model, and dump model.

        Parameters
        ----------
        param : dict
          hyper-parameter used by xlearn.
        model_path : str
          path of model checkpoint.
        """
        self._set_Param(param)
        self.handle.fit(model_path)

    def cv(self, param: dict[str, Any]) -> None:
        """ Do cross-validation

        Parameters
        ----------
        param : dict
          hyper-parameter used by xlearn
        """
        self._set_Param(param)
        self.handle.cv()

    def predict(self, model_path: str,
                out_path: Optional[str] = None) -> Optional[ndarray]:
        """Predict output

        Parameters
        ----------
        model_path : str. path of model checkpoint.
        out_path : str, default None. if a path of output result is set, then will save result to local file,
        and will not return numpy res.
        """
        if out_path is None:
            return self.handle.predict(model_path)
        else:
            self.handle.predict_to_file(model_path, out_path)

def create_linear() -> XLearn:
    """
    Create a linear model.
    """
    return XLearn(Model('linear'))


def create_fm() -> XLearn:
    """
    Create a factorization machine.
    """
    return XLearn(Model('fm'))


def create_ffm() -> XLearn:
    """
    Create a field-aware factorization machine.
    """
    return XLearn(Model('ffm'))
