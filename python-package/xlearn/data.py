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

import numpy as np
from numpy import ndarray

from ._core import DMatrix as CoreDMatrix
from .base import DataFrame, Series, issparse

# This class is the xLearn core data
class DMatrix(object):
    def __init__(self, data, label=None, field_map=None):
        """
        Initial function.
        Parameters:
        data: NumPy 2D, pandas DataFrame or SciPy sparse matrix of features data.
        label: one-dimensional array, it presents samples label.
        field_map: one-dimensional array, it presents the features'field respectively.
        This field_map like, [1, 2, 1, 3] means, the first and third features belong to field one, and the second belongs to field two, and so on.
        this parameter only useful for ffm model.
        Note: we only do roughly check, and do detail check in true work function.
        """

        self.__handle = None
        if (isinstance(data, ndarray) or isinstance(data, DataFrame)):
            self._init_from_npy2d(data, label, field_map)
        elif issparse(data):
            self._init_from_sparse(data, label, field_map)
        else:
            raise ValueError('Input data must be numpy.ndarray, pandas.DataFrame or scipy.sparse matrix')

    def _init_from_npy2d(self, mat, label, field_map):
        """
        This function do initialize DMatrix from numpy 2D and pandas DataFrame.
        Parameters
        """
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')

        if isinstance(mat, DataFrame):
            mat = mat.values

        data = np.ascontiguousarray(mat, dtype=np.float32)
        labels = self._check_label(label, mat.shape[0])
        fields = self._check_field_map(field_map, mat.shape[1])

        self.__handle = CoreDMatrix(data, labels, fields)

    def _init_from_sparse(self, mat, label, field_map):
        """
        This function do initialize DMatrix from a SciPy sparse matrix.
        Parameters
        """
        if len(mat.shape) != 2:
            raise ValueError('Input scipy.sparse matrix must be 2 dimensional')

        csr = mat.tocsr()
        labels = self._check_label(label, csr.shape[0])
        fields = self._check_field_map(field_map, csr.shape[1])

        self.__handle = CoreDMatrix(
            values=np.ascontiguousarray(csr.data, dtype=np.float32),
            indices=np.ascontiguousarray(csr.indices, dtype=np.uint32),
            indptr=np.ascontiguousarray(csr.indptr, dtype=np.uint32),
            label=labels,
            field_map=fields)

    def _check_label(self, label, num_row):
        if label is None:
            return None

        if isinstance(label, DataFrame):
            label = label.values
        if isinstance(label, Series):
            label = label.values
        if isinstance(label, list):
            label = np.array(label)
        if not isinstance(label, ndarray):
            raise ValueError('Input label must be numpy.ndarray')

        if (len(label.shape) > 2):
            raise ValueError('Input numpy.ndarray of label must be 1 dimensional or 2 dimensional with one dimensional is 1')
        if (len(label.shape) == 2) and (label.shape[0] != 1) and (label.shape[1] != 1):
            raise ValueError('Input numpy.ndarray of label must be 1 dimensional or 2 dimensional with one dimensional is 1')
        if (label.size != num_row):
            raise ValueError('Input label must has same elements as the data lines')

        return np.ascontiguousarray(label.reshape(label.size), dtype=np.float32)

    def _check_field_map(self, field_map, num_col):
        if field_map is None:
            return None

        if isinstance(field_map, DataFrame):
            field_map = field_map.values
        if isinstance(field_map, Series):
            field_map = field_map.values
        if isinstance(field_map, list):
            field_map = np.array(field_map)
        if not isinstance(field_map, ndarray):
            raise ValueError('Input of field_map must numpy.ndarray')

        if (len(field_map.shape) > 2):
            raise ValueError('Input numpy.ndarray of label must be 1 dimensional or 2 dimensional with one dimensional is 1')
        if (len(field_map.shape) == 2) and (field_map.shape[0] != 1) and (field_map.shape[1] != 1):
            raise ValueError('Input numpy.ndarray of field_map must be 1 dimensional or 2 dimensional with the one dimensional is 1')
        if (field_map.size != num_col):
            raise ValueError('Input field_map must has same elements as the data columns')

        return np.ascontiguousarray(field_map.reshape(field_map.size), dtype=np.uint32)

    @property
    def handle(self):
        return self.__handle
