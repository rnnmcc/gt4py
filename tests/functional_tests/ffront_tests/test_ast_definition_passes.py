# -*- coding: utf-8 -*-
#
# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import ast
import copy
from typing import TypeVar, Any

import numpy as np
from functional.ffront.decorator import field_operator, scan_operator
from functional.ffront.fbuiltins import (
    Dimension,
    DimensionKind,
    Field,
    float64,
)
from functional.iterator.embedded import (
    np_as_located_field,
)


DimsType = TypeVar("DimsType")
DType = TypeVar("DType")

IDim = Dimension("IDim")
JDim = Dimension("JDim")
KDim = Dimension("KDim", DimensionKind.VERTICAL)


class CustomASTPass(ast.NodeTransformer):
    def __init__(self, seed):
        self._seed = seed

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        ...

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        print(f"pre-processing function '{node.name}' according to seed={self._seed}")
        node_modified = self.generic_visit(node)
        self.modified = node_modified
        return node_modified


def analyze_results(result: np.array):
    pass


def test_modify_fieldop():
    size = 10
    inp = np_as_located_field(IDim)(np.ones((size)))
    result = np_as_located_field(IDim)(np.zeros((size)))

    def original_fieldop(inp: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return inp

    for seed in range(5):
        pre_processor1 = CustomASTPass(seed)
        pre_processor2 = CustomASTPass(seed)
        pre_processor3 = CustomASTPass(seed)

        modified_fieldop = field_operator(original_fieldop, py_ast_passes=[pre_processor1, pre_processor2, pre_processor3])
        modified_fieldop(inp, out=result, offset_provider={})

        analyze_results(result)

    assert np.allclose(inp, result)


def test_modify_fieldop():
    size = 10
    inp = np_as_located_field(KDim)(np.ones((size)))
    result = np_as_located_field(KDim)(np.zeros((size)))

    def original_scanop(_: float, value: float) -> float:
        return value

    for seed in range(5):
        pre_processor1 = CustomASTPass(seed)
        pre_processor2 = CustomASTPass(seed)
        pre_processor3 = CustomASTPass(seed)

        modified_scanop = scan_operator(
            original_scanop,
            axis=KDim,
            forward=True,
            init=0,
            py_ast_passes=[pre_processor1, pre_processor2, pre_processor3]
        )
        modified_scanop(inp, out=result, offset_provider={})

        analyze_results(result)

    assert np.allclose(inp, result)