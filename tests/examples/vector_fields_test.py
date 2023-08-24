"""Tests for `examples.vector_fields`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

from examples import vector_fields


class PlotFieldTest(unittest.TestCase):
    def test_generate_figure(self):
        vector_fields.plot_vector_fields(savefig=False)
