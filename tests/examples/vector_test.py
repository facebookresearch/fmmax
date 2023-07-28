"""Tests for `examples.vector`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

from examples import vector


class PlotFieldTest(unittest.TestCase):
    def test_generate_figure(self):
        vector.plot_vector_fields(savefig=False)
