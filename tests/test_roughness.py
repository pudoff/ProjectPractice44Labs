import unittest

import numpy as np
from runs.klapan_png2dxf.pipeline_b.pipeline import roughness_marks

class TestRoughness(unittest.TestCase):
    def test_touching_lines_sixty_degrees(self):
        a = np.array([0.,0.])
        first = {"p": a, "q": np.array([3.,0.])}
        second = {"p": a, "q": np.array([4., 6.92820323])}
        far = {"p": np.array([100.,100.]), "q": np.array([104.,100.])}
        self.assertEqual(roughness_marks([first, second, far]), {0,1})
    def test_empty(self):
        self.assertEqual(roughness_marks([]), set())
