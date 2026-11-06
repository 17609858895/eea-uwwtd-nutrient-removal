"""Algebra and support-boundary checks; no data downloads or model fitting."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis/scripts'))
from causal_core import joint_support, pseudo_outcome, weighted_se

# Strict positivity boundaries and the observed-country condition are separate.
m = joint_support([.05, .5, .95, .5, .5], [.5, .5, .5, .05, .5],
                  [.5] * 5, [True, True, True, True, False])
assert m.tolist() == [False, True, False, False, False]

d = pd.DataFrame({'T': [1, 0, 1, 0], 'R': [1, 1, 0, 0], 'Y': [1., 2., np.nan, np.nan]})
nu = {'mu1': np.full(4, .4), 'mu0': np.full(4, .7), 'e': np.full(4, .5),
      'pi1': np.full(4, .5), 'pi0': np.full(4, .5), 'off': np.array([.1, -.2, 0., 0.])}
expected = np.array([1.7, -6.3, -.3, -.3])
np.testing.assert_allclose(pseudo_outcome(d, nu), expected)
np.testing.assert_allclose(weighted_se(np.array([1., 3., 5., 7.]), np.ones(4),
                                      np.array(['A', 'A', 'B', 'B'])), [4., np.sqrt(32.) / 4])
print('Support boundaries, missing-outcome augmentation and country-clustered weighted SE: PASS')
