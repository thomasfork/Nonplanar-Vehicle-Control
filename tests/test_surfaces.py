''' test global to parametric position conversion '''
import time
import unittest

import numpy as np

from vehicle_3d.pytypes import BaseBodyState
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.surfaces.utils import load_surface, get_available_surfaces

EXACT_POSE_DIST_TOL = 1e-3
EXACT_POSE_TIME_MAX = 1e-2
APPROX_POSE_DIST_TOL = 1e0
APPROX_POSE_TIME_MAX = 1e-3

class TestSurfaces(unittest.TestCase):
    ''' test surfaces '''
    def test_surfaces(self):
        ''' run tests on all surfaces '''
        print('')
        for surf_name in get_available_surfaces():
            print(surf_name)
            self._test_surf(load_surface(surf_name))

    def _test_surf(self, surf: BaseSurface):
        self._x2p_test(surf)

    def _x2p_test(self, surf: BaseSurface):
        ''' test computing parametric pose from global position'''

        print('x2p test')

        n = 100

        S = np.random.uniform(low = surf.s_min(), high = surf.s_max(), size = n)
        Y = np.random.uniform(low = surf.y_min(S), high = surf.y_max(S), size = n)
        N = np.random.uniform(low = -1, high = 1, size = n)

        P = np.array((S,Y,N)).T
        X = surf.p2x(S[None], Y[None], N[None]).T

        state = BaseBodyState()

        exact_errs = []
        t0 = time.time()
        for p,x in zip(P,X):
            state.x.from_vec(x)
            surf.g2px(state, exact=True)
            exact_errs.append(np.linalg.norm(p - state.p.to_vec()))
        exact_time = time.time() - t0

        approx_errs = []
        t0 = time.time()
        for p,x in zip(P,X):
            state.x.from_vec(x)
            surf.g2px(state, exact=False)
            approx_errs.append(np.linalg.norm(p - state.p.to_vec()))
        approx_time = time.time() - t0

        print(f'{n} calculations')
        print(f'{exact_time:0.2f} seconds for exact method')
        print(f'{approx_time:0.2f} seconds for approximate method')
        print(f'{np.mean(exact_errs)} mean exact error')
        print(f'{np.mean(approx_errs)} mean approximate error')
        print('')

        self.assertTrue(np.sum(np.array(exact_errs)>= EXACT_POSE_DIST_TOL) <= 1)
        self.assertTrue(exact_time <= EXACT_POSE_TIME_MAX * n)
        self.assertTrue(np.mean(approx_errs) <= APPROX_POSE_DIST_TOL)
        self.assertTrue(approx_time <= APPROX_POSE_TIME_MAX * n)

if __name__ == '__main__':
    unittest.main(verbosity=2, exit = False)
