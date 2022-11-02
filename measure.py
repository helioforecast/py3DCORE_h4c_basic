import numpy as np
from py3dcore.models.toroidal import thin_torus_gh, thin_torus_qs, thin_torus_sq

from scipy.optimize import least_squares

def visualize_fieldline(obj, q0, index=0, steps=1000, step_size=0.01):
    
        """Integrates along the magnetic field lines starting at a point q0 in (q) coordinates and
        returns the field lines in (s) coordinates.

        Parameters
        ----------
        q0 : np.ndarray
            Starting point in (q) coordinates.
        index : int, optional
            Model run index, by default 0.
        steps : int, optional
            Number of integration steps, by default 1000.
        step_size : float, optional
            Integration step size, by default 0.01.

        Returns
        -------
        np.ndarray
            Integrated magnetic field lines in (s) coordinates.
        """

        _tva = np.empty((3,), dtype=obj.dtype)
        _tvb = np.empty((3,), dtype=obj.dtype)

        thin_torus_qs(q0, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_xs[index], _tva)

        fl = [np.array(_tva, dtype=obj.dtype)]
        def iterate(s):
            thin_torus_sq(s, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_sx[index],_tva)
            thin_torus_gh(_tva, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_xs[index], _tvb)
            return _tvb / np.linalg.norm(_tvb)

        while len(fl) < steps:
            # use implicit method and least squares for calculating the next step
            try:
                sol = getattr(least_squares(
                    lambda x: x - fl[-1] - step_size *
                    iterate((x.astype(obj.dtype) + fl[-1]) / 2),
                    fl[-1]), "x")

                fl.append(np.array(sol.astype(obj.dtype)))
            except Exception as e:
                print("ERROR")
                break

        fl = np.array(fl, dtype=obj.dtype)

        return fl