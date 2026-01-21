import numpy as np
import scipy.interpolate
import plotly.graph_objects as go


class Track:

    def __init__(self, Q, X, t):
        self.Q = Q
        self.X = X
        self.t = t

        # [x, y, z, theta, mu, phi, n_l, n_r]
        self.poly = []

        self.length = t[-1]

        for k in range(len(Q)):

            # Useful values for conversion between t and tau
            norm_factor = (t[k + 1] - t[k]) / 2
            t_tau_0 = (t[k + 1] + t[k]) / 2  # Global time t at tau = 0

            # Number of collocation points
            N_k = len(Q[k]) - 2
            tau, _ = np.polynomial.legendre.leggauss(N_k)
            tau = np.asarray([-1] + list(tau) + [1])
            self.poly.append(
                scipy.interpolate.BarycentricInterpolator(
                    tau, np.column_stack([X[k], Q[k]])
                )
            )

    



    def state(self, s):
        s = s % self.length
        k = np.searchsorted(self.t[1:], s)

        return np.asarray([self.poly[i](self.t_to_tau(s_i, i)) for s_i, i in zip(s, k)])
    
    def __call__(self, s):
        state = self.state(s)
        b_l, b_r = self._find_boundaries(state)

        return np.column_stack([b_l, b_r, state[:, :3]])


    def plot_uniform(self, approx_spacing):
        s = np.linspace(0, self.length, int(self.length // approx_spacing))
        points = self(s)

        plots = []

        plots.append(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], name="left", mode='lines'))
        plots.append(go.Scatter3d(x=points[:, 3], y=points[:, 4], z=points[:, 5], name="right", mode='lines'))
        plots.append(go.Scatter3d(x=points[:, 6], y=points[:, 7], z=points[:, 8], name="center", mode='lines'))

        return plots


    def _find_boundaries(self, state):

        # State is in the form [[x, y, z, theta, mu, phi, n_l, n_r], ...]
        x = state[:, :3]
        theta = state[:, 3]
        mu = state[:, 4]
        phi = state[:, 5]
        n_l = state[:, 6]
        n_r = state[:, 7]

        n = np.column_stack([
            np.cos(theta) * np.sin(mu) * np.sin(phi) - np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(mu) * np.sin(phi) + np.cos(theta) * np.cos(phi),
            np.cos(mu) * np.sin(phi)]
        )
        # print(n * n_l)
        b_l = x + n * n_l[:, np.newaxis]
        b_r = x + n * n_r[:, np.newaxis]

        return b_l, b_r

    def tau_to_t(self, tau, k):
        norm_factor = (self.t[k + 1] - self.t[k]) / 2
        shift = (self.t[k + 1] + self.t[k]) / 2
        return norm_factor * tau + shift

    def t_to_tau(self, t, k):
        norm_factor = 2 / (self.t[k + 1] - self.t[k])
        shift = (self.t[k + 1] + self.t[k]) / (self.t[k + 1] - self.t[k])
        return norm_factor * t - shift
