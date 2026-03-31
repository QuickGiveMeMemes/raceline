import pinocchio as pin

import numpy as np
from track_import.track import Track
from mlt.vehicle import Vehicle
from mesh_refinement.collocation import PSCollocation
import casadi as ca
from mlt.trajectory import Trajectory
import scipy


class MinCurvatureCollocation(PSCollocation):

    n_q: int = 1

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.track = Track.load(config["track"])
        self.start_t = 0
        self.end_t = 1

    def iteration(self, t: np.ndarray, N: np.ndarray, warm_start: None | Trajectory = None):
        self.opti = ca.Opti()
        self.vehicle = Vehicle(self.config["vehicle_properties"], self.track, self.opti)

        K = len(N)

        # Q, dQ, ddQ are (N_k + 2) x (n_q).
        Q = []  # Array containing Q matrices. q_j = [q2].
        dQ = []
        ddQ = []

        J = 0  # Cost accumulator

        # Constraints for each segment k
        for k in range(K):
            # Generates CasADi variables at collocation points
            if k == 0:
                Q.append(self.opti.variable(N[k] + 2, self.n_q))

            else:
                # Explicitly couples last of previous segment with first of current segment
                # by setting them as the same variable
                Q.append(ca.vertcat(Q[k - 1][-1, :], self.opti.variable(N[k] + 1, self.n_q)))

            dQ.append(self.opti.variable(N[k] + 2, self.n_q))
            ddQ.append(self.opti.variable(N[k] + 2, self.n_q))

            # Generation of LG collocation points
            tau, w = np.polynomial.legendre.leggauss(N[k])  # w is the quadrature weights
            tau = np.asarray([-1] + list(tau) + [1])
            D = PSCollocation.generate_D(tau)  # Differentiation matrix

            # Useful values for conversion between t and tau
            norm_factor = (t[k + 1] - t[k]) / 2
            t_tau_0 = (t[k + 1] + t[k]) / 2  # Global time t at tau = 0
            t_tau = norm_factor * tau + t_tau_0  # Global time (t) at collocation points

            self.opti.subject_to(norm_factor * dQ[k] == ca.mtimes(D, Q[k]))
            self.opti.subject_to(norm_factor * ddQ[k] == ca.mtimes(D, dQ[k]))

            # Continuity constraints
            if k != 0:
                self.opti.subject_to(dQ[k - 1][-1, :] == dQ[k][0, :])
                self.opti.subject_to(ddQ[k - 1][-1, :] == ddQ[k][0, :])

            # Collocati
            for i, q_1 in enumerate(t_tau[:-1]):
                n_l, n_r = self.track.state(np.array([q_1 * self.track.length]))[0][-2:]
                self.opti.subject_to(
                    self.opti.bounded(
                        n_r + max(self.vehicle.prop.g_t) / 2 + self.vehicle.prop.bound_tol,
                        Q[k][i],
                        n_l - max(self.vehicle.prop.g_t) / 2 - self.vehicle.prop.bound_tol,
                    )
                )

            min_curve_raceline = self.track.raceline_casadi(t_tau * self.track.length, Q[k])

            dr = ca.mtimes(D, min_curve_raceline) / norm_factor
            ddr = ca.mtimes(D, dr) / norm_factor

            curvature = ca.vertcat(
                *[ca.norm_2(ca.cross(dr[i + 1, :], ddr[i + 1, :])) / (ca.norm_2(dr[i + 1, :]) ** 3 + 1e-10) for i in range(N[k])]
            ) 

            # Quadrature cost
            for j in range(N[k]):
                lagrange_term = curvature[j, :]**2
                J += norm_factor * w[j] * lagrange_term

        # Periodicity
        self.opti.subject_to(Q[-1][-1, :] == Q[0][0, :])
        self.opti.subject_to(dQ[-1][-1, :] == dQ[0][0, :])

        self.opti.minimize(J)

        ipopt_settings = {
            "print_time": 0,
            "ipopt.sb": "no",
            "ipopt.max_iter": 3000,
            "detect_simple_bounds": True,
            "ipopt.linear_solver": "ma97",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.bound_relax_factor": 1e-6,
            "ipopt.hessian_approximation": "exact",
            "ipopt.tol": 1e-9,
            # "ipopt.hessian_approximation": "limited-memory",
            # "ipopt.limited_memory_max_history": 30,
            # "ipopt.limited_memory_update_type": "bfgs",
            "ipopt.derivative_test": "none",
        }

        # Solve!
        try:
            self.opti.solver("ipopt", ipopt_settings)
        except Exception as e:
            if "ipopt.linear_solver" in ipopt_settings:
                print(
                    f"Could not use solver {ipopt_settings['ipopt.linear_solver']}, using default!"
                )
                ipopt_settings["ipopt.linear_solver"] = "mumps"
                self.opti.solver("ipopt", ipopt_settings)

            else:
                raise e

        print(f"Solving with {self.opti.nx} variables.")
        try:
            sol = self.opti.solve()
            stats = sol.stats()
            print(f"Solve iteration succeeded in {stats['iter_count']} iterations")
        except:
            sol = self.opti.debug
            stats = sol.stats()
            print(f"Solve iteration failed after {stats['iter_count']} iteration...")

        print(f"Final cost: {sol.value(J)}")

        # Collect solution
        Q_sol = [sol.value(seg) for seg in Q]
        dQ_sol = [sol.value(seg) for seg in dQ]
        ddQ_sol = [sol.value(seg) for seg in ddQ]

        return Q_sol, dQ_sol, ddQ_sol
