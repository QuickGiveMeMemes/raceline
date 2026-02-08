from dataclasses import dataclass

# from track_import.track import Track
import casadi as ca
from pinocchio import casadi as cpin
import numpy as np
import yaml


@dataclass
class VehicleProperties:
    # Car properties sheet for the Dallara AV-24. Values are taken from Autonoma AWSIM + PAIRSim parameters.

    # Mass properties
    m_sprung: 750  # Sprung mass
    m_unsprung: 40  # Unsprung mass

    # Inertia properties
    i_xx: 550  # Roll inertia
    i_yy: 800  # Pitch inertia
    i_zz: 265  # Yaw inertia

    # Geometry properties
    g_com_h: 0.2  # COM height
    g_a1: 1.7  # Front axle to COM
    g_a2: 1.3  # Rear axle to COM
    g_t1: 1.676  # Front track width
    g_t2: 1.58  # Rear track width
    g_S: 1  # Frontal area
    g_hq1: 0.06  # Front Roll Center (guess)
    g_hq2: 0.12  # Rear Roll Center  (guess)

    # Aero properties
    a_Cx: 0.8581  # Drag coeff
    a_Cz1: 0.65  # Downforce coeff (front)
    a_Cz2: 1.18  # Downforce coeff (rear)

    # Suspension
    s_k11: 2.0e5  # Spring stiffness FL
    s_k12: 2.0e5  # Spring stiffness FR
    s_k21: 2.0e5  # Spring stiffness RL
    s_k22: 2.0e5  # Spring stiffness RR
    s_c11: 8.0e3  # Damping coeff FL
    s_c12: 8.0e3  # Damping coeff FR
    s_c21: 8.0e3  # Damping coeff RL
    s_c22: 8.0e3  # Damping coeff RR

    # Tire
    t_rw1: 0.3  # Front tire radius
    t_rw2: 0.326  # Rear tire radius
    t_Dy_1: 1.5
    t_Dy_2: 1.55
    t_Dy2_1: -0.2
    t_Dy2_2: -0.2
    t_Cy1: 1.5
    t_Cy2: 1.5
    t_sypeak1: 0.087
    t_sypeak2: 0.079
    t_Fznom1: 1700
    t_Fznom2: 2200
    t_Ey1: 0
    t_Ey2: 0

    # Setup
    p_kb: 0.5  # Brake Bias
    p_karb1: 436593  # Front ARB stiffness
    p_karb2: 0  # Rear ARB stiffness

    @staticmethod
    def load_yaml(config):
        with open(config, "r") as f:
            all_things = yaml.safe_load(f)

        print(all_things)

        return VehicleProperties(**all_things)


class Vehicle:
    def __init__(self, config):
        # self.model = pin.buildModelsFromUrdf("vehicle.urdf", root_joint=pin.JointModelFreeFlyer())
        self.model = cpin.Model()
        self.properties = VehicleProperties.load_yaml(config)

        # self.track = Track

        # default_pose =

        # Floating track joint
        self.track_id = self.model.addJoint(
            0, cpin.JointModelFreeFlyer(), cpin.SE3.Identity(), "track"
        )
        # Prismatic defining vehicle position on track
        self.road_lat_id = self.model.addJoint(
            self.track_id, cpin.JointModelPY(), cpin.SE3.Identity(), "road_lat"
        )
        # Revolute defining vehicle yaw
        self.yaw_id = self.model.addJoint(
            self.road_lat_id, cpin.JointModelRZ(), cpin.SE3.Identity(), "yaw"
        )
        # Prismatic defining suspension vertical travel
        self.vert_id = self.model.addJoint(
            self.yaw_id, cpin.JointModelPZ(), cpin.SE3.Identity(), "vert"
        )
        # Revolute defining vehicle pitch
        self.pitch_id = self.model.addJoint(
            self.vert_id, cpin.JointModelRY(), cpin.SE3.Identity(), "pitch"
        )
        # Revolute defining vehicle roll
        self.roll_id = self.model.addJoint(
            self.pitch_id, cpin.JointModelRX(), cpin.SE3.Identity(), "roll"
        )

        self.unsprung = cpin.Inertia(
            self.properties.m_unsprung, np.zeros(3), np.zeros((3, 3))
        )
        self.sprung = cpin.Inertia(
            self.properties.m_sprung,
            np.array([0, 0, self.properties.g_com_h]),
            np.diag(
                [
                    self.properties.i_xx,
                    self.properties.i_yy,
                    self.properties.i_zz,
                ]
            ),
        )

        self.model.appendBodyToJoint(self.yaw_id, self.unsprung, pin.SE3.Identity())
        self.model.appendBodyToJoint(
            self.roll_id,
            self.sprung,
            cpin.SE3(np.eye(3), np.array([0, 0, self.properties.g_com_h])),
        )

        self.data = self.model.createData()

    def rnea(self, q: np.ndarray, v: np.ndarray, a: np.ndarray, f_ext: list[cpin.Force]) -> tuple[np.ndarray, tuple[float, float, float]]:
        torques = cpin.rnea(self.model, self.data, q, v, a, f_ext)

        return torques, (self.data.f[3].linear[3], self.data.f[3].angular[0], self.data.f[3].angular[1])

    


if __name__ == "__main__":
    v = Vehicle("vehicle_properties/DallaraAV24.yaml")
    print(v.model)
    print(v.model.nbodies)
    foo = v.model

    print(cpin.neutral(foo))
    f_ext = [cpin.Force.Zero() for _ in range(foo.njoints)]

    f_ext[6] = cpin.Force(np.array([0, 0, -1000]), np.array([0, 0, 0]))
    data = foo.createData()

    torques = cpin.rnea(
        foo, data, cpin.neutral(foo), np.zeros(foo.nv), np.zeros(foo.nv), f_ext
    )
    print(torques)
    print(data.f[3])

    # for i in range(foo.njoints):
    #     print(f"Index {i}: {foo.names[i]}, mass={foo.inertias[i].mass:.4f}")
    # print(qddot, type(qddot))

    # for i in range(foo.njoints):
    #     name = foo.names[i]
    #     pos = data.oMi[i].translation
    #     rot = data.oMi[i].rotation

    #     print(f"Joint {i} ({name}):")
    #     print(f"  Position: {pos}")
    #     print(f"  Rotation:\n{rot}\n")
    # # print(data.v.linear, data.v.angular)
    # # print(data.a.linear, data.a.angular)

    # pin.computeAllTerms(foo, data, np.zeros(foo.nq), np.zeros(foo.nv),)
    # print(data.M)
