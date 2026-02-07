from dataclasses import dataclass

# from track_import.track import Track
import pinocchio as pin
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
    sypeak1: 0.087
    sypeak2: 0.1

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
        self.model = pin.Model()
        self.properties = VehicleProperties.load_yaml(config)

        # self.track = Track

        # default_pose =

        # Floating track joint
        self.track_id = self.model.addJoint(
            0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "track"
        )
        # Prismatic defining vehicle position on track
        self.road_lat_id = self.model.addJoint(
            self.track_id, pin.JointModelPY(), pin.SE3.Identity(), "road_lat"
        )
        # Revolute defining vehicle yaw
        self.yaw_id = self.model.addJoint(
            self.road_lat_id, pin.JointModelRZ(), pin.SE3.Identity(), "yaw"
        )
        # Prismatic defining suspension vertical travel
        self.vert_id = self.model.addJoint(
            self.yaw_id, pin.JointModelPZ(), pin.SE3.Identity(), "vert"
        )
        # Revolute defining vehicle pitch
        self.pitch_id = self.model.addJoint(
            self.vert_id, pin.JointModelRY(), pin.SE3.Identity(), "pitch"
        )
        # Revolute defining vehicle roll
        self.roll_id = self.model.addJoint(
            self.pitch_id, pin.JointModelRX(), pin.SE3.Identity(), "roll"
        )

        self.unsprung = pin.Inertia(
            self.properties.m_unsprung, np.zeros(3), np.zeros((3, 3))
        )
        self.sprung = pin.Inertia(
            self.properties.m_sprung,
            np.array([0, 0, self.properties.g_com_h]),
            np.diag(
                [
                    self.properties.i_xx
                    + self.properties.m_sprung * self.properties.g_com_h,
                    self.properties.i_yy
                    + self.properties.m_sprung * self.properties.g_com_h,
                    self.properties.i_zz,
                ]
            ),
        )  # TODO change

        self.model.appendBodyToJoint(self.yaw_id, self.unsprung, pin.SE3.Identity())
        # TODO add offset
        self.model.appendBodyToJoint(self.roll_id, self.sprung, pin.SE3.Identity())


if __name__ == "__main__":
    v = Vehicle("vehicle_properties/DallaraAV24 copy.yaml")
    print(v.model)
    print(v.model.nbodies)
    foo = v.model

    f_ext = [pin.Force.Zero() for _ in range(foo.njoints)]

    f_ext[6] = pin.Force(np.array([0, 0, 7749.9]), np.array([0, 0, 0]))
    data = foo.createData()

    qddot = pin.aba(
        foo, data, pin.neutral(foo), np.zeros(foo.nv), np.zeros(foo.nv), f_ext
    )

    print(qddot)


    torques = pin.rnea(foo, data, pin.neutral(foo), np.zeros(foo.nv), np.zeros(foo.nv), f_ext)
    print(torques)



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
