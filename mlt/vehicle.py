from dataclasses import dataclass

# from track_import.track import Track
import pinocchio as pin
import numpy as np


@dataclass
class VehicleProperties:
    # Suspension
    s_k: float = 0
    s_c: float = 0


@dataclass
class SetupProperties:
    pass


class Vehicle:
    def __init__(self):
        # self.model = pin.buildModelsFromUrdf("vehicle.urdf", root_joint=pin.JointModelFreeFlyer())
        self.model = pin.Model()
        # self.track = Track

        # default_pose =

        # Floating track joint
        self.track_id = self.model.addJoint(
            0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "track"
        )
        # Prismatic defining vehicle position on track
        self.j2_id = self.model.addJoint(
            self.track_id, pin.JointModelPY(), pin.SE3.Identity(), "j2"
        )
        # Revolute defining vehicle yaw
        self.j3_id = self.model.addJoint(
            self.j2_id, pin.JointModelRZ(), pin.SE3.Identity(), "j3"
        )
        # Prismatic defining suspension vertical travel
        self.j4_id = self.model.addJoint(
            self.j3_id, pin.JointModelPZ(), pin.SE3.Identity(), "j4"
        )
        # Revolute defining vehicle pitch
        self.j5_id = self.model.addJoint(
            self.j4_id, pin.JointModelRY(), pin.SE3.Identity(), "j5"
        )
        # Revolute defining vehicle roll
        self.j6_id = self.model.addJoint(
            self.j5_id, pin.JointModelRX(), pin.SE3.Identity(), "j6"
        )

        self.unsprung = pin.Inertia.Zero()
        self.sprung = pin.Inertia.Zero()        # TODO change

        self.model.appendBodyToJoint(self.j3_id, self.unsprung, pin.SE3.Identity())
        self.model.appendBodyToJoint(self.j6_id, self.sprung, pin.SE3.Identity())  # TODO add offset for VIP -> COM


if __name__ == "__main__":
    v = Vehicle()
    print(v.model)
    print(v.model.nbodies)
    # foo, *x = v.model

    # print(foo)
    # print(foo.njoints)
    # f_ext = [pin.Force.Zero() for _ in range(foo.njoints)]

    # f_ext[6] = pin.Force(np.array([0, 0, -100]), np.array([0, 0, 0]))
    # data = foo.createData()

    # qddot = pin.aba(foo, data, pin.neutral(foo) , np.zeros(foo.nv), np.zeros(foo.nv), f_ext)

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
