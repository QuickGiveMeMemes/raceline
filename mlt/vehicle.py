import pinocchio as pin


class Vehicle:
    def __init__(self):
        self.model = pin.buildModelsFromUrdf("vehicle.urdf", root_joint=pin.JointModelFreeFlyer())