from typing import Callable, override
import casadi as ca

class PacejkaModel:
    """
    A CasADi-compatible class that models the lateral Pacejka forces
    """

    def __init__(self, By: Callable, Cy: Callable, Dy: Callable, Ey: Callable):
        self.By = By
        self.Cy = Cy
        self.Dy = Dy
        self.Ey = Ey

    def __call__(self, alpha, fz):
        return self.Dy(fz) * ca.sin(self.Cy(fz) * ca.arctan(self.By(fz) * alpha - self.Ey(fz) * (self.By(fz) * alpha - ca.arctan(self.By(fz) * alpha))))

class AWSIMPacejka(PacejkaModel):
    """
    An extension of the Pacejka class that specifically uses the AWSIM simplified Pacejka model
    """

    @override
    def __init__(self, By: float, Cy: float, Dy: float, Dy2: float, Fznom: float, Ey: float):
        super().__init__(
            lambda _: By,
            lambda _: Cy,
            lambda fz: fz * (Dy + Dy2 * (fz - Fznom) / Fznom),
            lambda _: Ey
        )