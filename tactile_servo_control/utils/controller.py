from abc import ABC, abstractmethod
import numpy as np


class Controller(ABC):
    def __init__(self, t0=0, u0=0.0):
        self.t0 = t0
        self.t = t0
        self.u = u0

    @abstractmethod
    def _policy(self, y, r=None):
        # return control signal u
        pass

    def update(self, y, r=None):
        y = np.array(y)
        if r is None:
            r = self.ref
        else:
            r = np.array(r)
        self.u = self._policy(y, r)
        self.t += 1
        return self.u

    def reset(self, t0=0, u0=0.0):
        self.t0 = t0
        self.t = t0
        self.u = u0


class PIDController(Controller):
    def __init__(self,
                 t0=0, u0=0.0,
                 kp=0.0, ki=0.0, kd=0.0,
                 ep_clip=[-np.inf, np.inf], ei_clip=[-np.inf, np.inf], ed_clip=[-np.inf, np.inf],
                 alpha=1.0,
                 ref=0.0,
                 error=lambda y, r: r-y,
                 **kwargs
                 ):
        super().__init__(t0=t0, u0=np.array(u0))
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.ep_clip = np.array(ep_clip)
        self.ei_clip = np.array(ei_clip)
        self.ed_clip = np.array(ed_clip)
        self.alpha = np.array(alpha)
        self.ref = np.array(ref)
        if isinstance(error, str):
            error = eval(error)
        self.error = error

        self.ef = 0.0
        self.ef_prev = 0.0
        self.ei = 0.0

        self.y_hist = []
        self.r_hist = []
        self.e_hist = []
        self.ep_hist = []
        self.ei_hist = []
        self.ed_hist = []
        self.ef_hist = []
        self.u_hist = []

    def _policy(self, y, r):
        e = self.error(y, r)
        ep = np.clip(e, *self.ep_clip)

        self.ei += e
        self.ei = np.clip(self.ei, *self.ei_clip)

        if self.t == self.t0:
            self.ef = e
            ed = np.zeros(y.shape)
        else:
            self.ef = (1.0 - self.alpha) * self.ef + self.alpha * e
            ed = self.ef - self.ef_prev
        ed = np.clip(ed, *self.ed_clip)

        u = self.kp * ep + self.ki * self.ei + self.kd * ed

        self.ef_prev = self.ef

        self.y_hist.append(np.copy(y))
        self.r_hist.append(np.copy(r))
        self.e_hist.append(np.copy(e))
        self.ep_hist.append(np.copy(ep))
        self.ei_hist.append(np.copy(self.ei))
        self.ed_hist.append(np.copy(ed))
        self.ef_hist.append(np.copy(self.ef))
        self.u_hist.append(np.copy(u))

        return u

    def history(self):
        return {
            't': range(self.t0, self.t),
            'y': np.array(self.y_hist),
            'r': np.array(self.r_hist),
            'e': np.array(self.e_hist),
            'ep': np.array(self.ep_hist),
            'ei': np.array(self.ei_hist),
            'ed': np.array(self.ed_hist),
            'ef': np.array(self.ef_hist),
            'u': np.array(self.u_hist),
        }

    def reset(self, t0=0, u0=0.0):
        super().reset(t0=t0, u0=np.array(u0))
        self.ef = 0.0
        self.ef_prev = 0.0
        self.ei = 0.0

        self.y_hist = []
        self.r_hist = []
        self.e_hist = []
        self.ep_hist = []
        self.ei_hist = []
        self.ed_hist = []
        self.ef_hist = []
        self.u_hist = []
