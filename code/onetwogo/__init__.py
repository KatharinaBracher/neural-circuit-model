from typing import Dict
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from .functions import sigmoid


class Params:
    Wui: float
    """Weight of input to u"""
    Wvi: float
    """Weight of input to v"""
    Wuv: float
    """Weight from v to u"""
    Wvu: float
    """Weight from u to v"""
    dt: float  # int?
    """time step for simulation"""
    tau: int
    """time constant"""
    sigma: float
    """indicates the noise within each trial in u,v and y"""
    th: float
    """threshold for behavior, used to determine to determine rproduction"""
    IF: int
    """Impulse that functions as reset of u and v"""
    ntrials: int
    """number of parallel trials in paralell simulation or number of consecutive trials in experiment simulation"""
    uinit: float
    """initial value of u"""
    vinit: float
    """initial value of v"""
    yinit: float
    """initial value of y"""
    Iinit: float
    """initial value of Input I"""
    first_duration: int
    """duration first interval (ms) before trial or experiment starts"""
    delay: int
    """duration of interval (ms) between production phase and new stimulus in experiment simulation"""

    def __init__(
        self,
        Wui=6,
        Wuv=6,
        Wvi=6,
        Wvu=6,
        dt=10,
        tau=100,
        sigma=0.01,
        th=0.65,
        IF=50,
        ntrials=100,
        uinit=0.7,
        vinit=0.2,
        yinit=0.5,
        Iinit=0.8,
        first_duration=750,
        delay=500
    ):
        self.Wui = Wui
        self.Wuv = Wuv
        self.Wvi = Wvi
        self.Wvu = Wvu
        self.dt = dt
        self.tau = tau
        self.sigma = sigma
        self.th = th
        self.IF = IF
        self.ntrials = ntrials
        self.uinit = uinit
        self.vinit = vinit
        self.yinit = yinit
        self.Iinit = Iinit
        self.first_duration = first_duration
        self.delay = delay

    def from_dict(PARAMS_DICT: Dict[str, float]):
        return Params(**PARAMS_DICT)


class BaseSimulation:
    """
    A class used to compute the simulation of u, v, y, I for a stage (measurement, production delay) 
    and update the previous stages with the newly computed one


    Attributes
    ----------
    state_init: array
        inital values of u, v, y, I
    reset: bool
        1 if u, y are reseted
    K: int
        memory parameter, weights the update of I
    nbin: int
        number of steps of current duration
    """

    def __init__(self, params: Params):
        self.params = params

    def network(self, state_init, reset, K, nbin):
        """returns the simulation of u, v, y, I over nbins and a list of booleans when a reset happend in the trial"""

        params = self.params
        
        #IF_reset = params.IF
        #high_regime=True
        #if high_regime:
            #IF_reset = -params.IF

        u, v, y, Input = state_init.copy()
        ntrials = u.shape[0]

        # save bool with resets=1 over time
        reset_lst = []
        simulation = np.zeros([nbin, 4, ntrials])

        for i in range(nbin):
            Input += (reset * K * (y - params.th)) / params.tau * params.dt
            u += (-u + sigmoid(params.Wui * Input - params.Wuv * v - params.IF * reset +
                  np.random.randn(ntrials) * params.sigma)) / params.tau * params.dt
            v += (-v + sigmoid(params.Wvi * Input - params.Wvu * u + params.IF * reset + # does *2 make sense here?
                  np.random.randn(ntrials) * params.sigma)) / params.tau * params.dt
            y += (-y + u - v + np.random.randn(ntrials)
                  * params.sigma) / params.tau * params.dt

            simulation[i] = [u.copy(), v.copy(), y.copy(), Input.copy()]
            reset_lst.append(reset)

        return simulation, reset_lst

    def trial_update(self, simulation, reset_lst, reset, K, nbin, production_step=False):
        """returns the simulation and list of resets extended with the new stage that was computed by the network"""

        # get prev I,v,u,y to continue trial or experiment
        state_init = simulation[-1]
        # next step simulation
        simulation2, reset_lst2 = self.network(state_init, reset, K, nbin)

        # 0.4 of stim duration considered early phase
        earlyphase = int(0.2*nbin/2)

        if production_step:
            return self.production_step(
                simulation, reset_lst, simulation2, reset_lst2, earlyphase)

        # for all stages exept production
        simulation = np.concatenate((simulation, simulation2))
        reset_lst.extend(reset_lst2)
        return simulation, reset_lst

    def production_step(_simulation, _reset_lst, _simulation2, _reset_lst2, _earlyphase):
        '''place holder function, prodcution step is handed over by parallel_simulation or experiment_simulation'''
        pass
