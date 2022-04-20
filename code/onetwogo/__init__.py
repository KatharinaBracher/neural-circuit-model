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
    K:
        memory parameter, weighs how much input 
    nbin:
        number of steps of current duration


    Methods
    -------
    network(self, state_init, reset, K, nbin)
        returns the simulation of the current over nbins and a list of booleans when a reset happend in the trial

    trial_update(self, simulation, reset_lst, reset, K, nbin, production_step=False)
        returns the simulation and reset list extended with the new stage that was computed by the network  
    """

    def __init__(self, params: Params):
        self.params = params

    def network(self, state_init, reset, K, nbin):
        params = self.params

        u, v, y, Input = state_init.copy()
        ntrials = u.shape[0]

        # save bool with resets=1 over time
        reset_lst = []
        simulation = np.zeros([nbin, 4, ntrials])

        for i in range(nbin):
            Input += (reset * K * (y - params.th)) / params.tau * params.dt
            u += (-u + sigmoid(params.Wui * Input - params.Wuv * v - params.IF * reset +
                  np.random.randn(ntrials) * params.sigma)) / params.tau * params.dt
            v += (-v + sigmoid(params.Wvi * Input - params.Wvu * u + params.IF * reset +
                  np.random.randn(ntrials) * params.sigma)) / params.tau * params.dt
            y += (-y + u - v + np.random.randn(ntrials)
                  * params.sigma) / params.tau * params.dt

            simulation[i] = [u.copy(), v.copy(), y.copy(), Input.copy()]
            reset_lst.append(reset)

        return simulation, reset_lst

    def trial_update(self, simulation, reset_lst, reset, K, nbin, production_step=False):
        # get prev I,v,u,y to continue trial or experiment
        state_init = simulation[-1]
        # next step simulation
        simulation2, reset_lst2 = self.network(state_init, reset, K, nbin)

        # 0.4 of stim duration considered early phase
        earlyphase = int(0.4*nbin/2)

        if production_step:
            return self.production_step(
                simulation, reset_lst, simulation2, reset_lst2, nbin, earlyphase)

        # for all stages exept production
        simulation = np.concatenate((simulation, simulation2))
        reset_lst.extend(reset_lst2)
        return simulation, reset_lst

    def production_step(_simulation, _reset_lst, _simulation2, _reset_lst2, _nbin: int, _earlyphase):
        pass

    ######################################################################################################
    # TODO PCA in class
    def meas_prod_times(self, simu, prod, stimulus, sample):
        meas_start = int(self.first_duration/self.dt+1)
        meas_stop = int(self.first_duration/self.dt+1+stimulus/self.dt)  # without flashes
        measure = simu[meas_start:meas_stop]
        measure = np.mean(measure, 2)

        prod_lst = []
        for i, p in enumerate(prod):
            p_start = int(self.first_duration/self.dt+stimulus/self.dt+2)
            p_end = int(self.first_duration/self.dt+stimulus/self.dt+2+p)

            prod_lst.append(simu[p_start:p_end, :, i])
            # print(simu[p_start:p_end,:,i].shape)

        prod_sampled = [signal.resample(trial, sample) for trial in prod_lst]
        prod_sampled = np.array(prod_sampled)
        prod_sampled = np.mean(prod_sampled, 0)

        return measure, prod_sampled

    def PCA(self, stimuli_range, data=None, K=None, initI=None, experiment=True):
        if experiment:
            # data, has to be sliced according to stimuli and in measure and prod
            # return meas_all prod_all
            # meas_all
            # prod_all
            pass

        else:
            meas_all = []
            prod_all = []
            lengths_m = [0]
            lengths_p = [0]
            for stim in stimuli_range:
                simulation, res, production, timeout_trials = self.simulate_parallel(stim, K, initI)
                measure, prod_sampled = self.meas_prod_times(simulation, production, stim, int(stim/self.dt+10))
                meas_all.extend(measure)
                prod_all.extend(prod_sampled)
                lengths_m.append(measure[:, :3].shape[0])
                lengths_p.append(prod_sampled[:, :3].shape[0])

        pca_m = PCA()
        MPCA = pca_m.fit_transform(meas_all)
        pca_p = PCA()
        PPCA = pca_p.fit_transform(prod_all)

        MPCA_lst = []
        PPCA_lst = []

        for i in range(len(stimuli_range)):
            MPCA_lst.append(MPCA[sum(lengths_m[:i+1]):sum(lengths_m[:i+2])])
            PPCA_lst.append(PPCA[sum(lengths_p[:i+1]):sum(lengths_p[:i+2])])

        return MPCA_lst, PPCA_lst

    def plot_PCA(self, stimuli_range, K, initI, colors, separate=False):
        MPCA_lst, PPCA_lst = self.PCA(stimuli_range, data=None, K=K, initI=initI, experiment=False)
        TOP = [3, 2, 2]
        MIDDLE = [3, 2, 4]
        BOTTOM = [3, 2, 6]

        if separate:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.axes(projection='3d')
        for i in reversed(range(len(stimuli_range))):
            ax.plot3D(MPCA_lst[i][:, 0],  MPCA_lst[i][:, 1],  MPCA_lst[i][:, 2], c=colors[i], alpha=0.5, marker='.')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.title('Measurement')

        if separate:
            ax = fig.add_subplot(*TOP)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(MPCA_lst[i][:, 0], c=c, alpha=0.5, marker='.')
            ax = fig.add_subplot(*MIDDLE)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(MPCA_lst[i][:, 1], c=c, alpha=0.5, marker='.')
            ax = fig.add_subplot(*BOTTOM)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(MPCA_lst[i][:, 2], c=c, alpha=0.5, marker='.')
        plt.tight_layout()

        if separate:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.axes(projection='3d')
        for i in reversed(range(len(stimuli_range))):
            ax.plot3D(PPCA_lst[i][:, 0],  PPCA_lst[i][:, 1],  PPCA_lst[i][:, 2], c=colors[i], alpha=0.5, marker='.')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.title('Production')

        if separate:
            ax = fig.add_subplot(*TOP)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(PPCA_lst[i][:, 0], c=c, alpha=0.5, marker='.')
            ax = fig.add_subplot(*MIDDLE)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(PPCA_lst[i][:, 1], c=c, alpha=0.5, marker='.')
            ax = fig.add_subplot(*BOTTOM)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(PPCA_lst[i][:, 2], c=c, alpha=0.5, marker='.')
        plt.tight_layout()

        return MPCA_lst, PPCA_lst
