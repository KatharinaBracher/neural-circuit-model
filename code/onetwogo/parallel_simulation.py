

from . import BaseSimulation
from .result import SimulationResult
import numpy as np


def remove_timeouts(simu, production, timeout_trials):
    # idx = [i for i, j in enumerate(to) if not np.isfinite(j).all()]
    production = np.delete(np.array(production), timeout_trials)
    simu = np.delete(simu, timeout_trials, 2)  # delete all timeout trials
    return simu, production


class ParallelSimulation(BaseSimulation):

    def __init__(self, params):
        super().__init__(params)

    def simulate(self, stimulus, K, initI):
        params = self.params
        state_init = [np.ones(params.ntrials) * params.uinit,
                      np.ones(params.ntrials) * params.vinit,
                      np.ones(params.ntrials) * params.yinit,
                      np.ones(params.ntrials) * initI]

        nbin = int(stimulus / params.dt)  # stimulus
        nbinfirst = int(params.first_duration / params.dt)  # 750 ms

        # first duration
        simulation, reset_lst = self.network(state_init, reset=0, K=0, nbin=nbinfirst)
        # first flash, no update
        simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=1, K=0, nbin=1)
        # measurement
        simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=0, K=K, nbin=nbin)
        # flash update I in one bin
        simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=1, K=K, nbin=1)
        # behavior
        simulation, reset_lst, production, timeout_trials = self.trial_update(
            simulation, reset_lst,
            reset=0, K=K, nbin=nbin * 2,
            production_step=True
        )
        reset_lst[nbinfirst+2*nbin] = 1  # where th sould be reached

        simulation, production = remove_timeouts(simulation, production, timeout_trials)
        return SimulationResult(params, simulation, reset_lst, production, timeout_trials)

    def production_step(self, simulation, reset_lst, simulation2, reset_lst2, nbin, earlyphase):
        params = self.params
        production = []

        # Check if the bound is reached (sometimes it's not!)
        timeout_trials = []
        for i in range(params.ntrials):
            if np.where(np.diff(np.sign(simulation2[earlyphase:, 2, i]-params.th)))[0].size == 0:
                p = np.inf  # timeout
                timeout_trials.append(i)
            else:
                p = np.where(np.diff(np.sign(simulation2[earlyphase:, 2, i]-params.th)))[0][0] + earlyphase
            production.append(p)
        simulation = np.concatenate((simulation, simulation2))
        reset_lst.extend(reset_lst2)
        return simulation, reset_lst, production, timeout_trials
