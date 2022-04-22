from . import BaseSimulation
from .result import RangeParallelSimulationResult, SimulationResult
import numpy as np


class ParallelSimulation(BaseSimulation):
    """
    A class that simulates trials parallel for one stimulus


    Attributes
    ----------
    K: int
        memory parameter, weighs how much input
    stimulus: int
        stimulus for the measuremnt stage of the trials
    stimulus_range: list
        stimulus range for measurment stage, for each stimulus the parallel simulation is performed
    """

    def __init__(self, params):
        super().__init__(params)

    def simulate(self, stimulus, K):
        '''function that carries out the different stages of a trial and returns the SimulationResult
        (consisting of all parameters, the simulation, a list of production times, the timepoints of reset 
        and the indices of timeout trials)'''

        params = self.params
        state_init = [np.ones(params.ntrials) * params.uinit,
                      np.ones(params.ntrials) * params.vinit,
                      np.ones(params.ntrials) * params.yinit,
                      np.ones(params.ntrials) * params.Iinit]

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
        simulation, reset_lst, production, timeout_index = self.trial_update(
            simulation, reset_lst,
            reset=0, K=K, nbin=nbin * 2,
            production_step=True
        )
        reset_lst[nbinfirst+2*nbin] = 1  # where th sould be reached

        return SimulationResult(params, simulation, reset_lst, production, timeout_index, [stimulus])

    def production_step(self, simulation, reset_lst, simulation2, reset_lst2, _, earlyphase):
        '''function that defines the last stage of the trial (production stage)
        and determines the production time and timeout trials'''

        params = self.params
        production = []

        # Check if the bound is reached (sometimes it's not!)
        timeout_index = []
        for i in range(params.ntrials):
            if np.where(np.diff(np.sign(simulation2[earlyphase:, 2, i]-params.th)))[0].size == 0:
                timeout_index.append(i)
                p = np.inf
            else:
                p = np.where(np.diff(np.sign(simulation2[earlyphase:, 2, i]-params.th)))[0][0] + earlyphase
            production.append(p)
        simulation = np.concatenate((simulation, simulation2))
        reset_lst.extend(reset_lst2)
        return simulation, reset_lst, production, timeout_index

    def simulate_range(self, stimulus_range, K) -> RangeParallelSimulationResult:
        '''function that performess parallel simulation for a range of stimuli returns'''

        return RangeParallelSimulationResult([self.simulate(stim, K) for stim in stimulus_range],
                                             stimulus_range, self.params)


'''def remove_timeouts(simu, production, timeout_trials):
    ''''''removes all trials from production and simulation that were classified as timeout''''''
    production = np.delete(np.array(production), timeout_trials)
    simu = np.delete(simu, timeout_trials, 2)  # delete all timeout trials
    return simu, production'''
