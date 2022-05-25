from . import BaseSimulation
from .result import SimulationResult
import numpy as np
import random


class ExperimentSimulation(BaseSimulation):
    """
    simulates an experiment sampling stimuli randomly from a stimulus range


    Attributes
    ----------
    K: int
        memory parameter, weights the update of I
    stimulus_range: list
        stimulus range for measurment stage, for each stimulus the parallel simulation is performed
    """

    def __init__(self, params):
        super().__init__(params)

    def generate_stimulus_lst(self, stimulus_range):
        '''takes stimulus range and creates list with random stimuli for consecutive of trials)'''

        params = self.params
        stimulus_lst = [random.choice(stimulus_range) for i in range(params.ntrials)]
        return np.array(stimulus_lst)

    def simulate(self, stimulus_lst, K):
        '''carries out the different stages of a trial for each stimulus from the stimulus list
        and returns the SimulationResult:
        parameter object, the simulation of u, v, y, I, a list of production times, the timepoints
        of reset and the indices of timeout trials)'''

        params = self.params

        state_init = [np.ones(1) * params.uinit,
                      np.ones(1) * params.vinit,
                      np.ones(1) * params.yinit,
                      np.ones(1) * params.Iinit]

        nbinfirst = int(params.first_duration / params.dt)
        nbindelay = int(params.delay / params.dt)

        simulation, reset_lst = self.network(
            state_init, reset=0, K=0, nbin=nbinfirst)

        timeout_index = []
        production_lst = []
        for i, stimulus in enumerate(stimulus_lst):
            nbin = int(stimulus / params.dt)  # stimulus
            # reset after behavior
            simulation, reset_lst = self.trial_update(
                simulation, reset_lst, reset=1, K=0, nbin=1)
            simulation, reset_lst = self.trial_update(
                simulation, reset_lst, reset=0, K=K, nbin=nbindelay)
            # measurement
            simulation, reset_lst = self.trial_update(
                simulation, reset_lst, reset=1, K=0, nbin=1)
            simulation, reset_lst = self.trial_update(
                simulation, reset_lst, reset=0, K=K, nbin=nbin)
            # update I
            simulation, reset_lst = self.trial_update(
                simulation, reset_lst, reset=1, K=K, nbin=1)
            # production
            simulation, reset_lst, production, timeout = self.trial_update(
                simulation, reset_lst, reset=0, K=K, nbin=nbin * 2, production_step=True)
            production_lst.append(production)
            if timeout:
                timeout_index.append(i)

        reset_lst[-1] = 1  # last production

        return SimulationResult(params, simulation, reset_lst, production_lst, timeout_index, stimulus_lst)

    def production_step(self, simulation, reset_lst, simulation2, reset_lst2, earlyphase):
        '''defines the last stage of a trial (production stage)
        and determines the production time or if the trial was a timeout trial'''

        params = self.params

        # Determine timeout
        # timeout if threshold not reached in late phase (late timeout)
        # timoeut if reached in early phase only (early timeout)
        
        # if ((np.where(np.diff(np.sign(np.squeeze(simulation2[earlyphase:, 2])-params.th)))[0].size == 0) and
                # (np.where(np.diff(np.sign(np.squeeze(simulation2[:, 2])-params.th)))[0].size != 0)):
            # print(np.where(np.diff(np.sign(np.squeeze(simulation2[:, 2])-params.th)))[0])
            # print('n=1: late timeout early crossing, n>1: early timeout')
        # elif np.where(np.diff(np.sign(np.squeeze(simulation2[:, 2])-params.th)))[0].size == 0:
            # print('late timeout no crossing')

        if np.where(np.diff(np.sign(np.squeeze(simulation2[earlyphase:, 2])-params.th)))[0].size == 0:
            timeout = 1
            production = np.inf

            simulation = np.concatenate((simulation, simulation2))
            reset_lst.extend(reset_lst2)
            # remove time out trial completly from u,v,y,I: remove nbin/2 (stim), 3 flashes and delay
            # simulation = simulation[:-int(nbin/2+3+params.delay/params.dt)]
            # reset_lst = reset_lst[:-int(nbin/2+3+params.delay/params.dt)]
        else:
            timeout = 0
            # get last th crossing (problem if oscilating around th)
            # p = np.where(np.diff(np.sign(np.squeeze(simulation2)[:,2]-params.th)))[0][-1] #+1
            production = np.where(np.diff(np.sign(np.squeeze(simulation2[earlyphase:, 2])-params.th)))[
                0][0] + earlyphase  # +1 #cut first th crossing and then take first one
            simulation = np.concatenate((simulation, simulation2[:production+1]))
            reset_lst.extend(reset_lst2[:production+1])

        return simulation, reset_lst, production, timeout
