from . import BaseSimulation
from .result import SimulationResult
import numpy as np
import random


class ExperimentSimulation(BaseSimulation):

    def __init__(self, params):
        super().__init__(params)

    def generate_stimulus_lst(self, stimulus_range):
        params = self.params
        stimulus_lst = [random.choice(stimulus_range) for i in range(params.ntrials)]
        return stimulus_lst

    def simulate(self, stimulus_lst, K, initI):
        # TODO generate random stimuli list of range
        params = self.params

        state_init = [np.ones(1) * params.uinit,
                      np.ones(1) * params.vinit,
                      np.ones(1) * params.yinit,
                      np.ones(1) * initI]

        nbinfirst = int(params.first_duration / params.dt)
        nbindelay = int(params.delay / params.dt)

        simulation, reset_lst = self.network(
            state_init, reset=0, K=0, nbin=nbinfirst)

        timeout_idx = []
        for i, stimulus in enumerate(stimulus_lst):
            print(stimulus)
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
            if timeout:
                timeout_idx.append(i)

        reset_lst[-1] = 1  # last production
        # stimulus_lst[] remove timout_indx

        return SimulationResult(params, simulation, reset_lst, production, timeout_idx)

    def production_step(self, simulation, reset_lst, simulation2, reset_lst2, nbin, earlyphase):
        params = self.params
        production = []

        # timeout if threshold not reached in late phase (late timeout) or only reached in early phase (early timeout)
        if ((np.where(np.diff(np.sign(np.squeeze(simulation2[earlyphase:, 2])-params.th)))[0].size == 0) and
                (np.where(np.diff(np.sign(np.squeeze(simulation2[:, 2])-params.th)))[0].size != 0)):
            # only early phase
            print(
                np.where(np.diff(np.sign(np.squeeze(simulation2[:, 2])-params.th)))[0])
            print('1: late timeout, 2: early timeout')
        if np.where(np.diff(np.sign(np.squeeze(simulation2[earlyphase:, 2])-params.th)))[0].size == 0:
            timeout = 1
            print('late timeout')
            # production.append(np.inf)
            # remove time out trial completly from u,v,y,I
            # remove nbin/2 (measurement), 3 flashes and delay
            simulation = simulation[:-int(nbin/2+3+params.delay/params.dt)]
            # remove nbin/2 (measurement), 3 flashes and delay
            reset_lst = reset_lst[:-int(nbin/2+3+params.delay/params.dt)]
        else:
            timeout = 0
            # get last th crossing (problem if oscilating around th)
            # p = np.where(np.diff(np.sign(np.squeeze(simulation2)[:,2]-params.th)))[0][-1] #+1
            p = np.where(np.diff(np.sign(np.squeeze(simulation2[earlyphase:, 2])-params.th)))[
                0][0] + earlyphase  # +1 #cut first th crossing and then take first one
            production.append(p)
            simulation = np.concatenate((simulation, simulation2[:p+1]))
            reset_lst.extend(reset_lst2[:p+1])
        return simulation, reset_lst, production, timeout
