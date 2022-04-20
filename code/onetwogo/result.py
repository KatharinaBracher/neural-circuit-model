from typing import List
from .plot import BehavioralPlot, BehavioralPlotData, SimulationPlot, SimulationPlotData
import numpy as np
from scipy.stats import linregress


def remove_timeouts(production, timeout_index, stimulus_lst=None):
    # remove all np.inf productions
    production = np.delete(np.array(production), timeout_index)
    if stimulus_lst is not None:
        stimulus_lst = np.delete(np.array(stimulus_lst), timeout_index)
        return production, stimulus_lst
    return production


def production_statistics(production, params):
    production = np.array(production)*params.dt
    mean = np.mean(production)
    std = np.std(production)
    return mean, std


class SimulationResult:
    def __init__(self, params, simulation, reset_lst, production, timeout_index, stimulus_lst):
        self.params = params
        self.simulation = simulation
        self.reset_lst = reset_lst
        self.production = production           # production with np.inf porductions
        self.timeout_index = timeout_index     # boolean list converted to indices in exp.simulatoin
        self.stimulus_lst = stimulus_lst

    def create_simulation_plot_data(self):
        reset_indices = (np.where(np.array(self.reset_lst) == 1)[0]-1)*self.params.dt
        return SimulationPlotData(
            self.params,
            self.simulation.copy(),
            self.production.copy(),
            self.timeout_index.copy(),
            reset_indices.copy()
        )

    def create_simulation_plot(self):
        return SimulationPlot(self.create_simulation_plot_data())

    def number_of_timeouts(self):
        return len(self.timeout_index)

    def create_behavioral_data(self) -> BehavioralPlotData:

        stimulus_range = np.unique(self.stimulus_lst)
        stim_lst_unsuccess = np.array(self.stimulus_lst)[self.timeout_index]

        production, stimulus_lst = remove_timeouts(self.production, self.timeout_index, self.stimulus_lst)

        ntimeouts, nstimuli, production_means, production_stdts = [], [], [], []
        # If all trials are timeout retun 0 mean and 0 std
        if production.size == 0:
            for stim in stimulus_range:
                nstimuli.append(np.count_nonzero(stimulus_lst == stim))
                ntimeouts.append(np.count_nonzero(stim_lst_unsuccess == stim))
                production_means.append(None)
                production_stdts.append(None)
            timeouts = np.round(np.array(ntimeouts)/np.array(nstimuli), 2)
            slope = None
            return BehavioralPlotData(self.params, stimulus_range, production_means, production_stdts,
                                      timeouts, slope)

        for stim in stimulus_range:
            nstimuli.append(np.count_nonzero(stimulus_lst == stim))
            ntimeouts.append(np.count_nonzero(stim_lst_unsuccess == stim))
            # productions of one stimulus
            production_s = production[np.ma.where(stim == stimulus_lst)]*self.params.dt
            production_means.append(np.mean(production_s))
            production_stdts.append(np.std(production_s))

        timeouts = np.round(np.array(ntimeouts)/np.array(nstimuli), 2)
        regression_line = linregress(stimulus_range, production_means)
        slope = round(regression_line[0], 3)

        # TODO return indifference point
        # TODO return mean squared error over all trials 

        return BehavioralPlotData(self.params, stimulus_range, production_means, production_stdts,
                                  timeouts, slope)

    def create_behavioral_plot(self):
        return BehavioralPlot(self.create_behavioral_data())


class RangeParallelSimulationResult:

    def __init__(self, result_list: List[SimulationResult], stimulus_range, params):
        self.result_list = result_list
        self.stimulus_range = stimulus_range
        self.params = params

    def create_behavioral_plot_data(self) -> BehavioralPlotData:
        productions = [remove_timeouts(result.production, result.timeout_index) for result in self.result_list]

        production_means = [production_statistics(production, self.params)[0] for production in productions]
        production_stdts = [production_statistics(production, self.params)[1] for production in productions]
        ntimeouts = [result.number_of_timeouts() for result in self.result_list]
        nstimuli = [self.params.ntrials for i in self.stimulus_range]

        timeouts = np.round(np.array(ntimeouts)/np.array(nstimuli), 2)
        regression_line = linregress(self.stimulus_range, production_means)
        slope = round(regression_line[0], 3)
        return BehavioralPlotData(self.params, self.stimulus_range, production_means, production_stdts,
                                  timeouts, slope)

    def create_behavioral_plot(self) -> BehavioralPlot:
        return BehavioralPlot(self.create_behavioral_plot_data())
