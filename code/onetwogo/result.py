from typing import List
from .plot import BehavioralPlot, BehavioralPlotData, SimulationPlot, SimulationPlotData
import numpy as np
from scipy.stats import linregress


class SimulationResult:
    def __init__(self, params, simulation, reset_lst, production, timeout_index, stimulus_lst):
        self.params = params
        self.simulation = simulation
        self.reset_lst = reset_lst
        self.production = production
        self.timeout_index = timeout_index
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

    def production_statistics(self):
        # TODO adjust for experiment setting
        production = np.array(self.production)*self.params.dt
        mean = np.mean(production)
        std = np.std(production)
        return mean, std

    def number_of_timeouts(self):
        return len(self.timeout_index)

    def create_behavioral_data(self) -> BehavioralPlotData:
        stimulus_lst = self.stimulus_lst
        production = self.production

        stimulus_range = np.unique(stimulus_lst)
        stim_lst_success = np.delete(np.array(stimulus_lst), self.timeout_index)
        stim_lst_unsuccess = np.array(stimulus_lst)[self.timeout_index]

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
            production_s = production[np.ma.where(stim == stim_lst_success)]*self.params.dt
            production_means.append(np.mean(production_s))
            production_stdts.append(np.std(production_s))

        timeouts = np.round(np.array(ntimeouts)/np.array(nstimuli), 2)
        regression_line = linregress(stimulus_range, production_means)
        slope = round(regression_line[0], 3)
        # TODO return indifference point

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
        production_means = [result.production_statistics()[0] for result in self.result_list]
        production_stdts = [result.production_statistics()[1] for result in self.result_list]
        ntimeouts = [result.number_of_timeouts() for result in self.result_list]
        nstimuli = [self.params.ntrials for stim in self.stimulus_range]

        timeouts = np.round(np.array(ntimeouts)/np.array(nstimuli), 2)
        regression_line = linregress(self.stimulus_range, production_means)
        slope = round(regression_line[0], 3)
        return BehavioralPlotData(self.params, self.stimulus_range, production_means, production_stdts,
                                  timeouts, slope)

    def create_behavioral_plot(self) -> BehavioralPlot:
        return BehavioralPlot(self.create_behavioral_plot_data())
