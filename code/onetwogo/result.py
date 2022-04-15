from typing import List
from .plot import BehavioralPlot, BehavioralPlotData, SimulationPlot, SimulationPlotData
import numpy as np


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

    def create_behavioral_plot_data_experiment_simulation(self) -> BehavioralPlotData:
        stimulus_lst = self.stimulus_lst
        production = self.production

        stimulus_range = np.unique(stimulus_lst)

        stim_lst_success = np.delete(np.array(stimulus_lst), self.timeout_index)
        stim_lst_unsuccess = np.array(stimulus_lst)[self.timeout_index]

        ntimeouts, production_means, production_stdts = [], [], []
        for stim in stimulus_range:
            ntimeouts.append(np.count_nonzero(stim_lst_unsuccess == stim))
            # productions of one stimulus
            production_s = production[np.ma.where(stim == stim_lst_success)]*self.params.dt
            production_means.append(np.mean(production_s))
            production_stdts.append(np.std(production_s))
        print(stimulus_range, production_means)

        return BehavioralPlotData(self.params, stimulus_range, production_means, production_stdts, ntimeouts)

    def create_behavioral_plot(self):
        return BehavioralPlot(self.create_behavioral_plot_data_experiment_simulation())


class RangeParallelSimulationResult:

    def __init__(self, result_list: List[SimulationResult], stimulus_range, params):
        self.result_list = result_list
        self.stimulus_range = stimulus_range
        self.params = params

    def create_behavioral_plot_data(self) -> BehavioralPlotData:
        production_means = [result.production_statistics()[0] for result in self.result_list]
        production_stdts = [result.production_statistics()[1] for result in self.result_list]
        ntimeouts = [result.number_of_timeouts() for result in self.result_list]
        return BehavioralPlotData(self.params, self.stimulus_range, production_means, production_stdts, ntimeouts)

    def create_behavioral_plot(self) -> BehavioralPlot:
        return BehavioralPlot(self.create_behavioral_plot_data())
