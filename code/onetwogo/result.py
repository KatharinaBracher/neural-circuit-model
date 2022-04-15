from .plot import BehavioralPlot, BehavioralPlotData, SimulationPlot, SimulationPlotData
import numpy as np


class SimulationResult:
    def __init__(self, params, simulation, reset_lst, production, timeout_index):
        self.params = params
        self.simulation = simulation
        self.reset_lst = reset_lst
        self.production = production
        self.timeout_index = timeout_index

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

    def create_behavioral_plot_data(self):
        return BehavioralPlotData(
            self.params
        )

    def create_behavioral_plot(self):
        return BehavioralPlot(self.create_behavioral_plot_data())
