# from tkinter import W
from typing import List
from .plot import BehavioralPlot, BehavioralPlotData, SimulationPlot, SimulationPlotData
import numpy as np
from scipy.stats import linregress
import pickle
from sklearn.metrics import mean_squared_error


def remove_timeouts(production, timeout_index, stimulus_lst=None):
    '''removes all trials from production (and simulation) that were classified as timeouts'''
    # remove all np.inf productions
    production = np.delete(np.array(production), timeout_index)
    if stimulus_lst is not None:
        stimulus_lst = np.delete(np.array(stimulus_lst), timeout_index)
        return production, stimulus_lst
    return production


def production_statistics(production, params):
    '''computes the mean and standard deviation of production times'''

    production = np.array(production)*params.dt
    mean = np.mean(production)
    std = np.std(production)
    return mean, std


class BehavioralData:
    """
    contains extracted behavioral data of simulation
    """

    def __init__(self, params, stimulus_range, production_means, production_stds, timeouts, slope, ind_point, mse):
        self.params = params
        self.stimulus_range = stimulus_range
        self.production_means = production_means
        self.production_stds = production_stds
        self.timeouts = timeouts
        self.slope = slope
        self.ind_point = ind_point
        self.mse = mse
        # TODO add seed

    def write_to_disk(self, fp, srange, K):
        '''writes behavioral data into a dictionary and into pickle file

        fp: .pickle file
            open pickle file
        srange: str
            indicated if simulus range is short or long
        K: int
            memory parameter
        '''
        # TODO: seed
        tau = self.params.tau
        th = self.params.th
        delay = self.params.delay

        result = dict({'range': srange, 'K': K, 'tau': tau, 'threshold': th, 'delay': delay,
                      'slope': self.slope, 'ind_point': self.ind_point, 'MSE': self.mse})
        pickle.dump(result, fp)


class SimulationResult:
    '''
    contains results of simulation and extracts information of simulation data
    '''

    params: object
    """object that contains all parameters"""
    simulation: np.array
    '''array that contains simulation of u, v, y, I'''
    reset_lst: list
    '''contains sequence of reets over whole simulation'''
    production: list
    '''production time in bins of each trial'''
    timeout_index: list
    '''indices of trials that were classiefied as timeout trials'''
    stimulus_lst: np.array
    '''stimulus times of all stimuli used in simulation'''

    def __init__(self, params, simulation, reset_lst, production, timeout_index, stimulus_lst):  # TODO seed
        self.params = params
        self.simulation = simulation
        self.reset_lst = reset_lst
        self.production = production           # production with np.inf porductions
        self.timeout_index = timeout_index     # boolean list converted to indices in exp.simulatoin
        self.stimulus_lst = stimulus_lst

    def create_simulation_plot_data(self):
        '''returns relevant data for plotting the simulation time course'''
        reset_indices = (np.where(np.array(self.reset_lst) == 1)[0]-1)*self.params.dt
        return SimulationPlotData(
            self.params,
            self.simulation.copy(),
            self.production.copy(),
            self.timeout_index.copy(),
            reset_indices.copy()
        )

    def create_simulation_plot(self):
        '''returns SimulatioPot object'''
        return SimulationPlot(self.create_simulation_plot_data())

    def number_of_timeouts(self):
        '''retunrs number of timeouts'''
        return len(self.timeout_index)

    def create_behavioral_data(self) -> BehavioralData:
        '''computes behvaioral data based on simulation results and retunrs BehaviorlData object'''

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
            ind_point = None
            mse = None
            return BehavioralData(self.params, stimulus_range, production_means, production_stdts,
                                  timeouts, slope, ind_point, mse)

        for stim in stimulus_range:
            nstimuli.append(np.count_nonzero(stimulus_lst == stim))
            ntimeouts.append(np.count_nonzero(stim_lst_unsuccess == stim))
            # productions of one stimulus
            production_s = production[np.ma.where(stim == stimulus_lst)]*self.params.dt
            production_means.append(np.mean(production_s))
            production_stdts.append(np.std(production_s))

        timeouts = np.round(np.array(ntimeouts)/np.array(nstimuli), 2)
        regression_line = linregress(stimulus_range, production_means)
        slope = regression_line[0]
        ind_point = regression_line[1]/(1-regression_line[0])
        # TODO return mean squared error over all trials
        # production, stimulus_lst (timeouts removed)
        mse = mean_squared_error(stimulus_lst, production)
        # TODO return seed
        return BehavioralData(self.params, stimulus_range, production_means, production_stdts,
                              timeouts, slope, ind_point, mse)

    def crate_behavioral_plot_data(self):
        '''returns BehavioralPlotData object'''
        return BehavioralPlotData(self.create_behavioral_data())

    def create_behavioral_plot(self):
        '''returns BehavioralPlot object'''
        return BehavioralPlot(self.create_behavioral_data())


class RangeParallelSimulationResult:
    '''
    contains results of parallel simulation and extracts information of simulation data for behavioral data
    '''

    params: object
    """object that contains all parameters"""
    simulation: list
    '''list that contains SimulatioResults objects'''
    stimulus_range: list
    '''range of stimuli that were used for parallel simulation'''

    def __init__(self, result_list: List[SimulationResult], stimulus_range, params):
        self.result_list = result_list
        self.stimulus_range = stimulus_range
        self.params = params

    def create_behavioral_plot_data(self) -> BehavioralPlotData:
        '''computes behvaioral data based on simulation results and retunrs BehaviorlPlotData object'''

        productions = [remove_timeouts(result.production, result.timeout_index) for result in self.result_list]

        production_means = [production_statistics(production, self.params)[0] for production in productions]
        production_stdts = [production_statistics(production, self.params)[1] for production in productions]
        ntimeouts = [result.number_of_timeouts() for result in self.result_list]
        nstimuli = [self.params.ntrials for i in self.stimulus_range]

        timeouts = np.round(np.array(ntimeouts)/np.array(nstimuli), 2)
        regression_line = linregress(self.stimulus_range, production_means)
        slope = regression_line[0]
        ind_point = regression_line[1]/(1-regression_line[0])
        mse = None  # mean_squared_error(stimulus_lst, productions strip)
        return BehavioralPlotData(BehavioralData(self.params, self.stimulus_range, production_means, production_stdts,
                                  timeouts, slope, ind_point, mse))

    def create_behavioral_plot(self) -> BehavioralPlot:
        '''returns BehavioralPlot object'''
        return BehavioralPlot(self.create_behavioral_plot_data())
