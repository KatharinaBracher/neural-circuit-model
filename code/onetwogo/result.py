# from tkinter import W
from typing import List
from .plot import BehavioralPlot, BehavioralPlotData, SimulationPlot, SimulationPlotData, SortedPlot, SortedPlotData
import numpy as np
from scipy.stats import linregress
import pickle
from sklearn.metrics import mean_squared_error


def remove_timeouts_(production, timeout_index, stimulus_lst=None):
    '''removes all trials from production (and simulation) that were classified as timeouts'''
    # remove all np.inf productions
    production = np.delete(np.array(production), timeout_index)
    if stimulus_lst is not None:
        stimulus_lst = np.delete(np.array(stimulus_lst), timeout_index)
        return production, stimulus_lst
    return production

def remove_timeouts(timeout_index, lst):
    '''removes all trials from production (and simulation) that were classified as timeouts'''
    # remove all np.inf productions
    return np.delete(np.array(lst), timeout_index)

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
    params: object
    '''object that contains all parameters'''
    stimulus_range: list
    '''list of stimuli used in simulatoin'''
    production_means: list
    '''list with means of production times for each stimulus in stimulus range'''
    production_stds: list
    '''list with standard deviations of production times for each stimulus in stimulus range'''
    timeouts: np.array
    '''proportion of timouts of all trials per stimulus'''
    slope: float
    '''slope of stimulus vs. production'''
    ind_point: float
    '''indifference point of stimulus vs. production'''
    bias2: float
    '''mean squared bias over stimulus range '''
    bias: float
    '''mean bias over stimulus range '''
    var: float
    '''mean variance over stimulus range'''
    mse: float
    '''Mean Squared Error over all trials of simulation'''

    def __init__(self, params, stimulus_range, production_means, production_stds, timeouts, slope, ind_point, bias2, bias, var, mse):
        self.params = params
        self.stimulus_range = stimulus_range
        self.production_means = production_means
        self.production_stds = production_stds
        self.timeouts = timeouts
        self.slope = slope
        self.ind_point = ind_point
        self.bias2 = bias2
        self.bias = bias
        self.var = var
        self.mse = mse
        # TODO add seed

    def write_to_disk(self, fp, srange, K, seed):
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
        sigma = self.params.sigma

        result = dict({'range': srange, 'K': K, 'tau': tau, 'threshold': th, 'delay': delay, 'sigma': sigma,
                      'slope': self.slope, 'production_stds': self.production_stds, 'ind_point': self.ind_point, 'bias2': self.bias2, 'bias': self.bias, 'var': self.var, 'MSE': self.mse,
                      'seed': seed})
        pickle.dump(result, fp)

class SortedData:
    """
    contains cut and sorted data of simulation (y and I)
    """
    params: object
    '''object that contains all parameters'''
    stimulus_range: list
    '''list of stimuli used in simulation'''


    def __init__(self, params, measurement_sorted, production_sorted, I_sorted, stimulus_range):
        self.params = params
        self.stimulus_range = stimulus_range
        self.measurement_sorted = measurement_sorted
        self.production_sorted = production_sorted
        self.I_sorted = I_sorted

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
        reset_indices = self.get_reset_indices()*self.params.dt
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
    
    def get_reset_indices(self):
        return np.where(np.array(self.reset_lst) == 1)[0]-1

    def get_stimulus_range(self):
        return np.unique(self.stimulus_lst)

    def create_behavioral_data(self) -> BehavioralData:
        '''computes behvaioral data based on simulation results and retunrs BehaviorlData object'''

        stimulus_range = self.get_stimulus_range()
        stim_lst_unsuccess = np.array(self.stimulus_lst)[self.timeout_index]

        production = remove_timeouts(self.timeout_index, self.production)
        stim_lst_success = remove_timeouts(self.timeout_index,self.stimulus_lst)

        ntimeouts, nstimuli, production_means, production_stds = [], [], [], []

        # if all trials are timeout return 0 mean and 0 std: production.size == 0:
        # if more than 10% timeout
        if len(self.timeout_index)/len(self.stimulus_lst)>0.1:
            for stim in stimulus_range:
                nstimuli.append(np.count_nonzero(self.stimulus_lst == stim))
                ntimeouts.append(np.count_nonzero(stim_lst_unsuccess == stim))
                production_means.append(None)
                production_stds.append(None)
            timeouts = np.round(np.array(ntimeouts)/np.array(nstimuli), 2)
            slope, ind_point, bias2, bias, var, mse = None, None, None, None, None, None
            return BehavioralData(self.params, stimulus_range, production_means, production_stds,
                                  timeouts, slope, ind_point, bias2, bias, var, mse)

        for stim in stimulus_range:
            nstimuli.append(np.count_nonzero(self.stimulus_lst == stim))
            ntimeouts.append(np.count_nonzero(stim_lst_unsuccess == stim))
            # productions of one stimulus
            production_stim = production[np.ma.where(stim == stim_lst_success)]*self.params.dt
            production_means.append(np.mean(production_stim))
            production_stds.append(np.std(production_stim))

        timeouts = np.round(np.array(ntimeouts)/np.array(nstimuli), 2)
        if np.any(timeouts>0.1):
            slope, ind_point, bias2, bias, var, mse = None, None, None, None, None, None
        else:
            regression_line = linregress(stimulus_range, production_means)
            slope = regression_line[0]
            ind_point = regression_line[1]/(1-regression_line[0])
            # TODO return mean squared error over all trials
            # production, stimulus_lst (timeouts removed)
            bias2 = np.sum(np.square(np.array(production_means)-np.array(stimulus_range)))/len(stimulus_range)
            bias = np.sum(np.array(production_means)-np.array(stimulus_range))/len(stimulus_range)
            var = np.sum(np.square(production_stds))/len(stimulus_range)
            mse = mean_squared_error(stim_lst_success, production)
            # TODO return seed
        return BehavioralData(self.params, stimulus_range, production_means, production_stds,
                              timeouts, slope, ind_point, bias2, bias, var, mse)

    def crate_behavioral_plot_data(self):
        '''returns BehavioralPlotData object'''
        return BehavioralPlotData(self.create_behavioral_data())

    def create_behavioral_plot(self):
        '''returns BehavioralPlot object'''
        return BehavioralPlot(self.create_behavioral_data())

    def get_frames(self, start): #reuse in plot
        '''gets times of all measurement stages and production stages'''
        reset_indices = self.get_reset_indices()
        params = self.params
        # no delay
        start_frame = 0
        reg = 2
        # delay
        if params.delay > 0:
            start_frame = 1
            reg = 3
        
        m_start = reset_indices[start_frame::reg]
        m_stop = reset_indices[start_frame+1::reg]

        p_start = reset_indices[start_frame+1::reg]
        p_stop = reset_indices[start_frame+2::reg]
        return zip(p_start[start:], p_stop[start:], m_start[start:], m_stop[start:])  # for all remove here

        # TODO remove timeouts in m_start, etc and in stimulus_lst (successfull) for PCA

    def create_sorted_data(self, start=0):

        measurement_lst, production_lst, production_I = [], [], []
        for p_start, p_stop, m_start, m_stop in self.get_frames(start):
            measurement_lst.append(self.simulation[:,2][m_start:m_stop])
            production_lst.append(self.simulation[:,2][p_start:p_stop])
            production_I.append(self.simulation[:,3][p_start:p_stop])

        stimulus_range = self.get_stimulus_range()

        measurement_sorted, production_sorted, I_sorted = [], [], []
        for stim in stimulus_range:
            measurement_sorted.append(np.array(measurement_lst, dtype=object)[np.where(self.stimulus_lst[start:]==stim)[0]])  # for all remove here
            production_sorted.append(np.array(production_lst, dtype=object)[np.where(self.stimulus_lst[start:]==stim)[0]])  # for all remove here
            I_sorted.append(np.array(production_I, dtype=object)[np.where(self.stimulus_lst[start:]==stim)[0]])  # for all remove here
        
        return SortedData(self.params, measurement_sorted, production_sorted, I_sorted, stimulus_range)

    def crate_sorted_plot_data(self):
        '''returns SortedPlotData object'''
        return SortedPlotData(self.create_sorted_data())

    def create_sorted_plot(self):
        '''returns SortedPlot object'''
        return SortedPlot(self.create_sorted_data())



class RangeParallelSimulationResult:
    '''
    contains results of parallel simulation and extracts information of simulation data for behavioral data
    '''

    params: object
    """object that contains all parameters"""
    result_list: list
    '''list that contains SimulatioResults objects'''
    stimulus_range: list
    '''range of stimuli that were used for parallel simulation'''

    def __init__(self, result_list: List[SimulationResult], stimulus_range, params):
        self.result_list = result_list
        self.stimulus_range = stimulus_range
        self.params = params

    def create_behavioral_plot_data(self) -> BehavioralPlotData:
        '''computes behavioral data based on simulation results and retunrs BehaviorlPlotData object'''

        productions = [remove_timeouts(result.timeout_index, result.production) for result in self.result_list]

        production_means = [production_statistics(production, self.params)[0] for production in productions]
        production_stds = [production_statistics(production, self.params)[1] for production in productions]
        ntimeouts = [result.number_of_timeouts() for result in self.result_list]
        nstimuli = [self.params.ntrials for i in self.stimulus_range]

        timeouts = np.round(np.array(ntimeouts)/np.array(nstimuli), 2)
        regression_line = linregress(self.stimulus_range, production_means)
        slope = regression_line[0]
        ind_point = regression_line[1]/(1-regression_line[0])

        bias2 = np.sum(np.square(np.array(production_means)-np.array(self.stimulus_range)))/len(self.stimulus_range)
        bias = np.sum(np.array(production_means)-np.array(self.stimulus_range))/len(self.stimulus_range)
        var = np.sum(np.square(production_stds))/len(self.stimulus_range)
        mse = None
        return BehavioralPlotData(BehavioralData(self.params, self.stimulus_range, production_means, production_stds,
                                  timeouts, slope, ind_point, bias2, bias, var, mse))

    def create_behavioral_plot(self) -> BehavioralPlot:
        '''returns BehavioralPlot object'''
        return BehavioralPlot(self.create_behavioral_plot_data())
