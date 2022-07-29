import numpy as np
import matplotlib.pyplot as plt


colors_short = [(0.6196078431372549, 0.00392156862745098, 0.25882352941176473, 1.0),
 (0.7972318339100346, 0.20092272202998845, 0.3008073817762399, 1.0),
 (0.9139561707035756, 0.36239907727797, 0.27935409457900806, 1.0),
 (0.9748558246828143, 0.5574009996155325, 0.32272202998846594, 1.0),
 (0.9934640522875817, 0.7477124183006535, 0.4352941176470587, 1.0),
 (0.9966935793925413, 0.8975009611687812, 0.5770857362552863, 1.0),
 (1.0, 0.9803921568627451, 0.5529411764705883, 1.0)]

colors_long = [(1.0, 0.9803921568627451, 0.5529411764705883, 1.0),
 (0.9173394848135333, 0.9669357939254134, 0.6200692041522493, 1.0),
 (0.7477124183006538, 0.8980392156862746, 0.6274509803921569, 1.0),
 (0.5273356401384084, 0.8106113033448674, 0.6452133794694349, 1.0),
 (0.3280276816608997, 0.6805074971164936, 0.6802768166089965, 1.0),
 (0.2265282583621684, 0.4938869665513264, 0.7224913494809688, 1.0),
 (0.3686274509803922, 0.30980392156862746, 0.6352941176470588, 1.0)]

class SimulationPlotData:
    """
    contains relevant data for plotting simulation for both experiment and parallel setting
    """
    def __init__(self, params, simulation, production, timeout_index, reset_indices):
        self.params = params
        self.simulation = simulation
        self.reset_indices = reset_indices
        self.production = production
        self.timeout_index = timeout_index


class SimulationPlot:
    """
    creates plots of simulation for both experiment and parallel setting

    Attributes
    ----------
    trial: int
        index of example trial for parallel setting
    stimulus: int
        stimulus that was used for parallel simulation
    alpha: float
        u, v, y, I over time opacity
    """

    def __init__(self, data: SimulationPlotData):
        self.data = data

        params = data.params
        simulation = data.simulation

        self.steps = np.arange(len(simulation[:, 0])) * params.dt
        self.subplots = plt.subplots(4, 1, sharex=True, figsize=(6.4,4)) #20, 7 # 6.4,4

    def plot_example_trial(self, stimulus, trial=0):
        '''plots example trial to highlight one trial over all parallel trials'''
        params = self.data.params
        simulation = self.data.simulation
        production = self.data.production
        _, ax = self.subplots
        steps = self.steps

        # if production.size == 0:
        if not production:
            print('all trials timeout')
            return

        print('Stimulus:', stimulus, ', Production of trial', trial, '(blue):', production[trial]*params.dt)
        trial_production = params.first_duration+1*params.dt+stimulus+1*params.dt+production[trial]*params.dt
        ax[0].plot(steps, simulation[:, 0, trial], c='b', linewidth=0.9,  alpha=1)
        ax[1].plot(steps, simulation[:, 1, trial], 'blue', linewidth=0.9, alpha=1)
        ax[2].plot(steps, simulation[:, 2, trial], c='b', linewidth=0.9,  alpha=1)
        ax[2].plot(trial_production, params.th, 'x', c='blue')
        ax[3].plot(steps, simulation[:, 3, trial], c='b', linewidth=0.9,  alpha=1)

    def get_frames(self):
        '''gets times of all measurement stages and production stages'''
        reset_indices = self.data.reset_indices

        m_start = reset_indices[1::3]
        m_stop = reset_indices[2::3]

        p_start = reset_indices[2::3]
        p_stop = reset_indices[3::3]
        return zip(p_start, p_stop, m_start, m_stop)

    def plot_measurement_production_frames(self):
        '''underlays color to all measurment and production stages'''
        _, ax = self.subplots

        for p_start, p_stop, m_start, m_stop in self.get_frames():
            for a in [0, 1, 2, 3]:
                ax[a].axvspan(p_start, p_stop, facecolor='r', alpha=0.1)
                ax[a].axvspan(m_start, m_stop, facecolor='b', alpha=0.1)

    def plot_trials(self, alpha):
        '''plots u. v, y, I over time'''
        _ , ax = self.subplots
        params = self.data.params
        simulation = self.data.simulation
        reset_indices = self.data.reset_indices
        timeout_index = self.data.timeout_index
        steps = self.steps

        print('Timeouts', len(timeout_index))

        ax[0].plot(steps, simulation[:, 0], c='grey', alpha=alpha)
        ax[0].vlines(reset_indices, np.min(np.array(simulation[:, 0])),
                     np.max(np.array(simulation[:, 0])), color='grey', alpha=0.5)
        ax[0].set_title('du/dt', fontsize=11)

        ax[1].plot(steps, simulation[:, 1], 'grey', alpha=alpha)
        ax[1].vlines(reset_indices, np.min(np.array(simulation[:, 1])),
                     np.max(np.array(simulation[:, 1])), color='grey', alpha=0.5)
        ax[1].set_title('dv/dt', fontsize=11)

        ax[2].plot(steps, simulation[:, 2], 'grey', alpha=alpha)
        ax[2].hlines(params.th, 0, simulation.shape[0]*params.dt, linestyle='--', color='lightgray')
        ax[2].vlines(reset_indices, np.min(np.array(simulation[:, 2])),
                     np.max(np.array(simulation[:, 2])*1.1), color='grey', alpha=0.5)
        #ax[2].text(-steps[-1]/25, 0.7, 'timeouts:'+str(len(timeout_index)))
        ax[2].set_title('dy/dt', fontsize=11)

        ax[3].plot(steps, simulation[:, 3], 'grey', alpha=alpha)
        ax[3].vlines(reset_indices, np.min(np.array(simulation[:, 3])),
                     np.max(np.array(simulation[:, 3])), color='grey', alpha=0.5)
        ax[3].set_title('dI/dt', fontsize=11)
        ax[3].set_xlabel('Time (ms)')

        for i in [0,1,2,3]:
            # Hide the right and top spines
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
        
        plt.tight_layout()


        


class BehavioralPlotData:
    '''
    contains relevant data for plotting behavioral data for both experiment and parallel setting 
    '''
    def __init__(self, behavioral_data):
        self.params = behavioral_data.params
        self.stimulus_range = behavioral_data.stimulus_range
        self.production_means = behavioral_data.production_means
        self.production_stds = behavioral_data.production_stds
        self.timeouts = behavioral_data.timeouts
        self.slope = behavioral_data.slope
        self.ind_point = behavioral_data.ind_point


class BehavioralPlot:
    """
    creates behavioral plot (stimulus vs. production) for both experiment and parallel setting
    """

    def __init__(self, data: BehavioralPlotData):
        self.data = data

    def plot_behavior(self, ax=None):
        '''creates behavioral plot or returns the same as subplot'''
        data = self.data
        production_means = data.production_means
        stimulus_range = data.stimulus_range
        production_stds = data.production_stds

        if data.slope == None:
            slope = 0
        else:
            slope = round(data.slope, 3)
        # print('timeouts:', list(zip(stimulus_range, data.timeouts)))

        if ax is None:
            if np.any(data.production_means):
                plt.errorbar(stimulus_range, production_means, yerr=production_stds, fmt='-o', c='k', capsize=1, markersize=4)
            if np.any(data.timeouts):
                plt.text(np.min(stimulus_range)-100, np.max(stimulus_range)+60, 'to='+str(data.timeouts), size=7)
            plt.plot([stimulus_range[0]-100, stimulus_range[-1]+100],
                     [stimulus_range[0]-100, stimulus_range[-1]+100], c='grey', linestyle='--', lw=0.6)
            plt.text(np.min(stimulus_range)-100, np.max(stimulus_range)+100, 'slope='+str(slope))
            plt.xlabel('Stimulus (ms)')
            plt.ylabel('Production (ms)')
        else:
            subplot = ax.plot([stimulus_range[0]-100, stimulus_range[-1]+100],
                              [stimulus_range[0]-100, stimulus_range[-1]+100], c='grey', linestyle='--', lw=0.6)
            if np.any(data.production_means):
                ax.errorbar(stimulus_range, production_means, yerr=production_stds, fmt='-o', c='k', capsize=1, markersize=4)
            if np.any(data.timeouts):
                ax.text(np.min(stimulus_range)-100, np.max(stimulus_range), 'to='+str(data.timeouts), size=7)
            ax.text(np.min(stimulus_range)-100, np.max(stimulus_range)+50, 'slope=' +
                    str(slope))
            return subplot


class SortedPlotData:
    def __init__(self, params, measurement_sorted, production_sorted, I_sorted, stimulus_range):
        self.params = params
        self.stimulus_range = stimulus_range
        self.measurement_sorted = measurement_sorted
        self.production_sorted = production_sorted
        self.I_sorted = I_sorted


class SortedPlot:
    def __init__(self, data: SortedPlotData):
        self.data = data

    def plot_sorted(self):

        data = self.data
        params = self.data.params
        stimulus_range_len = len(data.stimulus_range)

        xticks=[0, 25, 50, 75, 100, 125]
        xticklabels=[0, 250, 500, 750, 1000, 1250]
        colors = colors_short

        # for long range
        if data.stimulus_range[0]>400:
            xticks=[0, 50, 100, 150, 200]
            xticklabels=[0, 500, 1000, 1500, 2000]
            colors = colors_long


        _, ax = plt.subplots(3,stimulus_range_len, sharex=True, sharey='row', figsize=(20,7))
        ax.flatten()[0].set_ylabel('y measurement')
        ax.flatten()[stimulus_range_len].set_ylabel('y reproduction')
        ax.flatten()[stimulus_range_len*2].set_ylabel('I reproduction')
        plt.setp(ax, xticks=xticks, xticklabels=xticklabels)

        for j, (c, stim, lst) in enumerate(zip(colors, data.stimulus_range,  data.measurement_sorted)):
            ax.flatten()[j].set_title(str(stim))
            for i in lst:
                ax.flatten()[j].plot(i, alpha=0.3, color=c)

        for j, (c, lst) in enumerate(zip(colors, data.production_sorted)):
            for i in lst:
                ax.flatten()[j+7].plot(i, alpha=0.3, color=c)
            # ax.flatten()[j+7].hlines(params.th, 0, 75, linestyle='--', color='lightgray')
            ax.flatten()[j+7].axhline(y=params.th, color='lightgray', linestyle='--')

            
        for j, (c, lst) in enumerate(zip(colors, data.I_sorted)):
            for i in lst:
                ax.flatten()[j+14].plot(i, alpha=0.3, color=c)
            ax.flatten()[j+14].set_xlabel('time [ms]')
