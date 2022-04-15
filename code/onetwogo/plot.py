import numpy as np
import matplotlib.pyplot as plt


class SimulationPlotData:
    def __init__(self, params, simulation, production, timeout_index, reset_indices):
        self.params = params
        self.simulation = simulation
        self.reset_indices = reset_indices
        self.production = production
        self.timeout_index = timeout_index


class SimulationPlot:

    def __init__(self, data: SimulationPlotData):
        self.data = data

        params = data.params
        simulation = data.simulation

        self.steps = np.arange(len(simulation[:, 0])) * params.dt
        self.subplots = plt.subplots(4, 1, sharex=True, figsize=(10, 7))

    def get_frames(self):
        reset_indices = self.data.reset_indices

        m_start = reset_indices[1::3]
        m_stop = reset_indices[2::3]

        p_start = reset_indices[2::3]
        p_stop = reset_indices[3::3]
        return zip(p_start, p_stop, m_start, m_stop)

    def plot_example_trial(self, stimulus, ntrial=5):
        params = self.data.params
        simulation = self.data.simulation
        production = self.data.production
        _, ax = self.subplots
        steps = self.steps

        if production.size == 0:
            print('all trials timeout')
            return

        trial_production = params.first_duration+1*params.dt+stimulus+1*params.dt+production[ntrial]*params.dt
        ax[0].plot(steps, simulation[:, 0, ntrial], c='b', linewidth=0.7,  alpha=0.5)
        ax[1].plot(steps, simulation[:, 1, ntrial], 'blue', linewidth=0.7, alpha=0.5)
        ax[2].plot(steps, simulation[:, 2, ntrial], c='b', linewidth=0.7,  alpha=0.5)
        print('Stimulus:', stimulus, ', Production trial', ntrial, '(blue)):', production[ntrial]*params.dt)
        ax[2].plot(trial_production, params.th, 'x', c='blue')
        ax[3].plot(steps, simulation[:, 3, ntrial], c='b', linewidth=0.7,  alpha=0.5)

    def plot_measurement_production_frames(self):
        fig, ax = self.subplots

        for p_start, p_stop, m_start, m_stop in self.get_frames():
            for a in [0, 1, 2, 3]:
                ax[a].axvspan(p_start, p_stop, facecolor='r', alpha=0.1)
                ax[a].axvspan(m_start, m_stop, facecolor='b', alpha=0.1)

    def plot_trials(self, alpha):
        _, ax = self.subplots
        params = self.data.params
        simulation = self.data.simulation
        reset_indices = self.data.reset_indices
        steps = self.steps

        ax[0].plot(steps, simulation[:, 0], c='grey', alpha=alpha)
        ax[0].vlines(reset_indices, np.min(np.array(simulation[:, 0])),
                     np.max(np.array(simulation[:, 0])), color='grey', alpha=0.5)
        ax[0].set_title('du/dt')

        ax[1].plot(steps, simulation[:, 1], 'grey', alpha=alpha)
        ax[1].vlines(reset_indices, np.min(np.array(simulation[:, 1])),
                     np.max(np.array(simulation[:, 1])), color='grey', alpha=0.5)
        ax[1].set_title('dv/dt')

        ax[2].plot(steps, simulation[:, 2], 'grey', alpha=alpha)
        ax[2].hlines(params.th, 0, simulation.shape[0]*params.dt, linestyle='--', color='lightgray')
        ax[2].vlines(reset_indices, np.min(np.array(simulation[:, 2])),
                     np.max(np.array(simulation[:, 2])), color='grey', alpha=0.5)
        ax[2].set_title('dy/dt')

        ax[3].plot(steps, simulation[:, 3], 'grey', alpha=alpha)
        ax[3].vlines(reset_indices, np.min(np.array(simulation[:, 3])),
                     np.max(np.array(simulation[:, 3])), color='grey', alpha=0.5)
        ax[3].set_title('dI/dt')
        ax[3].set_xlabel('Time (ms)')


class BehavioralPlotData:
    def __init__(self, params):
        pass


class BehavioralPlot:

    def __init__(self, data: BehavioralPlotData):
        pass
