import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def sigmoid(x):
    '''Activation function'''
    return 1 / (1 + np.exp(-x))

class OneTwoGo_Task:
    def __init__(self, PARAMS_DICT):
        self.Wut = PARAMS_DICT['Wut']
        self.Wuv = PARAMS_DICT['Wuv']
        self.Wvt = PARAMS_DICT['Wvt']
        self.Wvu = PARAMS_DICT['Wvu']
        self.dt = PARAMS_DICT['dt']
        self.tau = PARAMS_DICT['tau']
        self.sigma = PARAMS_DICT['sigma']
        self.th = PARAMS_DICT['th']
        self.IF = PARAMS_DICT['IF']
        self.ntrials = PARAMS_DICT['ntrials']
        self.uinit = PARAMS_DICT['uinit']
        self.vinit = PARAMS_DICT['vinit']
        self.yinit = PARAMS_DICT['yinit']
        self.first_duration = PARAMS_DICT['first_duration']


    def network(self, state_init, reset, K, nbin):

        u, v, y, I = state_init.copy()

        reset_lst = []
        simulation = np.zeros([nbin,4, self.ntrials]) 

        for i in range(nbin):
            I += (reset * K * (y - self.th)) / self.tau * self.dt
            u += (-u + sigmoid(self.Wut * I - self.Wuv * v - self.IF * reset + np.random.randn(self.ntrials) * self.sigma)) / self.tau * self.dt
            v += (-v + sigmoid(self.Wvt * I - self.Wvu * u + self.IF * reset + np.random.randn(self.ntrials) * self.sigma)) / self.tau * self.dt
            y += (-y + u - v + np.random.randn(self.ntrials) * self.sigma) / self.tau * self.dt

            simulation[i] = [u.copy(), v.copy(), y.copy(), I.copy()]
            reset_lst.append(reset)

        return simulation, reset_lst



    def network_update(self, simulation, reset_lst, reset, K, nbin, behavior=False):
        state_init = simulation[-1]
        simulation2, reset_lst2 = self.network(state_init, reset, K, nbin)

        if behavior:
        #Check if the bound is reached (sometimes it's not!)
            production = []
            for i in range(self.ntrials):
                #if np.where(np.diff(np.sign(simulation2[:,2,i]-self.th)))[0].size == 0:
                    #p = np.inf #not crossing threshold
                #else:
                    #p = np.where(np.diff(np.sign(simulation2[:,2,i]-self.th)))[0][-1]+1
                p = np.where(np.diff(np.sign(simulation2[:,2,i]-self.th)))[0][-1]+1
                production.append(p)
            simulation = np.concatenate((simulation,simulation2))
            reset_lst.extend(reset_lst2)
            return simulation, reset_lst, production

        simulation = np.concatenate((simulation,simulation2))
        reset_lst.extend(reset_lst2)
        return simulation, reset_lst



    def simulate_onetwogo(self, duration, K, initI):
        state_init = [np.ones(self.ntrials) * self.uinit, 
                      np.ones(self.ntrials) * self.vinit,
                      np.ones(self.ntrials) * self.yinit,
                      np.ones(self.ntrials) * initI]

        nbin = int(duration / self.dt) #stimulus
        nbinfirst = int(self.first_duration / self.dt) #750 ms

        #first duration
        simulation, reset_lst = self.network(state_init, reset=0, K=0, nbin=nbinfirst)

        #first flash, no update
        simulation, reset_lst = self.network_update(simulation, reset_lst, reset=1, K=0, nbin=1)

        #measurement
        simulation, reset_lst = self.network_update(simulation, reset_lst, reset=0, K=K, nbin=nbin)

        #flash update I in one bin
        simulation, reset_lst = self.network_update(simulation, reset_lst, reset=1, K=K, nbin=1)

        #behavior
        simulation, reset_lst, production = self.network_update(simulation, reset_lst, reset=0, K=K, nbin=nbin*2, behavior=True)
        reset_lst[nbinfirst+2*nbin] = 1 #where th sould be reached

        return simulation, reset_lst, production







    def statistics(self, s, production):
        std_s = np.std(s, 2)
        production = np.array(production)*self.dt
        mean = np.mean(production)
        std = np.std(production)
        return mean, std, std_s



    def plot_trials(self, duration, trial,  K, initI):
        simu, res, production = self.simulate_onetwogo(duration, K, initI)

        where = np.where(np.array(res)==1)[0] -1
        steps = len(simu[:,0])
        fig, ax = plt. subplots(4,1, sharex=True, figsize=(10,7))

        ax[0].plot(np.arange(steps) * self.dt, simu[:,0], c='grey', alpha=0.05)
        ax[0].plot(np.arange(steps) * self.dt, simu[:,0, trial], c='b', linewidth=0.7,  alpha=0.5)
        ax[0].vlines(where* self.dt, np.min(np.array(simu[:,0])), np.max(np.array(simu[:,0])), color='grey',alpha=0.5)
        ax[0].set_title('du/dt')
        ax[1].plot(np.arange(steps) * self.dt, simu[:,1], 'grey', alpha=0.05)
        ax[1].plot(np.arange(steps) * self.dt, simu[:,1, trial], 'blue', linewidth=0.7, alpha=0.5)
        ax[1].vlines(where*self.dt, np.min(np.array(simu[:,1])), np.max(np.array(simu[:,1])), color='grey', alpha=0.5)
        ax[1].set_title('dv/dt')
        ax[2].plot(np.arange(steps) * self.dt, simu[:,2], 'grey', label = duration, alpha=0.05)
        ax[2].plot(np.arange(steps) * self.dt, simu[:,2, trial], 'blue',linewidth=0.7,  alpha=0.5)
        ax[2].hlines(self.th, 0,self.first_duration+duration+2*duration+2*self.dt,linestyle='--', color='lightgray')
        ax[2].vlines(where* self.dt, np.min(np.array(simu[:,2])), np.max(np.array(simu[:,2])), color='grey',alpha=0.5)
        ax[2].plot(self.first_duration+1*self.dt+duration+1*self.dt+production[trial]*self.dt, self.th, 'x', c='blue')
        ax[2].set_title('dy/dt')
        #ax[2].legend()
        ax[3].plot(np.arange(steps) * self.dt, simu[:,3], 'grey', alpha=0.05)
        ax[3].plot(np.arange(steps) * self.dt, simu[:,3, trial], 'blue',linewidth=0.7,  alpha=0.5)
        ax[3].vlines(where* self.dt, np.min(np.array(simu[:,3])), np.max(np.array(simu[:,3])), color='grey',alpha=0.5)
        ax[3].set_title('dI/dt')
        ax[3].set_xlabel('Time (ms)')

        plt.tight_layout()
        print('Stimulus:', duration, ', Production trial', trial, '(blue)):',production[trial]*self.dt)

        return simu, res, production



    def plot_behavior(self, stimuli_range, K, initI, ax=None):
        mean_lst = []
        std_lst = []
        
        for stim in stimuli_range:
            simu, res, production = self.simulate_onetwogo(stim, K, initI)
            mean, std, _ = self.statistics(simu, production)
            mean_lst.append(mean)
            std_lst.append(std)
            
        reg = linregress(stimuli, mean_lst)
        
        if ax is None:
            plt.errorbar(stimuli, mean_lst, yerr=std_lst, fmt='-o', c='k')
            plt.plot( [stimuli[0]-100,stimuli[-1]+100],[stimuli[0]-100,stimuli[-1]+100], c='grey', linestyle='--')
            plt.text(np.min(stimuli)-100, np.max(stimuli)+100, 'slope='+str(round(reg[0],3)))
            plt.xlabel('Stimulus (ms)')
            plt.ylabel('Production (ms)')
        else:
            subplot = ax.errorbar(stimuli, mean_lst, yerr=std_lst, fmt='-o', c='k')
            ax.plot( [stimuli[0]-100,stimuli[-1]+100],[stimuli[0]-100,stimuli[-1]+100], c='grey', linestyle='--')
            ax.text(np.min(stimuli)-100, np.max(stimuli)+50, 'slope='+str(round(reg[0],3)))
            return subplot
     
        
    def meas_prod_times(self, simu, prod, duration, sample):
        meas_start = int(self.first_duration/self.dt+1)
        meas_stop = int(self.first_duration/self.dt+1+duration/self.dt) #without flashes
        measure = simu[meas_start:meas_stop]
        measure = np.mean(measure,2)

        prod_lst = []
        for i,p in enumerate(prod):
            p_start=int(self.first_duration/self.dt+duration/self.dt+2)
            p_end=int(self.first_duration/self.dt+duration/self.dt+2+p)

            prod_lst.append(simu[p_start:p_end,:,i])
            #print(simu[p_start:p_end,:,i].shape)

        prod_sampled = [signal.resample(trial, sample) for trial in prod_lst]
        prod_sampled = np.array(prod_sampled)
        prod_sampled = np.mean(prod_sampled, 0)

        return measure, prod_sampled



    
    
    def PCA(self, stimuli_range, data=None, K=None, initI=None, experiment=True):
        if experiment:
            #data, has to be sliced according to stimuli and in measure and prod
            #return meas_all prod_all
            #meas_all
            #prod_all
            pass
        
        else:
            meas_all = []
            prod_all = []
            len_m = [0]
            len_p = [0]
            for stim in stimuli_range:
                simu, res, production = self.simulate_onetwogo(stim, K, initI)
                measure, prod = self.meas_prod_times(simu, prod, stim, stim/10+10)
                meas_all.extend(measure)
                prod_all.extend(prod)
                len_m.extend(measure[:,:3].shape[0])
                len_p.extend(prod[:,:3].shape[0])                
        
        pca_m = PCA()
        M = pca_m.fit_transform(meas_all)
        pca_p = PCA()
        P = pca_p.fit_transform(prod_all)
        
        M_lst = []
        P_lst = []
        
        for i in range(len(stimuli_range)):
            M_lst.append(M[sum(M_lst[:i+1]):sum(M_lst[:i+2])])
            P_lst.append(P[sum(P_lst[:i+1]):sum(P_lst[:i+2])])
            
        return M_lst, P_lst    
        
            
        def plot_PCA(self, M_lst, P_lst, stimuli_range, colors):
            fig = plt.figure(figsize=(10,8))
            ax = plt.axes(projection='3d')
            
            for i in range(len(stimuli_range)):
                ax.plot3D(M_lst[i][:,0],  M_lst[i][:,1],  M_lst[i][:,2], c=colors[i], alpha=0.5, marker='.')

            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            plt.title('Measurement')
            
            plt.show()
        