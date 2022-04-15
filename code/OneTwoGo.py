import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy import signal
from sklearn.decomposition import PCA

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
        self.delay = PARAMS_DICT['delay']


    def network(self, state_init, reset, K, nbin):

        u, v, y, I = state_init.copy()
        
        #save bool with resets=1 over time
        reset_lst = []
        simulation = np.zeros([nbin,4,self.ntrials])

        for i in range(nbin):
            I += (reset * K * (y - self.th)) / self.tau * self.dt
            u += (-u + sigmoid(self.Wut * I - self.Wuv * v - self.IF * reset + np.random.randn(self.ntrials) * self.sigma)) / self.tau * self.dt
            v += (-v + sigmoid(self.Wvt * I - self.Wvu * u + self.IF * reset + np.random.randn(self.ntrials) * self.sigma)) / self.tau * self.dt
            y += (-y + u - v + np.random.randn(self.ntrials) * self.sigma) / self.tau * self.dt

            simulation[i] = [u.copy(), v.copy(), y.copy(), I.copy()]
            reset_lst.append(reset)

        return simulation, reset_lst


    def trial_update(self, simulation, reset_lst, reset, K, nbin, parallel_production=False, experiment=False):
        #get prev I,v,u,y to continue trial or experiment
        state_init = simulation[-1]
        #next step simulation
        simulation2, reset_lst2 = self.network(state_init, reset, K, nbin)
        production = []
        earlyphase = int(0.4*nbin/2) #0.4 of stim duration considered early phase
                
        #for parallel simulation: production stage step
        if parallel_production:
            #Check if the bound is reached (sometimes it's not!)
            timeout_trials = []
            for i in range(self.ntrials):
                if np.where(np.diff(np.sign(simulation2[earlyphase:,2,i]-self.th)))[0].size == 0:
                    p = np.inf #timeout
                    timeout_trials.append(i)
                else:
                    p = np.where(np.diff(np.sign(simulation2[earlyphase:,2,i]-self.th)))[0][0] + earlyphase
                production.append(p)
            simulation = np.concatenate((simulation,simulation2))
            reset_lst.extend(reset_lst2)
            return simulation, reset_lst, production, timeout_trials
        
        #for experiment: production stage step 
        if experiment:
            #timeout if threshold not reached in late phase (late timeout) or only reached in early phase (early timeout)
            if ((np.where(np.diff(np.sign(np.squeeze(simulation2[earlyphase:,2])-self.th)))[0].size == 0) and (np.where(np.diff(np.sign(np.squeeze(simulation2[:,2])-self.th)))[0].size != 0)):
                print(np.where(np.diff(np.sign(np.squeeze(simulation2[:,2])-self.th)))[0]) #only early phase
                print('1: late timeout, 2: early timeout')
            if np.where(np.diff(np.sign(np.squeeze(simulation2[earlyphase:,2])-self.th)))[0].size == 0:
                timeout=1
                print('late timeout')
                #production.append(np.inf)
                #remove time out trial completly from u,v,y,I
                simulation = simulation[:-int(nbin/2+3+self.delay/self.dt)] #remove nbin/2 (measurement), 3 flashes and delay 
                reset_lst = reset_lst[:-int(nbin/2+3+self.delay/self.dt)] #remove nbin/2 (measurement), 3 flashes and delay
            else:
                timeout=0
                #p = np.where(np.diff(np.sign(np.squeeze(simulation2)[:,2]-self.th)))[0][-1] #+1 #get last th crossing (problem if oscilating around th)
                p = np.where(np.diff(np.sign(np.squeeze(simulation2[earlyphase:,2])-self.th)))[0][0] + earlyphase  #+1 #cut first th crossing and then take first one
                production.append(p)
                simulation = np.concatenate((simulation,simulation2[:p+1]))
                reset_lst.extend(reset_lst2[:p+1])
            return simulation, reset_lst, production, timeout                           
        
        #for all stages exept production
        simulation = np.concatenate((simulation,simulation2))
        reset_lst.extend(reset_lst2)
        return simulation, reset_lst



    def simulate_parallel(self, stimulus, K, initI):
        state_init = [np.ones(self.ntrials) * self.uinit, 
                      np.ones(self.ntrials) * self.vinit,
                      np.ones(self.ntrials) * self.yinit,
                      np.ones(self.ntrials) * initI]

        nbin = int(stimulus / self.dt) #stimulus
        nbinfirst = int(self.first_duration / self.dt) #750 ms

        #first duration
        simulation, reset_lst = self.network(state_init, reset=0, K=0, nbin=nbinfirst)
        #first flash, no update
        simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=1, K=0, nbin=1)
        #measurement
        simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=0, K=K, nbin=nbin)
        #flash update I in one bin
        simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=1, K=K, nbin=1)
        #behavior
        simulation, reset_lst, production, timeout_trials = self.trial_update(simulation, reset_lst, reset=0, K=K, nbin=nbin*2, parallel_production=True)
        reset_lst[nbinfirst+2*nbin] = 1 #where th sould be reached
        
        simulation, production = self.remove_timeouts(simulation, production, timeout_trials)
        return simulation, reset_lst, production, timeout_trials
    
    
    def simulate_experiment(self, stimulus_lst, K, initI):
        #TODO generate random stimuli list of range
        
        state_init = [np.ones(self.ntrials) * self.uinit, 
                      np.ones(self.ntrials) * self.vinit,
                      np.ones(self.ntrials) * self.yinit,
                      np.ones(self.ntrials) * initI]
                                          
        nbinfirst = int(self.first_duration / self.dt)
        nbindelay = int(self.delay / self.dt) 
        
        simulation, reset_lst = self.network(state_init, reset=0, K=0, nbin=nbinfirst)
        
        timeout_idx = []
        for i,stimulus in enumerate(stimulus_lst):
            print(stimulus)
            nbin = int(stimulus / self.dt) #stimulus
            #reset after behavior
            simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=1, K=0, nbin=1)
            simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=0, K=K, nbin=nbindelay)
            #measurement
            simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=1, K=0, nbin=1)
            simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=0, K=K, nbin=nbin)
            #update I
            simulation, reset_lst = self.trial_update(simulation, reset_lst, reset=1, K=K, nbin=1)
            #production
            simulation, reset_lst, production, timeout = self.trial_update(simulation, reset_lst, reset=0, K=K, nbin=nbin*2, experiment=True)
            if timeout:
                timeout_idx.append(i)
                
        reset_lst[-1] = 1 #last production
        #stimulus_lst[] remove timout_indx
        return simulation, reset_lst, timeout_idx



    def remove_timeouts(self, simu, production, timeout_trials):
        #idx = [i for i, j in enumerate(to) if not np.isfinite(j).all()]
        production = np.delete(np.array(production), timeout_trials)
        simu = np.delete(simu, timeout_trials, 2) #delete all timeout trials
        return simu, production


    def statistics(self, s, production):
        std_s = np.std(s, 2)
        production = np.array(production)*self.dt
        mean = np.mean(production)
        std = np.std(production)
        return mean, std, std_s
    
    def get_frames(self, reset_lst, trial=False):
        if trial:
            reset_lst = np.delete(reset_lst, 3)
            
        m_start = reset_lst[1::3]
        m_stop = reset_lst[2::3]
        
        p_start = reset_lst[2::3]
        p_stop = reset_lst[3::3]
        return zip(p_start, p_stop, m_start, m_stop)

    
    def plot_trials(self, simu, res, production, timeout_trials, stimulus=None, trial=False):
        if production.size==0:
            print('all trials timeout')
            return None

        where = (np.where(np.array(res)==1)[0]-1)*self.dt
        steps = np.arange(len(simu[:,0]))* self.dt
        alpha=1
        
        fig, ax = plt. subplots(4,1, sharex=True, figsize=(10,7))
        if trial:
            alpha = 0.01
            ntrial=5
            trial_production = self.first_duration+1*self.dt+stimulus+1*self.dt+production[ntrial]*self.dt
            where = np.concatenate((np.array([0]),where,np.array([trial_production])))
            ax[0].plot(steps, simu[:,0, ntrial], c='b', linewidth=0.7,  alpha=0.5)
            ax[1].plot(steps, simu[:,1, ntrial], 'blue', linewidth=0.7, alpha=0.5)
            ax[2].plot(steps, simu[:,2, ntrial], c='b',linewidth=0.7,  alpha=0.5)
            print('Stimulus:', stimulus, ', Production trial', ntrial, '(blue)):',production[ntrial]*self.dt)
            ax[2].plot(trial_production, self.th, 'x', c='blue')
            ax[3].plot(steps, simu[:,3, ntrial], c='b',linewidth=0.7,  alpha=0.5)
        
        prod_frame = self.get_frames(where, trial)

        ax[0].plot(steps, simu[:,0], c='grey', alpha=alpha)
        ax[0].vlines(where, np.min(np.array(simu[:,0])), np.max(np.array(simu[:,0])), color='grey',alpha=0.5)
        ax[0].set_title('du/dt')
        
        ax[1].plot(steps, simu[:,1], 'grey', alpha=alpha)
        ax[1].vlines(where, np.min(np.array(simu[:,1])), np.max(np.array(simu[:,1])), color='grey', alpha=0.5)
        ax[1].set_title('dv/dt')
        
        ax[2].plot(steps, simu[:,2], 'grey', alpha=alpha)
        ax[2].hlines(self.th, 0, simu.shape[0]*self.dt,linestyle='--', color='lightgray')
        ax[2].vlines(where, np.min(np.array(simu[:,2])), np.max(np.array(simu[:,2])), color='grey',alpha=0.5)
        ax[2].set_title('dy/dt')
        
        ax[3].plot(steps, simu[:,3], 'grey', alpha=alpha)
        ax[3].vlines(where, np.min(np.array(simu[:,3])), np.max(np.array(simu[:,3])), color='grey',alpha=0.5)
        ax[3].set_title('dI/dt')
        ax[3].set_xlabel('Time (ms)')
        
        for p_start, p_stop, m_start, m_stop in prod_frame:
            for a in [0,1,2,3]:
                ax[a].axvspan(p_start, p_stop, facecolor='r', alpha=0.1)
                ax[a].axvspan(m_start, m_stop, facecolor='b', alpha=0.1)

        plt.tight_layout()
        
        print('timeouts =', len(timeout_trials))



    def plot_behavior(self, stimuli_range, K, initI, ax=None):
        mean_lst = []
        std_lst = []
        
        for stim in stimuli_range:
            simu, res, production, timeout_trials = self.simulate_parallel(stim, K, initI)
            #print('stimulus', stim, '; timout trials', len(timeout_trials))
            mean, std, _ = self.statistics(simu, production)
            mean_lst.append(mean)
            std_lst.append(std)
            
        reg = linregress(stimuli_range, mean_lst)
        
        if ax is None:
            plt.errorbar(stimuli_range, mean_lst, yerr=std_lst, fmt='-o', c='k')
            plt.plot( [stimuli_range[0]-100,stimuli_range[-1]+100],[stimuli_range[0]-100,stimuli_range[-1]+100], c='grey', linestyle='--')
            plt.text(np.min(stimuli_range)-100, np.max(stimuli_range)+100, 'slope='+str(round(reg[0],3)))
            plt.xlabel('Stimulus (ms)')
            plt.ylabel('Production (ms)')
        else:
            subplot = ax.errorbar(stimuli_range, mean_lst, yerr=std_lst, fmt='-o', c='k')
            ax.plot( [stimuli_range[0]-100,stimuli_range[-1]+100],[stimuli_range[0]-100,stimuli_range[-1]+100], c='grey', linestyle='--')
            ax.text(np.min(stimuli_range)-100, np.max(stimuli_range)+50, 'slope='+str(round(reg[0],3))+', to='+str(len(timeout_trials)))
            return subplot
     
        
    def meas_prod_times(self, simu, prod, stimulus, sample):
        meas_start = int(self.first_duration/self.dt+1)
        meas_stop = int(self.first_duration/self.dt+1+stimulus/self.dt) #without flashes
        measure = simu[meas_start:meas_stop]
        measure = np.mean(measure,2)

        prod_lst = []
        for i,p in enumerate(prod):
            p_start=int(self.first_duration/self.dt+stimulus/self.dt+2)
            p_end=int(self.first_duration/self.dt+stimulus/self.dt+2+p)

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
            lengths_m = [0]
            lengths_p = [0]
            for stim in stimuli_range:
                simulation, res, production, timeout_trials = self.simulate_parallel(stim, K, initI)
                measure, prod_sampled = self.meas_prod_times(simulation, production, stim, int(stim/self.dt+10))
                meas_all.extend(measure)
                prod_all.extend(prod_sampled)
                lengths_m.append(measure[:,:3].shape[0])
                lengths_p.append(prod_sampled[:,:3].shape[0])                
        
        pca_m = PCA()
        MPCA = pca_m.fit_transform(meas_all)
        pca_p = PCA()
        PPCA = pca_p.fit_transform(prod_all)
        
        MPCA_lst = []
        PPCA_lst = []
        
        for i in range(len(stimuli_range)):
            MPCA_lst.append(MPCA[sum(lengths_m[:i+1]):sum(lengths_m[:i+2])])
            PPCA_lst.append(PPCA[sum(lengths_p[:i+1]):sum(lengths_p[:i+2])])
            
        return MPCA_lst, PPCA_lst    
        
            
    def plot_PCA(self, stimuli_range, K, initI, colors, separate=False):
        MPCA_lst, PPCA_lst = self.PCA(stimuli_range, data=None, K=K, initI=initI, experiment=False)
        TOP = [3,2,2]
        MIDDLE = [3,2,4]
        BOTTOM = [3,2,6]
        
        
        if separate: 
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1,2,1, projection='3d')
        else:
            fig = plt.figure(figsize=(8,6))
            ax = plt.axes(projection='3d')
        for i in reversed(range(len(stimuli_range))):
            ax.plot3D(MPCA_lst[i][:,0],  MPCA_lst[i][:,1],  MPCA_lst[i][:,2], c=colors[i], alpha=0.5, marker='.')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.title('Measurement')
        
        
        
        if separate: 
            ax = fig.add_subplot(*TOP)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(MPCA_lst[i][:,0], c=c, alpha=0.5, marker='.')
            ax = fig.add_subplot(*MIDDLE)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(MPCA_lst[i][:,1], c=c, alpha=0.5, marker='.')
            ax = fig.add_subplot(*BOTTOM)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(MPCA_lst[i][:,2], c=c, alpha=0.5, marker='.')
        plt.tight_layout()
        
         
        if separate: 
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1,2,1, projection='3d')
        else:
            fig = plt.figure(figsize=(8,6))
            ax = plt.axes(projection='3d')
        for i in reversed(range(len(stimuli_range))):
            ax.plot3D(PPCA_lst[i][:,0],  PPCA_lst[i][:,1],  PPCA_lst[i][:,2], c=colors[i], alpha=0.5, marker='.')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.title('Production')
        
        if separate: 
            ax = fig.add_subplot(*TOP)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(PPCA_lst[i][:,0], c=c, alpha=0.5, marker='.')
            ax = fig.add_subplot(*MIDDLE)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(PPCA_lst[i][:,1], c=c, alpha=0.5, marker='.')
            ax = fig.add_subplot(*BOTTOM)
            for i, c in zip(reversed(range(len(stimuli_range))), reversed(colors)):
                ax.plot(PPCA_lst[i][:,2], c=c, alpha=0.5, marker='.')
        plt.tight_layout()
        
        return MPCA_lst, PPCA_lst