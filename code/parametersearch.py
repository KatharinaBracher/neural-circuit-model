from onetwogo import Params
from onetwogo.experiment_simulation import ExperimentSimulation
from multiprocessing import Pool
# import pickle
import time
import numpy as np

def execute(d):
    '''performs simulation and returns behavioral data '''

    stimulus_lst, K, th, t, delay, seed = d # stimulus_range
    print("executing with", K, th, t, delay, "= K, th, t, delay")

    print(seed)
    np.random.seed(seed)
    params = Params(ntrials=500, delay=delay, tau=t, th=th, sigma=sigma)
    expsim = ExperimentSimulation(params)
    # stimulus_lst = expsim.generate_stimulus_lst(stimulus_lst)
    # stimulus_lst = expsim.find_stimulus_lst(stimulus_lst, 20, 0.9)

    # return BehavioralData def save to disk
    sim_result = expsim.simulate(stimulus_lst, K)  # TODO seed
    result = sim_result.create_behavioral_data()
    return (result, K)


# create seach space
def create_search_space(srange, K_lst, th_lst, tau, delay_lst):
    '''creates search space of relevant hyperparameter'''

    search_space = []

    if srange == 'short':
        # stimulus = [400, 450, 500, 550, 600, 650, 700]
        stimulus = np.loadtxt('stimlst_short_400_700_7_a.txt', dtype=int)
    if srange == 'long':
        # stimulus = [700, 750, 800, 850, 900, 950, 1000]
        stimulus = np.loadtxt('stimlst_long_700_1000_7_a.txt', dtype=int)

    seed = 0
    for K in K_lst:
        for th in th_lst:
            for t in tau:
                for delay in delay_lst:
                    search_space.append((stimulus, K, th, t, delay, seed))
                    # seed+=1
    return search_space


def run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, name):
    '''
    runs simulation for simulationspace
    batchsize: int
        batch on which simulation executing and results are kept in working memory
    pool: int
        number of parallel computations
    srange: str
        indicating short or long interval
    K_lst: np.array
        array of memory parameters
    th_lst: np.array
        array of thresholds
    tau: np.array
        array of time constants
    delay_lst: np.array
        array of delay times between trials
    name: str
        name of pickle saved to server
    '''
    
    search_space = create_search_space(srange, K_lst, th_lst, tau, delay_lst)
    with open('/home/bracher/results/%s-%s-output.pickle' % (name, time.strftime("%Y%m%d-%H%M%S")), 'ab') as fp:
        for i in range(0, len(search_space), batchsize):
            print(i, i+batchsize, 'of', len(search_space))
            with Pool(pool) as p:
                results = p.map(execute, search_space[i:i + batchsize])
                for result, K in results:
                    print('writing to disk as', name)
                    result.write_to_disk(fp, srange, K)

K_lst = np.arange(1, 15.5, 0.5)  # np.arange(1, 22, 1) np.arange(0.5, 10.5, 0.5)
th_lst = np.arange(0.6, 0.75, 0.01)
delay_lst = np.arange(400, 1000, 50)
tau = np.arange(90, 120, 5)  # np.arange(60, 200, 10)

sigma = 0.02

# choose parameter range #############################################################
srange = 'short'
# K_lst = [8.0]*250
th_lst = [0.75]
# tau = [100]
delay_lst = [700]

#name = 'LONG_SAME_K8_TAU100_TH08_DEL700'
name = 'SHORT_K4-15TAU_th075_del700_sig02_fix_seed'
# name = 'LONG_KTAU_th08_del700'

pool = 20
batchsize = pool

run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, name)
