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
    params = Params(ntrials=500, Iinit=Iinit, delay=delay, tau=t, th=th, sigma=sigma, IF=reset)
    expsim = ExperimentSimulation(params)
    # stimulus_lst = expsim.generate_stimulus_lst(stimulus_lst)
    # stimulus_lst = expsim.find_stimulus_lst(stimulus_lst, 20, 0.9)

    # return BehavioralData def save to disk
    sim_result = expsim.simulate(stimulus_lst, K)  # TODO seed
    result = sim_result.create_behavioral_data()
    return (result, K, seed)


# create seach space
def create_search_space(srange, K_lst, th_lst, tau, delay_lst, seed_lst):
    '''creates search space of relevant hyperparameter'''

    search_space = []

    if srange == 'short':
        # stimulus = [400, 450, 500, 550, 600, 650, 700]
        stimulus = np.loadtxt('stimlst_short_400_700_7_a.txt', dtype=int)
    if srange == 'long':
        # stimulus = [700, 750, 800, 850, 900, 950, 1000]
        stimulus = np.loadtxt('stimlst_long_700_1000_7_a.txt', dtype=int)

    for K in K_lst:
        for seed in seed_lst:
            for th in th_lst:
                for t in tau:
                    for delay in delay_lst:
                        search_space.append((stimulus, K, th, t, delay, seed))
                    # seed+=1
    return search_space


def run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, seed_lst, name):
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
    
    search_space = create_search_space(srange, K_lst, th_lst, tau, delay_lst, seed_lst)
    with open('/home/bracher/results/'+regime+'/%s-%s-output.pickle' % (name, time.strftime("%Y%m%d-%H%M%S")), 'ab') as fp:
        for i in range(0, len(search_space), batchsize):
            print(i, i+batchsize, 'of', len(search_space))
            with Pool(pool) as p:
                results = p.map(execute, search_space[i:i + batchsize])
                for result, K, seed in results:
                    print('writing to disk as', name)
                    result.write_to_disk(fp, srange, K, seed)

# intermediate I
regime = 'intermediateI'
sigma = 0.02
reset=50
Iinit=0.8

K_lst = np.arange(1, 22, 0.5)  # np.arange(1, 22, 1) np.arange(0.5, 10.5, 0.5)
th_lst = np.arange(0.6, 0.7, 0.75)
delay_lst = np.arange(400, 1000, 50)
tau = np.arange(100, 230, 10)  # np.arange(60, 200, 10)

# high I
'''regime = 'highI'
sigma = 0.02
reset=-500
Iinit=1.02

K_lst = np.arange(1, 18, 0.5)
th_lst = np.arange(0.05, 0.1, 0.2)
tau = np.arange(20, 150, 10)'''

# choose parameter range #############################################################
srange = 'long'
# K_lst = [8.0]*250
th_lst = [0.7]
tau = [130]
delay_lst = [700]
seed_lst = np.arange(0, 21, 1)
# seed_lst = [0]

#name = 'LONG_SAME_K8_TAU100_TH08_DEL700'
name = 'LONG_K1-22_TAU130_th07_del700_sig02_seed'
# name = 'SHORT_K20-30_tau200_th07_del700_sig02_fixed_seed'

pool = 24
batchsize = pool

run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, seed_lst, name)
