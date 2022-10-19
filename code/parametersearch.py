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

    if srange == 'mid':
        stimulus = np.loadtxt('stimlst_mid_550_850.txt', dtype=int)
    if srange == 'all':
        stimulus = np.loadtxt('stimlst_all_400_1000.txt', dtype=int)
    if srange == 'few_short':
        stimulus = np.loadtxt('stimlst_few_short_400_700.txt', dtype=int)
    if srange == 'few_all':
        stimulus = np.loadtxt('stimlst_few_all_400_1000.txt', dtype=int)
    if srange == 'extralong':
        stimulus = np.loadtxt('stimlst_extralong_900_1200.txt', dtype=int)
    if srange == 'few_short2':
        stimulus = np.loadtxt('stimlst_few_short_400_700_2.txt', dtype=int)
    

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
regime = 'range/delay0best'
sigma = 0.02
reset=50
Iinit=0.8

'''# high I
regime = 'underestimation/delayhigh'
sigma = 0.02
reset=-500
Iinit=1.02

tau = np.arange(20, 200, 10)'''

'''# range
regime = 'range'
sigma = 0.02
reset=50
Iinit=0.8

'''

###############################

pool = 24
batchsize = pool

th_lst = [0.7]
# seed_lst = [0]
seed_lst = np.arange(0, 21, 1)
K_lst = np.arange(1, 35, 1)  # np.arange(1, 22, 1) np.arange(0.5, 10.5, 0.5)
# tau = np.arange(90, 250, 10)  # np.arange(60, 200, 10)
delay_lst = [0]

tau = [170]
srange='mid'
name = 'MID_170_th7_seed_delay0'
run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, seed_lst, name)

tau = [150]
srange='all'
name = 'all_150_th7_seed_delay0'
run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, seed_lst, name)

tau = [120]
srange='few_short2'
name = 'fewshort_120_th1_seed_delay0'
run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, seed_lst, name)

tau = [160]
srange='few_all'
name = 'fewall_160_th7_seed_delay0'
run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, seed_lst, name)

tau = [230]
srange='extralong'
name = 'extralong_230_th7_seed_delay0'
run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, seed_lst, name)

tau = [130]
srange='short'
name = 'short_130_th7_seed_delay0'
run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, seed_lst, name)

tau = [200]
srange='long'
name = 'long_200_th7_seed_delay0'
run_parallel(batchsize, pool, srange, K_lst, th_lst, tau, delay_lst, seed_lst, name)
