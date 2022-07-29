import matplotlib as mpl
#mpl.rcParams['font.size'] = 20
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pickle
from matplotlib import rc

#rc('text', usetex=True)
#font = {'family' : 'helvetica',
#        'weight' : 'bold',
#        'size'   : 11}
#mpl.rc('font', **font)
##sns.set_style("white", {'font.family':'sans-serif', 
#               'font.sans-serif':['helvetica'], 'weight':'bold',
#               'size': 11})
#sns.axes_style("white", {'font.family':['sans-serif'], 
#               'font.sans-serif':['helvetica'], 
#               'weight':'bold',
#               'size': 11})
#sns.set(style={'font.family':'sans-serif', 
#               'font.sans-serif':['helvetica'], 
#               'weight':'bold',
#               'size': 11, 
#               'figure.facecolor': 'white'})
#
#sns.set_theme(style="white", rc={'font.family':['sans-serif'], 
#               'font.sans-serif':['helvetica'],'font.serif':['helvetica']})


stimulus_range_s = [400, 450, 500, 550, 600, 650, 700]
stimulus_range_l = [700, 750, 800, 850, 900, 950, 1000]

stimulus_lst_short = np.loadtxt('stimlst_short_400_700_7_a.txt', dtype=int)
stimulus_lst_long = np.loadtxt('stimlst_long_700_1000_7_a.txt', dtype=int)

def load_data(short_path, long_path):
    short_data = []
    long_data = []
    with open(short_path, 'rb') as short:
        with open(long_path, 'rb') as long:
            try:
                while True:
                    short_data.append(pickle.load(short))
                    long_data.append(pickle.load(long))
            except EOFError:
                pass
    return short_data, long_data



# p = K, th, tau, delay
def to_matrix(result_lst, p1, p2, result):
    matrix = np.zeros((p1,p2))
    for i in range(len(result_lst)):
        matrix.flat[i] = result_lst[i][result]
    return matrix



def create_parameter_plot(short, long, shortlong, p1, p1_lst, p2, p2_lst, cmap, n_colors=20, norm=False):
    cmap = sns.color_palette(cmap, n_colors=n_colors)
    minmin = np.min([np.nanmin(short), np.nanmin(long), np.nanmin(shortlong)])
    maxmax = np.max([np.nanmax(short), np.nanmax(long), np.nanmax(shortlong)])
    print(minmin, maxmax)
    
    fig, ax = plt.subplots(1,3, figsize=(15,4), sharex=True, sharey=True)
    if norm == 'log':
        norm = matplotlib.colors.LogNorm(vmin=minmin, vmax=maxmax)
        
    if not norm:
        h1 = sns.heatmap(short, xticklabels=p2_lst, yticklabels=p1_lst, ax=ax[0], cmap = cmap, cbar=True)#,  vmin=minmin, vmax=maxmax)
        h2 = sns.heatmap(long, xticklabels=p2_lst, yticklabels=p1_lst, ax=ax[1], cmap = cmap, cbar=True)#,  vmin=minmin, vmax=maxmax)
        h3 = sns.heatmap(shortlong, xticklabels=p2_lst, yticklabels=p1_lst, ax=ax[2],  cmap = cmap)#, vmin=minmin, vmax=maxmax) #cbar_ax= cbar_ax,
    else:
        cbar_ax = fig.add_axes([.91, .25, .01, .5]) #x, y, breite, höhe
        h1 = sns.heatmap(short, xticklabels=p2_lst, yticklabels=p1_lst, ax=ax[0], cmap = cmap, cbar=False,  vmin=minmin, vmax=maxmax, norm=norm)
        h2 = sns.heatmap(long, xticklabels=p2_lst, yticklabels=p1_lst, ax=ax[1], cmap = cmap, cbar=False,  vmin=minmin, vmax=maxmax, norm=norm)
        h3 = sns.heatmap(shortlong, xticklabels=p2_lst, yticklabels=p1_lst, ax=ax[2],  cmap = cmap, cbar_ax= cbar_ax, vmin=minmin, vmax=maxmax, norm=norm)
    
    h1.set_xlabel(p2)
    h1.set_ylabel(p1)
    h1.set_title('short')

    h2.set_xlabel(p2)
    h2.set_title('long')

    h3.set_xlabel(p2)
    h3.set_title('short~long')
    
    plt.show()
    
def figure_create_parameter_plot(short, long, shortlong, p1, p1_lst, p2, p2_lst, cmap, n_colors=20, norm=False):
    cmap = sns.color_palette(cmap, n_colors=n_colors)
    minmin = np.min([np.nanmin(short), np.nanmin(long), np.nanmin(shortlong)])
    maxmax = np.max([np.nanmax(short), np.nanmax(long), np.nanmax(shortlong)])
    print(minmin, maxmax)
    
    fig, ax = plt.subplots(1,2, figsize=(5,2), sharex=True, sharey=True)
    if norm == 'log':
        norm = mpl.colors.LogNorm(vmin=minmin, vmax=maxmax)
        
    if not norm:
        h1 = sns.heatmap(short, xticklabels=p2_lst, yticklabels=p1_lst, ax=ax[0], cmap = cmap, cbar=True)#,  vmin=minmin, vmax=maxmax)
        h2 = sns.heatmap(long, xticklabels=p2_lst, yticklabels=p1_lst, ax=ax[1], cmap = cmap, cbar=True)#,  vmin=minmin, vmax=maxmax)
        
    else:
        cbar_ax = fig.add_axes([.91, .25, .03, .6]) #x, y, breite, höhe
        h1 = sns.heatmap(short, xticklabels=p2_lst, yticklabels=p1_lst, ax=ax[0], cmap = cmap, cbar=False,  vmin=minmin, vmax=maxmax, norm=norm)
        h2 = sns.heatmap(long, xticklabels=p2_lst, yticklabels=p1_lst, ax=ax[1], cmap = cmap, cbar_ax= cbar_ax, vmin=minmin, vmax=maxmax, norm=norm)
       

    '''for ind, label in enumerate(h1.get_yticklabels()):
        if ind % 9 == 0:  # every 9th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)'''
    ax[0].locator_params(axis='y', nbins=4)
    ax[0].locator_params(axis='x', nbins=4)
    
    h1.set_xlabel(p2)
    h1.set_ylabel(p1)
    h1.set_title('short')

    h2.set_xlabel(p2)
    h2.set_title('long')
    
    plt.show()
    
    
    
def get_mse(data, K_lst, tau):
    bias2 = to_matrix(data, len(K_lst), len(tau), 'bias2')
    var = to_matrix(data, len(K_lst), len(tau), 'var')
    return bias2+var



def plot_slope(short, long, K_lst, tau, n_colors=20):
    short_ktau_slope = to_matrix(short, len(K_lst), len(tau), 'slope')
    long_ktau_slope = to_matrix(long, len(K_lst), len(tau), 'slope')
    divnorm=mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-1.2, vmax=1.2)
    # specify plot style
    figure_create_parameter_plot(short_ktau_slope, long_ktau_slope, short_ktau_slope-long_ktau_slope, 'K', K_lst, 'tau', tau, 'coolwarm', n_colors=n_colors, norm=divnorm)
    plt.show()
    

    
def plot_mse(short, long, K_lst, tau, full=True):
    plot = True
    if full:
        error_short = get_mse(short, K_lst, tau)
        error_long = get_mse(long, K_lst, tau)
    if full=='bias2':
        error_short = to_matrix(short, len(K_lst), len(tau), 'bias2')
        error_long = to_matrix(long, len(K_lst), len(tau), 'bias2')
    if full=='var':
        error_short = to_matrix(short, len(K_lst), len(tau), 'var')
        error_long = to_matrix(long, len(K_lst), len(tau), 'var')
    if full=='bias':
        error_short = to_matrix(short, len(K_lst), len(tau), 'bias')
        error_long = to_matrix(long, len(K_lst), len(tau), 'bias')
        minmin = np.min([np.nanmin(error_short), np.nanmin(error_long)])
        maxmax = np.max([np.nanmax(error_short), np.nanmax(error_long)])
        compromise=minmin
        divnorm=mpl.colors.TwoSlopeNorm(vcenter=0, vmin=compromise, vmax=-compromise)
        # specify plot style
        figure_create_parameter_plot(error_short, error_long, (error_short+error_long)/2, 'K', K_lst, 'tau', tau, 'coolwarm', n_colors=50, norm=divnorm)
        plt.show()
        plot=False
    if plot:
        # specify plot style
        figure_create_parameter_plot(error_short, error_long, (error_short+error_long)/2, 'K', K_lst, 'tau', tau, 'gist_heat', n_colors=50, norm = 'log')
        plt.show()
    


def plot_mse_total(short, long, K_lst, tau):
    short_ktau_mse = to_matrix(short, len(K_lst), len(tau), 'MSE')
    long_ktau_mse = to_matrix(long, len(K_lst), len(tau), 'MSE')
    # specify plot style
    figure_create_parameter_plot(short_ktau_mse, long_ktau_mse, (short_ktau_mse+long_ktau_mse)/2, 'K', K_lst, 'tau', tau, 'gist_heat', n_colors=50)
    plt.show()
    

    
def plot_ind_point(short, long, K_lst, tau):
    short_ktau_indp = to_matrix(short, len(K_lst), len(tau), 'ind_point')
    long_ktau_indp = to_matrix(long, len(K_lst), len(tau), 'ind_point')
    divnorm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-500, vmax=500)
    short_ktau_indp_delta =  short_ktau_indp - stimulus_range_s[int(len(stimulus_range_s)/2)]
    long_ktau_indp_delta =  long_ktau_indp - stimulus_range_l[int(len(stimulus_range_l)/2)]
    # specify plot style
    figure_create_parameter_plot(short_ktau_indp_delta, long_ktau_indp_delta, (short_ktau_indp_delta-long_ktau_indp_delta)/2, 'K', K_lst, 'tau', tau, 'coolwarm', n_colors=20, norm=divnorm)
    plt.show()
    
    
    
def get_opt_K(data, K_lst, tau, mse=False, var=False, bias=False, bias2=False):
    if var: 
        data_ = to_matrix(data, len(K_lst), len(tau), 'var')
    if bias2: 
        data_ = to_matrix(data, len(K_lst), len(tau), 'bias2')
    if bias: 
        data_ = abs(to_matrix(data, len(K_lst), len(tau), 'bias'))
    if mse: 
        data_ = get_mse(data, K_lst, tau)
    data_ = np.nan_to_num(data_, nan=np.inf)
    opt = np.nanargmin(data_, axis=0)
    opt_overall = np.where(data_==np.nanmin(data_))
    print(tau[opt_overall[1][0]], K_lst[opt_overall[0][0]])
    
    return list(zip(tau, K_lst[opt]))



def create_search_space(srange, K_lst, th_lst, tau, delay_lst):
    search_space = []

    if srange == 'short':
        stimulus_range = [400, 450, 500, 550, 600, 650, 700]
    if srange == 'long':
        stimulus_range = [700, 750, 800, 850, 900, 950, 1000]
    if K_lst== True:
        K_lst = np.arange(0.5, 10.5, 0.5)
    if th_lst== True:
        th_lst = np.arange(0.6, 0.75, 0.01)
    if delay_lst== True:
        delay_lst = np.arange(400, 1000, 50)
    if tau== True:
        tau = np.arange(60, 200, 10)
    
    i = 0
    for K in K_lst:
        for th in th_lst:
            for t in tau:
                for delay in delay_lst:
                    i += 1
                    search_space.append((stimulus_range, K, th, t, delay))
    print(i)
    return search_space