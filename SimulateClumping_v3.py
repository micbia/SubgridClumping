# coding: utf-8
''' 
Author: Michele Bianco (M.Bianco@sussex.ac.uk)
Modified by Michele 29/08/2019 
'''
import numba
from numba import f8    # float64

import numpy as np, pandas as pd, os, sys, glob as gb, time
from scipy.stats import lognorm
from numpy import pi, log, log10, sqrt, exp, cos, sin
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams['figure.figsize'] = 13, 11 
plt.rcParams['agg.path.chunksize'] = 20000
plt.rcParams['font.size'] = 25
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.linewidth'] = 1.5
labels_size = 20

from tools21cm import set_verbose, get_dens_redshifts, conv, DensityFile
from tools21cm.helper_functions import _get_redshifts_in_range

from clump_functions import func, SubgridClumping, SaveBinaryFile, OpenBinaryFile, detWeight, detWeightPDFparam, GoodLookingPlotData, PrintTimeElapsed, PercentContours, TheParametrizator2, generateLogNormNoise, MergeImages


print '  _________.__                 .__            __           '
print ' /   _____/|__|  _____   __ __ |  |  _____  _/  |_   ____  '
print ' \_____  \ |  | /     \ |  |  \|  |  \__  \ \   __\_/ __ \ '
print ' /        \|  ||  Y Y  \|  |  /|  |__ / __ \_|  |  \  ___/ '
print '/_______  /|__||__|_|__/|____/ |____/(_____/ |__|   \___/  '
print '_________  .__                           .__               '
print '\_   ___ \ |  |   __ __   _____  ______  |__|  ____    ____  '
print '/    \  \/ |  |  |  |  \ /     \ \____ \ |  | /    \  / ___\ '
print '\     \____|  |__|  |  /|  Y Y  \|  |_| ||  ||   |  \/ /_/  |'
print ' \________/|____/|____/ |__|_|__/|   __/ |__||___| _/\___  / '
print '                                 |__|               /_____/  \n'

timestr = time.strftime('%y%m%d')
np.set_printoptions(precision=3)

'''**********Modify as needed*************'''
# LB: Large, low resolute box size in Mpc/h and mesh size (eg: '*n_all.dat')
boxSize_LB = 244.
meshSize_LB = 250
resLB = round(boxSize_LB/meshSize_LB, 3)
LB_path = '/home/m/mb/mb756/inputs/sph_smooth_cubepm_130329_10_4000_244Mpc_ext2_test/global/so/nc250/'

# SB: Small, high resolute box size [Mpc/h] and mesh size (output of analysis_clumping.py)
boxSize_SB = 6.3
meshSize_SB = 1200
SB_path = '/its/home/mb756/SubGrid_test/outputs/output_subgrid/AnClumpMic_190829_6.3Mpc_nc1200-so-n-MCPR_NEW2/noc13_bins5_test/' 
noc = int(SB_path[SB_path.find('/noc')+4:SB_path.find('_bins')])
MaxBin = int(SB_path[-2:-1])


# OUTPUT PATH
output_path = '/its/home/mb756/SubGrid_test/outputs/output_simclump/SimClumpMic_%s_%dMpc_nc%d/' %(timestr, boxSize_LB, meshSize_LB)
'''*****************************************'''

# Enable verbose output and set conversion factors. 
set_verbose(True)
conv.set_sim_constants(boxsize_cMpc = boxSize_LB)  

# Create output directories
if not (os.path.exists(LB_path)):
    print 'Input path not found'
    sys.exit()
elif not (os.path.exists(output_path)):
    os.system('mkdir %s' %output_path)
    os.system('mkdir %sscat/' %output_path)
    os.system('mkdir %squadfit/' %output_path)
    os.system('mkdir %sdensity_data/' %output_path)
    os.system('mkdir %spdf/' %output_path)
else:
    print '\nOutput directory already exist. ATTENTION IT CAN OVERWRITES FILES!'

redshift_SB = np.loadtxt('%sz_%sMpc_nc%d.txt' %(SB_path, str(boxSize_SB), meshSize_SB))
#redshift_LB = (_get_redshifts_in_range(get_dens_redshifts(LB_path), redshift_SB.min(), redshift_SB.max(), False)[::-1])
redshift_LB = np.array([21.062, 16.095, 13.557, 10.877, 8.892, 7.305])
np.savetxt('%sz_%dMpc_nc%d.txt' %(output_path, boxSize_LB, meshSize_LB), redshift_LB, fmt='%.3f')
print 'ATTENTION! Redshift must to be sorted in DECREASING ORDER!\n'

print "The distribution parameters are fitted for number of coarsening %d and redshift range [%.3f; %.3f]" %(noc, redshift_SB[0], redshift_SB[-1])
print 'The large box simulation resolution size is %.3f Mpc/h for redshfit range [%.3f ; %.3f]' %(resLB, redshift_LB[0], redshift_LB[-1])

print "...Reading the mean, quadratic and lognormal distribution parameters and its related quantaties.\n"
meanParamsFile = pd.read_csv('%sparams/par_mean_%s_nc%d.txt' %(SB_path, str(boxSize_SB), meshSize_SB), sep='\t', skiprows=1).rename(columns={'# c2': 'c2'})
quadParamsFile = pd.DataFrame(np.loadtxt('%sparams/par_quad_%s_nc%d.txt' %(SB_path, str(boxSize_SB), meshSize_SB), usecols=(1,2,3,4,5,6)), index=redshift_SB, columns=['a', 'b', 'c', 'err_a', 'err_b', 'err_c']).rename_axis('z')
lognormParamsFile = pd.read_csv('%sparams/par_lognorm_%s_nc%d.csv' %(SB_path, str(boxSize_SB), meshSize_SB), index_col=0, converters={'bin%d' %i: lambda string: np.array(string[1:-1].split(', '), dtype=float) for i in range(MaxBin)})


def OverplotOriginalCoarsedData(z, noc):
    z_w_sup, z_w_inf, z_index = detWeight(redshift_SB, z)

    if(z_w_inf == 0):
        arr_overplot = np.array([redshift_SB[z_index+1]])
        markers = ['^']
        factors = [1]
    elif(z_w_sup == 0):
        arr_overplot = np.array([redshift_SB[z_index]])
        markers = ['v']
        factors = [1]
    else:
        arr_overplot = np.array([redshift_SB[z_index], redshift_SB[z_index+1]])
        markers = ['^', 'v']
        factors = [1/z_w_inf, 1/z_w_sup]
    
    print '\nOverploting original coarsed data'
    for i, val in enumerate(arr_overplot):
        x_overplt = OpenBinaryFile(SB_path+'coarsed_data/xy/%.3fx_all_nc%d.dat' %(val, noc))
        y_overplt = OpenBinaryFile(SB_path+'coarsed_data/xy/%.3fy_all_nc%d.dat' %(val, noc))
        x_overplot_gl, y_overplot_gl = GoodLookingPlotData(x_overplt, y_overplt, 1)
        plt.scatter(x_overplot_gl, y_overplot_gl, c='black', marker='x', s=20) 
    return 'done'


@numba.jit(f8[:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:]))
def NumbaLoops(x_arr, med_arr, sig_arr, xbs, xbs_m, yq):
    y_arr = np.zeros_like(x_arr)
    for i in tqdm(range(x_arr.size)):        
        median_avrg, sigma_avrg = detWeightPDFparam(xbs_m, x_arr[i], yq[i], med_arr, sig_arr)
        y_arr[i], u0, u1 = generateLogNormNoise(median_avrg, sigma_avrg)
        """
        for j in range(med_arr.size):
            if(xbs[j,0] <= x_arr[i] <= xbs[j,1]):
                #y_arr[i], u0, u1 = generateLogNormNoise(med_arr[j], sig_arr[j])
                y_arr[i], u0, u1 = generateLogNormNoise(yq[i], sig_arr[j])
            elif(x_arr[i] < xbs[j,0]):
                #y_arr[i], u0, u1 = generateLogNormNoise(med_arr[0], sig_arr[0])
                y_arr[i], u0, u1 = generateLogNormNoise(yq[i], sig_arr[0])
            elif(x_arr[i] > xbs[j,1]):
                #y_arr[i], u0, u1 = generateLogNormNoise(med_arr[-1], sig_arr[-1])
                y_arr[i], u0, u1 = generateLogNormNoise(yq[i], sig_arr[-1])
        """    
    return y_arr



def SimulateClumping(z, mean_parData, quad_parData, lognorm_parData):  
    print '_' * 70+'\n*********** Reading Model Parameters *************'
    
    # Load pre-computed and weighted parameters for redshift
    mean_par = mean_parData.T
    lognorm_par = pd.DataFrame(np.array([[lognorm_parData.loc[z, :].values[i][j] for j in range(5)] for i in range(lognorm_parData.columns.size)]), index=lognorm_parData.columns, columns=['mu', 'sig', 'dleft', 'dright', 'dmid'])
    quad_par = quad_parData.loc[z, :]
    
    # LOGNORMAL PARAMETERS
    mu_arr = lognorm_par['mu'].values
    sigma_arr = lognorm_par['sig'].values
    x_bins = lognorm_par.loc[:, 'dleft':'dright'].values
    x_bins_mid = lognorm_par['dmid'].values
    expval_arr = exp(mu_arr+0.5*sigma_arr**2)      # EXPECTED VALUE
    std_arr = expval_arr*sqrt(exp(sigma_arr**2)-1)      # STANDARD DEVIATION
    
    print("\nBy interpretation, the quadratic fit of the distribution at redshift %.3f and number of coarsening %d is:\n\tf(x) = %.3f*x^2 + %.3f*x + %.3f" %(z, noc, quad_par['a'], quad_par['b'], quad_par['c']))
    print("and by interpretation, the fit of the mean coarsed Clumping at number of coarsening %d is:\n\tC(z) = %.2f*exp(%.3f*z + %.4f*z^2) + 1\n" %(noc, mean_parData['C0'], mean_parData['c1'], mean_parData['c2']))
    
    print('*********** Opening Density file *************\nDensity file Data:')
    dfile = DensityFile(LB_path + str("%0.3f" %z) + 'n_all.dat')
    print('The size of the mesh is (%d, %d, %d)\n' %(dfile.mesh_x, dfile.mesh_y, dfile.mesh_z))
    dnorm = dfile.raw_density/np.mean(dfile.raw_density)    # Normalize the density data.
    x = dnorm.flatten() #log10(dnorm).reshape(meshSize_LB**3)   # x = log10(1+<delt>)
    
    if(np.sum(np.isnan(dnorm)) != 0):
        print('there are NaN in file: %.3fn_all.dat' %z)
        sys.exit()
    elif(np.sum(np.isnan(x)) != 0):
        print('there are NaN in the lognormal values of file: %.3fn_all.dat' %z)
        sys.exit()
    else:
        SaveBinaryFile('density_data/logdens_z%.3f.dat' %z, [meshSize_LB, meshSize_LB, meshSize_LB], x)

    y_mean = SubgridClumping(z, *mean_parData) #log10(SubgridClumping(z, *mean_parData))
    print('Mean coarsed Clumping:\t%.3f\n' %(10**y_mean))
    y_quad = func(x, *quad_par)
    print('Evaluating BoxMuller Random Field:')
    y_log = NumbaLoops(x, np.exp(mu_arr), sigma_arr, x_bins, x_bins_mid, y_quad)
    
    if(np.sum(np.isnan(y_log)) != 0):
        print('there are NaN in file y_log at %.3f' %z)
        print(x[i], y_quad[i], y_log[i], median_avrg, sigma_avrg)
	sys.exit()
    elif(np.sum(np.isnan(y_quad)) !=0):
        print('there are NaN in file y_quad at %.3f' %z)
        sys.exit()
    else: pass
    
    fig, ax = plt.subplots(x_bins_mid.size, figsize=(10,8))
    for i in range(x_bins_mid.size):
        y_log_in_bin = y_log[(x >= x_bins[i,0])*(x <= x_bins[i,1])]
        n, bins = np.histogram(y_log_in_bin, bins=200, density=True)
        yfit_log = np.linspace(y_log_in_bin.min(), y_log_in_bin.max(), 100)
        
        # Plot with parameters
        lognorm_fit_par = lognorm.pdf(yfit_log, sigma_arr[i], 0.0, np.exp(mu_arr)[i])
        ax[i].plot(yfit_log, lognorm_fit_par, 'b', label="$\sigma=%.3f$\n$\mu=%.3f$\n$\delta \in$ [%.3f, %.3f]" %(sigma_arr[i], np.exp(mu_arr)[i], x_bins[i,0], x_bins[i,1]))

        # Plot from fitting histogram
        sigma, location, median = lognorm.fit(y_log_in_bin, floc=0)
        lognorm_fit = lognorm.pdf(yfit_log, sigma, location, median)
        ax[i].plot(yfit_log, lognorm_fit, 'r', label="$\sigma=%.3f$\n$\mu=%.3f$\n$\delta \in$ [%.3f, %.3f]" %(sigma, np.log(median), x_bins[i,0], x_bins[i,1]))

        # Plot histogram
        ax[i].hist(y_log_in_bin, bins=300, density=True, color='g', histtype='stepfilled');
        ax[i].legend()
    plt.savefig('pdf/plotComparePDF_z%.3f_noc%d.png' %(z, noc), bbox_inches='tight')
    fig.clf()
 
    # Data for good looking plots
    x_quad_plot = np.linspace(x.min(), x.max(), 30)
    y_quad_plot = func(x_quad_plot, *quad_par)
           
    print('\nPloting realisation data, models and contours:')
    t_contour = time.time()
    
    # Plot clumping realizations
    fig = plt.figure(figsize=(13,11))
    n, xedges, yedges, img = plt.hist2d(x, y_log, bins=300, cmap='Greens', norm=LogNorm())   # cmap = Greens, summer
    
    # Plot contours
    PercentContours(x, y_log, nr_bins=300, style='-', colour='lightgreen', perc_arr=[0.95, 0.68, 0.38])
    PrintTimeElapsed(t_contour, 'Plotting Contours')

    # Overplot original coarsed data
    if (z >= min(redshift_SB)):
        OverplotOriginalCoarsedData(z, noc)
    else:
        pass
    
    # Plot models
    plt.axhline(y=y_mean, linewidth=0.5, ls='-', color='black', label='GCM')		# GCM
    plt.plot(x_quad_plot, y_quad_plot, 'r-', label='DCM')				# DCM
    plt.errorbar(x_bins_mid, expval_arr, yerr=std_arr, fmt='o-', color='cornflowerblue', label='SCM')		# SCM
    for b in x_bins.flatten():
        plt.axvline(x=b, linewidth=0.5, ls='--', color='grey') # plot bin limits
    
    # Plot setup
    #plt.xlim(x.min()*0.9, np.amax(x_bins)+0.5), plt.ylim(y_log.min(), y_log.max()*0.6)
    xrang = {21.062:[0.5, 2.2], 16.095:[0.47, 2.8], 13.557:[0.398, 3.98], 10.877:[0.316, 4.78], 8.892:[0.239, 6.3], 7.305:[0.2, 7.08]} 
    yrang = {21.062:4.5, 16.095:11, 13.557:16, 10.877:42, 8.892:110, 7.305:300.} 
    plt.xlim(xrang[z][0], xrang[z][1]), plt.ylim(1, yrang[z])
    plt.xlabel (r'$\rm{log}(1 + \langle \delta \rangle)$'), plt.ylabel(r'$log(C)$')
    plt.title('z=%.3f' %z, size=labels_size)
    plt.legend(loc=4, borderpad=0.5, prop={'size':labels_size-5}) 
    # Save Plot
    plt.savefig('scat/plotSimCorr_z%.3f_noc%d.png' %(z, noc), bbox_inches='tight')     # saved in 'scat/'
    plt.clf()
    
    # Save extrapolated data
    print('\nSaving data files.')
    name_data_quad = 'quadfit/z%.3f_quad.dat' %z
    SaveBinaryFile(name_data_quad, [meshSize_LB, meshSize_LB, meshSize_LB], y_quad, indxord='F')
    name_data_scat = 'scat/z%.3f_scat.dat' %z
    SaveBinaryFile(name_data_scat, [meshSize_LB, meshSize_LB, meshSize_LB], y_log, indxord='F')
    return 'done'

# ---------------------------------- Main ------------------------------------------
print('_'*70)
t_start = time.time()
os.chdir(output_path)

# Weight Parameters 
print('*********** Interpolating Parameters *************')
meanParamsFile_w, quadParamsFile_w, lognormParamsFile_w = TheParametrizator2(redshift_LB, redshift_SB, meanParamsFile, quadParamsFile, lognormParamsFile, Lbox=boxSize_LB)
t_clock = PrintTimeElapsed(t_start, 'Weight Parameters Values')

# Loop over redshift to simulate clumping
for z in redshift_LB:
    SimulateClumping(z, meanParamsFile_w, quadParamsFile_w, lognormParamsFile_w)
    t_clock = PrintTimeElapsed(t_clock, 'task for z=%.3f' %z)
print('_'*70)
PrintTimeElapsed(t_start, 'FINISH Simulated Clumping')

print('Creating Merged image for few relevant redshift:')
os.chdir(output_path+'scat/')
arr_plot = np.array(['plotSimCorr_z%.3f_noc%d.png' %(z, noc) for z in [21.062, 16.095, 13.557, 10.877, 8.892, 7.305]])
MergeImages(new_image_name='plot%dc' %boxSize_LB, old_image_name=arr_plot, form=(3,2), output_path='/its/home/mb756/SubGrid_test/outputs/output_result/Result_500Mpc_f5_8.2pS_300_stochastic_CAll/', delete_old=False)
# ---------------------------------------------------------------------------------
