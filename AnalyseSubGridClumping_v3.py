# coding: utf-8
''' 
Author: Michele Bianco (M.Bianco@sussex.ac.uk)
Modified by Michele 03/03/2019 
'''

import numpy as np, pandas as pd, os, sys, glob as gb, time, math
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['figure.figsize'] = 10, 8 
plt.rcParams['font.size'] = 25
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.linewidth'] = 1.5
labels_size = 20

from tools21cm import *
from scipy.optimize import curve_fit
from scipy.stats import lognorm, norm

from joblib import Parallel, delayed

from clump_functions import func, ExpDist, getSmartBin, SubgridClumping, MergeImages, SumDiffSizeArray, ClumpingFile, coarse_grid2, PrintTimeElapsed


print '   _____                .__               .__        '
print '  /  _  \   ____  ____  |  | ___.__. _____|__| ______'
print ' /  /_\  \ /    \ \__ \ |  |<   |  |/  ___/  |/  ___/'
print '/    |    \   |  \/ __ \|  |_\___  |\___ \|  |\___ \ '
print '\____|____/___|__(______/____/\____/______|__/______|'
print '_________  .__                           .__                 '
print '\_   ___ \ |  |   __ __   _____  ______  |__|  ____    ____  '
print '/    \  \/ |  |  |  |  \ /     \ \____ \ |  | /    \  / ___\ '
print '\     \____|  |__|  |  /|  Y Y  \|  |_| ||  ||   |  \/ /_/  |'
print ' \________/|____/|____/ |__|_|__/|   __/ |__||___| _/\___  / '
print '                                 |__|               /_____/\n' 

np.set_printoptions(precision=3)
set_verbose(False)
timestr = time.strftime('%y%m%d')

'''**********Modify as needed*************'''
boxSize = 6.3
meshSize = 1200

#subgrid_path = '/research/prace/sph_smooth_cubepm_130328_12_3456_13Mpc_ext2/nc1200/'
subgrid_path = '/home/m/mb/mb756/inputs/sph_smooth_cubepm_130314_6_1728_'+str(boxSize)+'Mpc_ext2/nc'+str(meshSize)+'/'  # new data
#subgrid_path = '/research/prace/sph_smooth_cubepm_old_6_1728_'+str(boxSize)+'Mpc/global/so/nc'+str(meshSize)+'/'   # old data

# Redshift must to be sorted in decreasing order, meanwhile num of coarsening in increasing order!
redshift = get_dens_redshifts(subgrid_path)[::-1]
#redshift = np.array([21.062, 16.095, 13.557, 10.877, 8.892, 7.305])
#redshift = np.array([7.305, 8.892])
noc = 8			 # number of coarsening desired
resLB = 500./300	 # resolution to achieve (large box)
resSB = boxSize/meshSize # resolution small box

MaxBin = 5    # number of density bin (look for 'nr_BINS' in the method 'getSmartBin')

#output_path = '../outputs/output_subgrid/AnClumpMic_'+timestr+'_'+str(boxSize)+'Mpc_nc'+str(meshSize)+'-so-n-MCPR_4bins/'
output_path = '/its/home/mb756/SubGrid_test/outputs/output_subgrid/AnClumpMic_190829_6.3Mpc_nc1200-so-n-MCPR_NEW2/noc%d_bins%d_test/' %(noc, MaxBin)
'''*****************************************'''
conv.set_sim_constants(boxsize_cMpc = boxSize) # Set conversion factors.  

# Create output directories
if not (os.path.exists(subgrid_path)):
    print('Base path not found')
    sys.exit()
elif not (os.path.exists(output_path)):
    os.system('mkdir %s' %output_path)
    os.system('mkdir %splot/' %output_path)
    os.system('mkdir %sparams/' %output_path)
    os.system('mkdir %scoarsed_data/' %output_path)
    os.system('mkdir %scoarsed_data/xy/' %output_path)
    os.system('mkdir %scoarsed_data/dc/' %output_path)
else:
    print '\nOutput directory already exist. Attenttion maybe it overwrites files!\n'

# Save redshift and resolution files
np.savetxt(output_path+'z_%.1fMpc_nc%d.txt' %(boxSize, meshSize), redshift, fmt='%.3f')

# The parameter data will be saved 
name_q = output_path+'params/par_quad_'+str(boxSize)+'_nc'+str(meshSize)+'.txt'
name_l = output_path+'params/par_lognorm_'+str(boxSize)+'_nc'+str(meshSize)+'.csv'
name_m = output_path+'params/meanClumpData'+str(boxSize)+'_nc'+str(meshSize)+'.txt'

meanClumpData = np.zeros((redshift.size, 2))
quadParams = np.zeros((redshift.size, 7))
lognormParams = pd.DataFrame(index=redshift, columns=['bin%d' %i for i in range(MaxBin)])

# Txt with description of the main parameters
text_description = '   _____                .__               .__        \n  /  _  \   ____  ____  |  | ___.__. _____|__| ______\n /  /_\  \ /    \ \__ \ |  |<   |  |/  ___/  |/  ___/\n/    |    \   |  \/ __ \|  |_\___  |\___ \|  |\___ \ \n\____|____/___|__(______/____/\____/______|__/______|\n_________  .__                           .__                 \n\_   ___ \ |  |   __ __   _____  ______  |__|  ____    ____  \n/    \  \/ |  |  |  |  \ /     \ \____ \ |  | /    \  / ___\ \n\     \____|  |__|  |  /|  Y Y  \|  |_| ||  ||   |  \/ /_/  |\n \________/|____/|____/ |__|_|__/|   __/ |__||___| _/\___  / \n                                 |__|               /_____/\n\nSubgrid files path:\n %s\nOutput results path:\n %s\n\nRedshift range from %.3f to %.3f for a total of %d snapshot\nNumber of Coarsening:\t%d\nCoarsed resolution:\t%.3f\nOverlap of sub-grid volumes to represent coarse size:\t%.2f\n\nParameter files are:\n * par_mean_%s_nc%s.txt\t\tshape: %s\n\tfor C(z) = C0 * exp(c1*z + c2*z^2) + 1\tsee file description for more details\n * par_quad_%s_nc%s.npy\t\tshape: %s\t\t\t[z]\n\t[a, b, c, err a, b and c]\n * par_lognorm_%s_nc%s.npy\t\tshape: %s\t\t[z][MaxBin]\n\t[median, sigma, density bin low, high limit and middle value]\n' %(subgrid_path, output_path, max(redshift), min(redshift), len(redshift), noc, resLB, (resLB/resSB-float(meshSize)/noc)/(resLB/resSB)*100, str(boxSize), str(meshSize), np.shape(meanClumpData), str(boxSize), str(meshSize), np.shape(quadParams), str(boxSize), str(meshSize), np.shape(lognormParams))
with open(output_path+'text_description.txt', "w") as file:
    file.write(text_description)

# some cool colors
colors = ['indianred', 'darkorange', 'olivedrab', 'royalblue', 'darkviolet']   

def ReadingFiles(i, z, path='./'):
    # reading files per redshift
    print '_'*70+'\n     ****** Reading density and clumping file for z=%.3f ******\n' %z
    print 'Density file Data:'
    dfile = DensityFile(path+'%.3fn_all.dat' %z)
    print 'The size of the mesh is (', dfile.mesh_x, dfile.mesh_y, dfile.mesh_z, ')'
    print "Average density [cgs]:\t\t%.3e"  %np.mean(dfile.cgs_density)
    print "Max and Min density [cgs]:\t%.3e   %.3e\n" %(np.amax(dfile.cgs_density), np.amin(dfile.cgs_density))
    print 'Clumping file Data:'
    cfile = ClumpingFile(path+'%.3fc_all.dat' %z)
    print 'The size of the mesh is (', cfile.mesh_x, cfile.mesh_y, cfile.mesh_z, ')'
    print "\nFilter off unreasonablely clumping values:"
    cfile.raw_clumping[cfile.raw_clumping >= 100] = 1.
    cfile.raw_clumping[cfile.raw_clumping < 1] = 1.
    print "Average raw clumping factor after filtering:\t%.3e" %np.mean(cfile.raw_clumping)
    print "Maximum and Minimum raw clumping after filtering:\t%.3e   %.3e\n" %(np.amax(cfile.raw_clumping), np.amin(cfile.raw_clumping))
    return dfile, cfile#, dtotfile


def AnalysisClumping(dfile, cfile, i_z, val_z):
    print '_'*70+'\n     ****** Manipulating the data for noc=%d ******\n' %noc
    c_raw = cfile.raw_clumping		# for c_all
    n_igm = dfile.raw_density		# for n_all
   
    Coarse_nigm = coarse_grid2(n_igm, resLB, resSB, noc)	# coarse-grained density, i.e: <n>_coarse
    sq_Coarse_nigm = Coarse_nigm**2		# squared coarse-grained density, i.e: <n>^2_coarse
    print '...The square of coarse-grained density has been evaluated.'

    cXsq_nigm = c_raw*n_igm**2						# i.e: <n^2> = C_all * <n_all>^2
    Coarse_sq_nigm = coarse_grid2(cXsq_nigm, resLB, resSB, noc)	# i.e: <n^2>_coarse
    print '...The coarse-grained square of density has been evaluated.'
    
    Coarse_c = Coarse_sq_nigm/sq_Coarse_nigm				# <C_sim >_coarse
    overd = Coarse_nigm/np.mean(dfile.raw_density)			# 1+<delta>_coarse
    x = overd.flatten() #np.log10(overd.flatten())		# = log(1+<delta>)
    y = Coarse_c.flatten() #np.log10(Coarse_c.flatten())		# = log(C)
    
    # Mean coarsed clumping
    mean_c = np.mean(Coarse_c)
    meanClumpData[i_z] = [val_z, mean_c]
    print 'Mean coarsed Clumping:\t%.3f' %mean_c
    print '...Coarse-grained clumping factor and overdensity are calculated.\n'   

    # Save data to test realisation
    save_cbin(output_path+'coarsed_data/dc/%.3fn_all_nc%d.dat' %(val_z, noc), Coarse_nigm)
    save_cbin(output_path+'coarsed_data/dc/%.3fc_all_nc%d.dat' %(val_z, noc), Coarse_c)    
    # Normalized coarsed density and clumping files
    save_cbin(output_path+'coarsed_data/xy/%.3fx_all_nc%d.dat' %(val_z, noc), x)
    save_cbin(output_path+'coarsed_data/xy/%.3fy_all_nc%d.dat' %(val_z, noc), y)

    print '     ****** Fitting the correlation for noc=%d ******\n' %noc
    n_x_bins = getSmartBin(np.sort(x), str(MaxBin))
    x_bins = np.vstack((n_x_bins[:-1], n_x_bins[1:])).T
    x_bins_mid = np.array([])
    
    '''QUARDRATIC FIT'''
    popt, pcov = curve_fit(func, x, y)
    perr = np.sqrt(np.diag(pcov))		# the standard deviation errors on the parameters

    # Save quadratic results
    quadParams[i_z] = np.array([val_z, popt[0], popt[1], popt[2], perr[0], perr[1], perr[2]])
    print 'The fitted parameters of the QUADRATIC function:\na=%.3f   b=%.3f   c=%.3f\n' %(popt[0], popt[1], popt[2])

    # Data for plotting quadratic results
    x_plot = np.linspace(x.min(), x.max(), 100)
    yfit_quad = func(x_plot, *popt)
    yfit_quad_err = np.sqrt((perr[0]*x_plot**2)**2 + (perr[1]*x_plot)**2 + (perr[2])**2 )

    fig, ax = plt.subplots(figsize=(23, 15))
    ax.set_xlim(min(n_x_bins), max(n_x_bins)), ax.set_ylim(min(y)*0.8, max(y)*1.2)
    #ax.set_xlim(-0.17, 0.14), ax.set_ylim(1.1, 1.8)
    
    divider = make_axes_locatable(ax)
    axHisty = divider.append_axes("right", 3., pad=0.1, sharey=ax)    # y-variable histogram (right)

    plt.setp(axHisty.get_yticklabels(), visible=False)
    plt.setp(axHisty.get_xticklabels(), visible=False)
    
    E_arr, SD_arr = np.array([]), np.array([])
    print 'The fitted parameters of the LOGNORMAL distribution:'
    
    for i_bin, val_bin in enumerate(x_bins):
        y_in_bin = np.array([])			# clumping values which density is within the selected x_bins
	x_in_bin = np.array([])			# array to calculate a more ponder mean for lognorm mean
	for val_x, val_y in zip(x,y):
            if(val_bin[0] <= val_x <= val_bin[1]):
                y_in_bin = np.append(y_in_bin, val_y)
		x_in_bin = np.append(x_in_bin, val_x)
	    else: pass
        x_bins_mid = np.append(x_bins_mid, np.mean(x_in_bin))
        y_in_bin.sort()
        
        '''LOGNORMAL DISTRIBUTION FIT'''
        sigma, location, median = lognorm.fit(y_in_bin, floc=0)
        mu = np.log(median)
        expect_val, standard_dev = np.exp(mu + 0.5*sigma**2), np.exp(mu + 0.5*sigma**2)*np.sqrt(np.exp(sigma**2)-1)
        E_arr, SD_arr = np.append(E_arr, expect_val), np.append(SD_arr, standard_dev)

        print 'E[X]=%.3f  SD[X]=%.3f\tfor density: [%.3f, %.3f]' %(expect_val, standard_dev, val_bin[0], val_bin[1])
        lognormParams.loc[val_z, 'bin%d' %i_bin] = ([mu, sigma, val_bin[0], val_bin[1], x_bins_mid[i_bin]])
        
        # fit lognorm
        yfit_log = np.linspace(y_in_bin.min(), y_in_bin.max(), 40) 	# x value for the lognorm plot
        lognorm_fit = lognorm.pdf(yfit_log, sigma, location, median) 

        '''PLOT LOGNORMAL PDF'''
        txt = r'$\it{log}(1+\langle\delta\rangle _{cell}) \in$[%.3f, %.3f]'"\n"'$\mu=%.3f$  $\sigma=%.3f$  $\it{log}(\mathcal{C})_{peak}=%.3f$' %(val_bin[0], val_bin[1], mu, sigma, median)
        axHisty.hist(y_in_bin, bins='auto', alpha=0.3, density=True, color=colors[i_bin], label=txt, orientation='horizontal')    
        axHisty.plot(lognorm_fit, yfit_log, color=colors[i_bin], lw=3)
        axHisty.legend(loc='lower right', bbox_to_anchor=(-0.03, 0.01), prop={'size':labels_size-2}, framealpha=1.)
    
    '''PLOT CORRELATION''' 
    #ax.set_xlabel (r'$(1+\delta)$'), ax.set_ylabel(r'$\it{log}(\mathcal{C})$')
    ax.set_xlabel (r'$1+\delta$'), ax.set_ylabel(r'$\mathcal{C}$')
    txt = 'z=%.3f'"\t"r'$Res_{crs}$=%.3f\,Mpc' %(val_z, resLB/0.7)
    ax.set_title(txt)
    #ax.text(0.95, 0.95, txt, size=labels_size-3, horizontalalignment='right', verticalalignment='top', transform=ax.gca().transAxes)

    ax.plot(x, y, 'x', markersize=10, color='black', label='noc data')
    ax.axhline(y=mean_c, lw=3, c='black', ls='--', label='GCM')
    ax.plot(x_plot, yfit_quad, 'r-', label='DCM', lw=3)
    ax.errorbar(x_bins_mid, E_arr, yerr=SD_arr, capsize=12, lw=2, markersize=15, fmt='.-', c='blue', label='SCM')
    for val_bin in n_x_bins:
        ax.axvline(x=val_bin, lw=2, ls='-', c='grey')
    ax.legend(loc=2, borderpad=0.5, prop={'size':labels_size+2})
    ax.set_yscale('log'), ax.set_xscale('log')
    fig.savefig(output_path+'plot/%.3f_%.1f_%.3f_nc%d_clumping.png' %(val_z, boxSize, resLB, meshSize), bbox_inches='tight')
    plt.clf()
    
    # Save result in files
    np.savetxt(name_m, meanClumpData, fmt='%.3f\t%.3f', header='z\tmean C')
    np.savetxt(name_q, quadParams, fmt='%.3f\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e', header='z\ta\tb\tc\terr_a\terr_b\terr_c')
    lognormParams.to_csv(path_or_buf=name_l, float_format='%.4f')
    return 0
   

def MeanClumping(z, meanClump, numberofcoarse):
    print '_'*70+'\n     ****** Fitting the Mean Clumping Parameters *******\n'
    
    z, meanData = meanClump.T
    popt, pcov = curve_fit(SubgridClumping, z, meanData, p0=[0.001, -0.1, meanData[0]])
    perr = np.sqrt(np.diag(pcov))
    
    # Plot data
    plt.title('noc=%.3f' %numberofcoarse)
    plt.plot(z, SubgridClumping(z, *popt), 'r-')
    plt.plot(z, meanData, 'bx')
    plt.ylabel(r'$\mathcal{C}_{GCM}(z)$')
    plt.xlabel('$z$')
    txt = 'Fit Parameters:\n  c2 = %.3f\n  c1 = %.3f\n  C0 = %.1f' %(popt[0], popt[1], popt[2])
    plt.text(0.75, 0.95, txt, size=labels_size-3, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
    plt.savefig(output_path+'plot/meanClumping_noc%.3f.png' %numberofcoarse, bbox_inches='tight')
    plt.clf()
    name_m_par = output_path+'params/par_mean_'+str(boxSize)+'_nc'+str(meshSize)+'.txt'
    np.savetxt(name_m_par, np.expand_dims(np.append(popt, perr), axis=1).T, delimiter='\t', fmt='%.4e', header='For clumping C(z) = C0 * exp(c1*z + c2*z^2) + 1\nc2\tc1\tC0\terr_c2\terr_c1\terr_C0')
    print '...done\n'+'_' * 70
    

# ------------------------------Main Function--------------------------------------
t_start = time.time()

print '_' * 70
for i, z in enumerate(redshift):
    clock_time = time.time()
    dfile, cfile = ReadingFiles(i, z, subgrid_path)
    clock_time = PrintTimeElapsed(clock_time, 'Opening Files')
    AnalysisClumping(dfile, cfile, i, z)
    clock_time = PrintTimeElapsed(clock_time, 'tasks for z=%.3f' %z)

MeanClumping(redshift, meanClumpData, noc)
clock_time = PrintTimeElapsed(clock_time, 'Creating Mean Clumping Parameters')

print '_' * 70
t_end = PrintTimeElapsed(t_start, 'ANALYSIS CLUMPING')
# ---------------------------------------------------------------------------------
