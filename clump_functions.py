# VERSION: 04/08/2019
import numba, os, sys, glob as gb, numpy as np, pandas as pd, time, matplotlib.pyplot as plt
import operator
from numpy import exp, log, cos, sin, sqrt, pi
from PIL import Image


def SameValuesInArray(arr1, arr2):
    """ return interpolated array in decreasing order"""
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    interp_arr = np.sort(np.array(list(set(arr1).intersection(arr2))))[::-1]
    return interp_arr


def GetContours(x, y, colour='green', style=[':', '--', '-'], perc_arr=[0.68, 0.93], lw=3):
    x_edges = np.arange(min(x)-0.05, max(x)+0.05, 0.03)
    y_edges = np.arange(min(y)-0.05, max(y)+0.05, 0.03)
    hist, xedges, yedges = np.histogram2d(x, y, bins=(x_edges, y_edges))

    xidx = np.digitize(x, x_edges)-1
    yidx = np.digitize(y, y_edges)-1
    srt = np.sort((hist[xidx, yidx]))
    perc = (np.array(perc_arr)*len(x)).astype(int)
    levels = srt[perc]

    plt.contour(hist.T, extent=[xedges.min(), xedges.max(), yedges.min(
    ), yedges.max()], levels=levels, colors=colour, linestyles=style, linewidths=lw)
    plt.clabel(c, c.levels, inline=True,
               inline_spacing=10, fmt='%.2f', fontsize=16)


def PercentContours(x, y, nr_bins=None, colour='green', style=[':', '--', '-'], perc_arr=[0.99, 0.95, 0.68], lw=3):
    if(type(nr_bins) == int):
        hist, xedges, yedges = np.histogram2d(x, y, bins=nr_bins)
    else:
        #x_edges = np.arange(np.min(x), np.max(x), 1e-3)
        #y_edges = np.arange(np.min(y), np.max(y), 1e-3)
        x_edges = np.linspace(np.min(x), np.max(x), 300)
        y_edges = np.linspace(np.min(y), np.max(y), 300)
        hist, xedges, yedges = np.histogram2d(x, y, bins=(x_edges, y_edges))

    sort_hist = np.sort(hist.flatten())[::-1]
    perc = (np.array(perc_arr)*np.sum(sort_hist)).astype(int)
    levels = np.zeros_like(perc)
    
    j = -1
    for i, val in enumerate(sort_hist):
        if(np.sum(sort_hist[:i]) >= perc[j]):
            levels[j] = val
            if(j == -len(perc)):
                break
            j -= 1 
    c = plt.contour(hist.T, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], levels=levels, colors=colour, linestyles=style, linewidths=lw)
    c.levels = np.array(perc_arr)*100.
    plt.clabel(c, c.levels, inline=True,inline_spacing=10, fmt='%d%%', fontsize=16)
    #plt.draw()

def PrintTimeElapsed(t_start, mess='task'):
    """ Gives elapsed time after completing a task.
        Parameters:
            * t_start (float): the current time in seconds since the Epoch, start time defined before the call of this method
            * mess = 'task' (string): text that describe the task, will be printed in the timing message
        Returns:
            t_end (float): the current time in seconds since the Epoch
        Example (multiple task in a python file)
            >>> start_time = time.time()
            >>> ...
            >>> time_t1 = PrintTimeElapsed(start_time, task='nothing')
            >>> ...
            >>> time_t2 = PrintTimeElapsed(time_t1, task='nothing again')
    """
    t_end = time.time()
    t_elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(t_end - t_start))
    print('\nElapsed time to end %s is:\t%s' % (mess, t_elapsed))
    return t_end


def MergeImages(new_image_name, old_image_name, output_path='./', form='v', delete_old=False):
    """ Merge images togheter to create new image.
        Parameters:
            * new_image_name (string): name of the new image
            * old_image_name (string or array): name of the old images, can be a string or an array of strings, if as string then it create a list of paths matching a pathname pattern (attention to the order!)
            * output_path (string): output path save image
            * form (string or tuple): if string it can be 'v' or 'h', otherwise tuplet with the shape to optain
            * delete_old (bool): to delete old images or not
        Returns:
            nothing
    """
    if(isinstance(old_image_name, str)):
        arrsize = len(gb.glob(output_path+old_image_name+'*.png'))
        arr_images = np.array(
            [output_path+old_image_name+str(i)+'.png' for i in range(arrsize)])
    else:
        arr_images = np.array(old_image_name)
    # Open Images
    images = [Image.open(im_name) for im_name in arr_images]
    height, width, chan = np.shape(images[0])

    # identify which form is desired
    if(isinstance(form, str)):
        if(form == 'v'):
            total_height = height*len(images)
            total_width = width
            x_displ, y_displ = 0, height
            retbool = True
        elif(form == 'h'):
            total_height = height
            total_width = width*len(images)
            x_displ, y_displ = width, 0
            retbool = True
    else:
        total_height = height*form[0]
        total_width = width*form[1]
        x_displ, y_displ = width, height
        retbool = False

    # Create new empty image
    new_im = Image.new('RGB', size=(
        total_width, total_height), color=(255, 255, 255, 0))

    # Start paste old images on new empty image
    x_offset, y_offset = 0, 0
    if(retbool):
        for im in images:
            new_im.paste(im, (x_offset, y_offset))
            x_offset += x_displ
            y_offset += y_displ
    else:
        idx = 0
        for h in range(form[0]):
            for v in range(form[1]):
                new_im.paste(images[idx], (x_offset, y_offset))
                x_offset += x_displ
                idx += 1
            x_offset = 0
            y_offset += y_displ

    # Save new image
    new_im.save('%s.png' % (output_path+new_image_name))

    # Delete old image if required
    if(delete_old):
        if isinstance(old_image_name, np.ndarray):
            for im in old_image_name:
                os.system("rm %s" % (output_path+im))
        else:
            os.system("rm %s*.png" % (output_path+old_image_name))
    else:
        # old images not deleted.
        pass


def SaveMatrix(filename, matrix, size, bits=32, order='C'):
    ''' Save a matrix as binary file. Indicated for very large matrix.

        Parameters:
                * filename (string): the filename to save to
                * matrix (numpy array): the matrix to save
                * size (tuple): the matrix size, must be a tuple of integer
                * bits = 32 (integer): the number of bits in the file
                * order = 'C' (string): the ordering of the data (can be'C' for C/C++ or 'F' Fortran)
        Returns:
                Nothing

    '''
    print('Saving matrix file: %s' % filename)

    assert(bits == 32 or bits == 64)
    f = open(filename, 'wb')
    mesh = np.array(size).astype('int32')
    mesh.tofile(f)
    datatype = (np.float32 if bits == 32 else np.float64)
    matrix.flatten(order=order).astype(datatype).tofile(f)
    f.close()


def ReadMatrix(filename, bits=32, order='C', dimensions=3):
    ''' Read a binary file with three inital integers (a cbin file).

    Parameters:
        * filename (string): the filename to read from
        * bits = 32 (integer): the number of bits in the file
        * order = 'C' (string): the ordering of the data. Can be 'C'
            for C style ordering, or 'F' for fortran style.
        * dimensions (int): the number of dimensions of the data (default:3)

    Returns:
        The data as a three dimensional numpy array.
    '''

    assert(bits == 32 or bits == 64)

    f = open(filename)

    temp_mesh = np.fromfile(f, count=dimensions, dtype='int32')

    datatype = np.float32 if bits == 32 else np.float64
    data = np.fromfile(f, dtype=datatype, count=np.prod(temp_mesh))
    data = data.reshape(temp_mesh, order=order)
    return data


def SaveBinaryFile(nameFile, mesh_arr, arrData, indxord='C'):
    # reshape in depending on the index ordering
    arrData.reshape(np.shape(arrData), order=indxord)    
    # Convert into data type compatible with c2raytools
    arrMesh_32i = np.array(mesh_arr, dtype=np.int32)
    arrData_32f = arrData.astype(np.float32)
    # Open file with given name and path
    f = open(nameFile, "wb")
    new_meshData = bytearray(arrMesh_32i)    # convert array into binary
    new_arrData = bytearray(arrData_32f)
    f.write(new_meshData)
    f.write(new_arrData)
    f.close()
    print('Saved binary file: %s' % (nameFile.rpartition("/")[-1]))


def OpenBinaryFile(filename, dimensions=3, floatpoint=32, reshape=False):
    print('Open file: %s' % (filename.rpartition("/")[-1]))
    f = open(filename)
    if(dimensions != 1):
        temp = tuple(np.fromfile(f, count=dimensions, dtype='int32'))
    else:
        pass
    datatype = np.float32 if floatpoint == 32 else np.float64
    raw_data = np.fromfile(f, dtype=datatype)
    f.close()
    if(reshape):
        output_data = raw_data.reshape(temp)
    else:
        output_data = raw_data
    return output_data

# Removing duplicate columns and rows from a 2D array


def UniqueRows(arr):
    arr = np.ascontiguousarray(arr)
    unique_arr = np.unique(arr.view([('', arr.dtype)]*arr.shape[1]))
    return unique_arr.view(arr.dtype).reshape((unique_arr.shape[0], arr.shape[1]))


def OrderNdimArray(arr, idx_to_sort):
    ''' Order N-dim array giving the index of sorting
    Parameters:
        * arr (array): N-dim or simple array to sort
        * idx_to_sort (int): sorteing array index
    Returns:
        sorted array by desired array index
        '''
    new_arr = np.array(sorted(arr.T, key=operator.itemgetter(idx_to_sort)))
    return new_arr


def DeleteEmptyRows(arr):
    bool_arr = np.array([any(val) for val in arr])
    new_arr = arr[bool_arr == True]
    return new_arr


def SumDiffSizeArray(arr_a, arr_b):
    if len(arr_a) < len(arr_b):
        arr_c = arr_b.copy()
        arr_c[:len(arr_a)] += arr_a
    else:
        arr_c = arr_a.copy()
        arr_c[:len(arr_b)] += arr_b
    return arr_c


def FindNearest(arr, val):
    """ Find nearest index (integer) of a value(s) in an array.
        Parameters:
            * arr (array): array to search through.
            * val (array or float): the value(s) to look for.
        Returns:
            The index(ices) of the value(s) and the value in the array.
    """
    try:
        len(val)
        val = np.array(val)
        idx, value = [], []
        for a in val:
            idx_closest = (abs(arr-a)).argmin()
            value = np.append(value, arr[idx_closest])
            idx = np.append(idx, idx_closest).astype(np.int)
    except:
        idx = (abs(arr-val)).argmin()
        value = arr[idx]
    return idx, value


def CreateGif(filename, array, fps=5, scale=1., fmt='gif'):
    ''' Create and save a gif or video from array of images.
        Parameters:
            * filename (string): name of the saved video
            * array (list or string): array of images name already in order, if string it supposed to be the first part of the images name (before iteration integer)
            * fps = 5 (integer): frame per seconds (limit human eye ~ 15)
            * scale = 1. (float): ratio factor to scale image hight and width
            * fmt (string): file extention of the gif/video (e.g: 'gif', 'mp4' or 'avi')
        Return:
            * moviepy clip object
    '''
    if(isinstance(array, str)):
        arrsize = len(gb.glob(array+'*.png'))
        array = [array+str(i)+'.png' for i in range(arrsize)]
    else:
        pass
    from moviepy.editor import ImageSequenceClip
    filename += '.'+fmt
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    if(fmt == 'gif'):
        clip.write_gif(filename, fps=fps)
    elif(fmt == 'mp4'):
        clip.write_videofile(filename, fps=fps, codec='mpeg4')
    elif(fmt == 'avi'):
        clip.write_videofile(filename, fps=fps, codec='png')
    else:
        print('Error! Wrong File extension.')
        sys.exit()
    command = os.popen('du -sh %s' % filename)
    print(command.read())
    return clip


''''''''''''''' Michele analysis_clumping_michele.py Functions '''''''''''''''
# The mesh size is often indivisible by the ratio. To have an integer for the number of fine-grained cells merged to a coarse-grained cell, remainders at the edge of the simulation box were not considered. For mesh size 1200^3 and 14 coarse-grained cells per dimension for the simulation box, the total number of fine-grained cells taken is 85x14 = 1190. 10 out of 1200 fine-grained cells per dimension were not taken into account.


def getSmartBin(arrData, numcorse):
    ''' It creates a total number of 'nr_BINS' bins, containing the same number of particle (num_part) in each one.
        The bin size choise depend on the number of corsening.
        It is very important that, a priori, you know the number of coarsening that you require, so to manualy change the bin size, 'nr_BINS', to be the same for for both of your required number of coarsening.
        This because when you will use the simulate_clumping_michele.py, weighted lognorm and quad parameter will be calculated, so we want to avoid to mix-up parameter from two unrelated density bin 
        N.B: this is something to work out in simulate_clumping_michele.py (18/01/18)'''
    if isinstance(numcorse, str):
        # if > 7 than numbcorse is the number of bins that you require
        nr_BINS = int(numcorse)
    elif numcorse == 3:
        nr_BINS = 2
    elif numcorse == 4:
        nr_BINS = 2
    elif numcorse == 6:
        nr_BINS = 4
    elif numcorse == 7:
        nr_BINS = 4
    elif numcorse > 7:
        nr_BINS = 5
    else:
        print('Not possible to create histogram, number of coarsening is too small\n')
    num_part = int(len(arrData)/nr_BINS)
    bins = np.array([arrData[0]])
    if nr_BINS != 1:
        for i in range(1, nr_BINS):
            bins = np.append(bins, arrData[i*num_part-1])
    else:
        pass
    bins = np.append(bins, arrData[-1])
    if numcorse != -1:
        print('number of density bins: %d' %nr_BINS)
        print('Approximate number of particle per density bin: %d\n' %num_part)
    return bins


def SubgridClumping(z, a, b, C):
    return C*exp(b*z + a*z**2)+1


''''''''''''''' Michele simulate_clumping_michele.py Functions '''''''''''''''


def func(x, a, b, c):
    y = a*x**2 + b*x + c
    try:
        if(y < 0):
            y = 0.
    except:
        y[y < 0] = 0.
    return y


def ExpDist(z, A, b):
    return A*exp(b*z) + 1


def GoodLookingPlotData(x, y, size):
    if(size != 1):
        len_data_plot = int(len(x)/size)
        x_plot = np.zeros((len_data_plot))
        y_plot = np.zeros((len_data_plot))
        size = int(round(size))
        for i in range(1, len_data_plot):
            x_plot[i] = x[size*(i-1)]
            y_plot[i] = y[size*(i-1)]
    else:
        x_plot, y_plot = x, y
    return x_plot, y_plot


@numba.jit
def generateLogNormNoise(median, sigma, prev_u0=1, prev_u1=1):
    ''' Box-Muller method to sample lognorm distribution with uniform random values '''
    u0 = np.random.uniform(0, prev_u0)
    u1 = np.random.uniform(0, prev_u1)
    z0 = sqrt(-2.0 * log(u0)) * cos(2*pi * u1)
    z1 = sqrt(-2.0 * log(u0)) * sin(2*pi * u1)   # for a bivariate distribution
    X0 = exp(z0 * sigma)*median
    #X1 = exp(z1 * sigma)*median
    return X0, u0, u1


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''' Determing weight family '''''''''''''''''''''''''''''''''


def detWeight(xData, x):
    ''' Determining the weights (w_sup, w_inf) of the parameter with values at 'xData' for interpretations of value 'x' (P.S: for decreasing redshift and increasing noc, i.e: decreasing res).'''
    w_sup = 0
    w_inf = 0
    j = 0
    if(x == xData[0]):
        w_sup = 0
        w_inf = 1
        j = 0
    elif(x == xData[-1]):
        w_sup = 1
        w_inf = 0
        j = len(xData)-2
    elif x in xData:
        w_sup = 0
        w_inf = 1
        j = np.where(xData == x)[0][0]
    else:
        while (w_sup == 0) & (w_inf == 0):
            if x > xData[j+1]:
                w_sup = (xData[j] - x)/(xData[j] - xData[j+1])
                w_inf = 1 - w_sup
            j += 1
        j -= 1
    return w_sup, w_inf, j


def detMeanPar(ressup, resinf, resj, parData):
    par = np.zeros(6)
    for i in range(0, 6):
        par[i] = parData[resj+1][i]*ressup + parData[resj][i]*resinf
    return par


def detQuadPar(zsup, zinf, zj, ressup, resinf, resj, parData):
    ''' Using the detWeight function, interpret the parameters for the quardratic fitting of the desired distribution. '''
    par = np.zeros(8)
    for i in range(0, 8):
        a = (parData[zj+1][resj+1][i]*zsup +
             parData[zj][resj+1][i]*zinf)*ressup
        b = (parData[zj+1][resj][i]*zsup + parData[zj][resj][i]*zinf)*resinf
        par[i] = a + b
    return par


def detLognormPar(zsup, zinf, zj, ressup, resinf, resj, parData):
    ''' Using the detWeight function, interpret the parameters for the lognormal distribution for a certain number of bins of the desired distribution. '''
    range_j, range_i = np.shape(
        parData[0, 0])  # get shape from the first element (index_z = 0 and index_noc = 0)
    par = np.zeros_like(parData[0, 0])
    for j in range(0, range_j):
        for i in range(0, range_i):
            a = (parData[zj+1][resj+1][j][i]*zsup +
                 parData[zj][resj+1][j][i]*zinf)*ressup
            b = (parData[zj+1][resj][j][i]*zsup +
                 parData[zj][resj][j][i]*zinf)*resinf
            par[j][i] = a + b
    return par


def TheParametrizator(redshift_in, redshift_sub, res_in, res_sub, MeanParams, QuadParams, LogParams, out='./', Lbox=None):
    print('Interpolating parameters for the quardratic equation, lognormal distribution and mean coarsed clumping.')
    w_res_sup, w_res_inf, index_res = detWeight(res_sub, res_in)
    Weighted_MeanParams = detMeanPar(w_res_sup, w_res_inf, index_res, MeanParams)
    Weighted_QuadParams = np.empty(QuadParams[:, 0, :].shape)
    Weighted_LogParams = np.empty(LogParams[:, 0, :, :].shape)
    for i, z in enumerate(redshift_in):
        w_z_sup, w_z_inf, index_z = detWeight(redshift_sub, z)
        Weighted_QuadParams[i] = detQuadPar(
            w_z_sup, w_z_inf, index_z, w_res_sup, w_res_inf, index_res, QuadParams)
        Weighted_LogParams[i] = detLognormPar(
            w_z_sup, w_z_inf, index_z, w_res_sup, w_res_inf, index_res, LogParams)

    np.save('%spar_quad_%dMpc' % (out, Lbox), Weighted_QuadParams)
    np.save('%spar_lognorm_%dMpc' % (out, Lbox), Weighted_LogParams)
    np.savetxt('%spar_mean_%dMpc.txt' % (out, Lbox), Weighted_MeanParams.T,
               header='For clumping C(z)=C*exp(b*z+a*z^2)+1\nin the following column variable are: a, b, C, err_a, err_b and err_C')
    return Weighted_MeanParams[:3], Weighted_QuadParams[:, :3], Weighted_LogParams


def TheParametrizator2(redshift_in, redshift_sub, MeanParams, QuadParams, LogParams, out='./', Lbox=None):
    print('Interpolating parameters for the quardratic equation, lognormal distribution and mean coarsed clumping.')
    Weighted_QuadParams = pd.DataFrame(index=redshift_in, columns=QuadParams.columns)
    Weighted_LogParams = pd.DataFrame(index=redshift_in, columns=LogParams.columns)

    for z in redshift_in:
        if(z in redshift_sub):
            Weighted_QuadParams.loc[z, :] = QuadParams.loc[z, :]
            Weighted_LogParams.loc[z, :] = LogParams.loc[z, :]
        else:
            w_z_sup, w_z_inf, index_z = detWeight(redshift_sub, z)
            val_z_sup, val_z_inf = redshift_sub[index_z+1], redshift_sub[index_z]
            Weighted_QuadParams.loc[z, :] = w_z_inf*QuadParams.iloc[index_z, :]+w_z_sup*QuadParams.iloc[index_z+1, :]
            for val_log in LogParams.columns:
                Weighted_LogParams.loc[z, val_log] = w_z_inf*LogParams.loc[val_z_sup, val_log]+w_z_sup*LogParams.loc[val_z_inf, val_log]
    
    # Save weighted parameters data for LB   
    np.savetxt('%spar_mean_%dMpc.txt' % (out, Lbox), MeanParams.values.T, delimiter='\t', fmt='%.4e', header='For clumping C(z)=C*exp(b*z+a*z^2)+1\n'+'\t'.join(MeanParams.columns.values))
    np.savetxt('%spar_quad_%dMpc.txt' % (out, Lbox), np.hstack((np.expand_dims(redshift_in, axis=1), Weighted_QuadParams.values)), fmt='%.3f\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e', header='z\t'+'\t'.join(QuadParams.columns.values))
    Weighted_LogParams.to_csv(path_or_buf='%spar_lognorm_%dMpc.csv' % (out, Lbox), float_format='%.4f')
    
    return MeanParams.loc[0, 'c2':'C0'], Weighted_QuadParams.loc[:, 'a':'c'], Weighted_LogParams

@numba.jit
def detWeightPDFparam(xData, x, y_quad, median_arr, sigm_arr):
    ''' Determining the weights within a PDF plot for a weighted mu and sigma and avoid steps in the PDF interpretations. '''
    w_sup = 0
    w_inf = 0
    j = 0
    return_median = 0.
    return_sigm = 0.
    if(x <= np.min(xData)):
        # log_y = log(y_quad) if log(y_quad)!=-np.inf else 0. #log(0.0001)
        w_sup = 0
        w_inf = 1 - w_sup
        return_median = y_quad*w_inf + median_arr[0]*w_sup
        return_sigm = sigm_arr[0]
    elif(x >= xData[-1]):
        #log_y = log(y_quad) if log(y_quad)!=-np.inf else 0.
        w_inf = 0
        w_sup = 1 - w_inf
        return_median = y_quad*w_sup + median_arr[-1]*w_inf
        return_sigm = sigm_arr[-1]
    else:
        while (w_sup == 0) & (w_inf == 0):
            if(x < xData[j+1]):
                # quadratic regression of the parameters within bins
                w_sup = (x - xData[j])**2/(xData[j+1] - xData[j])**2
                w_inf = 1 - w_sup
            j += 1
        j = j-1
        return_median = median_arr[j]*w_inf + median_arr[j+1]*w_sup
        return_sigm = sigm_arr[j]*w_inf + sigm_arr[j+1]*w_sup
    return return_median, return_sigm


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''' clumping_file.py '''''''''''''''''''''''''''''''''


class ClumpingFile:
    '''
    A CubeP3M clumping factor file.

    Use the read_from_file method to load a clumping factor file, or 
    pass the filename to the constructor.

    Some useful attributes of this class are:

    * raw_clumping (numpy array): the clumping factor in simulation units
    * z (float): the redshift of the file (-1 if it couldn't be determined from the file name)

    '''

    def __init__(self, filename=None):
        '''
        Initialize the file. If filename is given, read data. Otherwise,
        do nothing.

        Parameters:
            * filename = None (string): the file to read from.
        Returns:
            Nothing
        '''
        if filename:
            self.read_from_file(filename)

    def read_from_file(self, filename):
        '''
        Read data from file.

        Parameters:
            * filename (string): the file to read from.
        Returns:
            Nothing
        '''

        print('Reading clumping factor file: %s ...' %
              (filename.rpartition("/")[-1]))
        self.filename = filename
        # Read raw data from clumping factor file
        f = open(filename, 'rb')
        temp_mesh = np.fromfile(f, count=3, dtype='int32')
        self.mesh_x, self.mesh_y, self.mesh_z = temp_mesh
        self.raw_clumping = np.fromfile(f, dtype='float32')
        self.raw_clumping = self.raw_clumping.reshape(
            (self.mesh_x, self.mesh_y, self.mesh_z), order='F')

        f.close()
        print('Average raw clumping:\t\t%.3e' % np.mean(self.raw_clumping))
        print('Min and Max clumping:\t%.3e  %.3e' % (np.min(self.raw_clumping), np.max(self.raw_clumping)))

        # Store the redshift from the filename
        try:
            import os.path
            name = os.path.split(filename)[1]
            self.z = float(name.split('c_')[0])
        except:
            print('Could not determine redshift from file name')
            self.z = -1
        print('...done')


''''''''''''''''''''''''' coarse.py Functions '''''''''''''''''''''''''''''''''


def coarse_grid(file_input, filename, mesh, noc):
    ''' It coarse a grid from shape=(mesh mesh mesh) to shape=(noc noc noc), a tot of ratio^3 original cells are used to average and create noc cube.
        Parameters:
            * file_input (narray)	: 3D array of data
            * filename=None (string)	: if given look for the C2Ray files (dfile or cfile)
            * mesh (int)		: dimension of original data
            * noc (int)			: dimension of coarsening of data
    '''
    ratio = int(float(mesh)/noc)
    if isinstance(filename, str):
        if filename == "cfile":
            arr = file_input.raw_clumping
        if filename == "dfile":
            arr = file_input.raw_density
    else:
        arr = file_input

    coarse_data = np.zeros([noc, noc, noc])  # nocXnocXnoc
    for i in range(noc):
        for j in range(noc):
            for k in range(noc):
                cut = arr[(ratio*i):(ratio*(i+1)), (ratio*j):(ratio*(j+1)), (ratio*k):(ratio*(k+1))]
                coarse_data[i][j][k] = np.mean(cut)
    return coarse_data

def coarse_grid1(arr, noc):
    ''' It coarse a grid from shape=(mesh mesh mesh) to shape=(noc noc noc), a tot of ratio^3 original cells are used to average and create noc cube.
        Parameters:
            * arr (narray)	: 3D array of data
            * noc (int)			: dimension of coarsening of data
    '''
    ratio = int(float(arr.shape[0])/noc)
    coarse_data = np.zeros([noc, noc, noc])  # nocXnocXnoc
    for i in range(noc):
        for j in range(noc):
            for k in range(noc):
                cut = arr[(ratio*i):(ratio*(i+1)), (ratio*j):(ratio*(j+1)), (ratio*k):(ratio*(k+1))]
                coarse_data[i][j][k] = np.mean(cut)
    return coarse_data

def coarse_grid2(arr_to_coarse, ResLB, ResSB, noc=None):
    ''' It coarse a grid from shape=(mesh mesh mesh) to shape=(noc noc noc), a tot of ratio^3 original cells are used to average and create noc cube.
        Parameters:
            * arr_to_coarse (narray)	: 3D array of data
            * ResLB (float)				: Large box resolution
            * ResSB (float)				: Small box resolution
            * noc (int)					: Desired dimension of data coarsening 
    '''
    meshSB = arr_to_coarse.shape[0]
    if (noc == None):
        noc = int(round(meshSB/ResLB*ResSB, 0))
        overlap_perc = 0
    else:     
        overlap_perc = round((ResLB/ResSB-float(meshSB)/noc)/(ResLB/ResSB)*100, 2)
    print(overlap_perc, '% overlap')
    ncount = int(round(ResLB/ResSB, 0))
    ratio = int(round(float(meshSB)/noc, 0))
    coarse_data = np.zeros([noc, noc, noc])  # shape=(noc noc noc)
    for i in range(noc):
        for j in range(noc):
            for k in range(noc):
                cut = arr_to_coarse[ratio*i:ratio*i+ncount, ratio*j:ratio*j+ncount, ratio*k:ratio*k+ncount]
                coarse_data[i,j,k] = np.mean(cut)
    return coarse_data

