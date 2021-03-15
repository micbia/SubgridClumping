def ModelsParameters(xHII=[0.7, 0.5, 0.3], use_col=3, path='./'):
	from pandas import DataFrame
	from clump_functions import FindNearest
	import numpy as np, os
	
	xHII = np.array(xHII)
	print('For mean ionized fraction (volume):\txHII =', xHII, '\n')
	
	models = DataFrame([['BHC', 'r', '-', None, None, None, None, None], ['IC', 'b', '--', None, None, None, None, None], ['SC', 'g', '-', None, None, None, None, None]], index=['mean', 'quadfit', 'scat'] , columns=['name', 'color', 'style', 'lw', 'redshift_to_xHII', 'redshift_dT_to_xHII', 'xHII', 'xHI'])
	
	for fit in models.index:
		data_path = path+fit+'/'
		# for lw
		models.loc[fit, 'lw'] = 3.5 if fit=='quadfit' else 2

		# for xfrac
		data = np.loadtxt(data_path+'PhotonCounts2_%s.txt' %fit , usecols=(0, use_col), dtype={'names': ('z', 'xHII_v'), 'formats': (np.float, np.float)})	# usecols 3 for x_v, 4 for x_m
		idx, values = FindNearest(data['xHII_v'], xHII)
		models.loc[fit, 'redshift_to_xHII'] = data['z'][idx]
		models.loc[fit, 'xHII'] = values
		models.loc[fit, 'xHI'] = [round(1.-val, 3) for val in values]

		# for dT_b
		redshift_dT = np.loadtxt(data_path+'21cm/dTb_data/redshift.txt')
		idx, redshift_Pk = FindNearest(arr=redshift_dT, val=models.loc[fit, 'redshift_to_xHII'])
		models.loc[fit, 'redshift_dT_to_xHII'] = redshift_Pk

	return models

#models.to_csv('models.csv')
