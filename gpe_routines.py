# GPE routines module
from __future__ import print_function, division

from module_load import *


# Create Latin Hypercube Sample of X and y data
def lhs_sample(selected_data):
	
	lhs_samp = np.loadtxt('lhs_samp.dat')
	X = selected_data[:,0]
	Y = selected_data[:,1]
	
	data_samp = np.array([]).reshape(0,3)
	
	# Scale lhs_samp to data slice
	lhs_samp[:,0] = lhs_samp[:,0] * (max(X)-min(X)) + min(X)
	lhs_samp[:,1] = lhs_samp[:,1] * (max(Y)-min(Y)) + min(Y)
	
	
	# Use LHS sample to choose data from slice
	for samp in range(0,np.size(lhs_samp,0)):
	    nearest = find_nearest(selected_data[:,0:2],lhs_samp[samp,:])
	    data_samp = np.vstack((data_samp,selected_data[nearest,:]))
	
	return data_samp
    

# Function to find element in 'array' nearest to 'value'
def find_nearest(array,value):
	    
	# Euclidean distance
	return (np.sqrt(np.sum(np.square(array-value),axis=1))).argmin()


# Function for linear regression
def lin_reg(x, a, b, c, d):
    
    #return a*np.sqrt(x[:]) + b*x + c
    #return a*np.sqrt(x[:]) + b*x**2 + c*x + d
    return a*x**(1./3) + b*x**2 + c*x + d


def fit_trend(in_data):
	
	data = np.copy(in_data)
	
	# Least squares regression
	popt, pcov = curve_fit(lin_reg, data[:,1], data[:,2])
	
	# Remove square root trend
	#data[:,2] = data[:,2] - lin_reg(data[:,1],popt[0],popt[1],popt[2])
	data[:,2] = data[:,2] - lin_reg(data[:,1],popt[0],popt[1],popt[2],popt[3])
	
	return data,popt


# Rescale parameters onto unit cube
def unit_cube(data):
    
    params = (min(data[:,0]),max(data[:,0]),min(data[:,1]),max(data[:,1]))
    
    cube_data = np.copy(data)
    cube_data[:,0] = cube_data[:,0] - min(cube_data[:,0])
    cube_data[:,0] = cube_data[:,0] / max(cube_data[:,0])
    
    cube_data[:,1] = cube_data[:,1] - min(cube_data[:,1])
    cube_data[:,1] = cube_data[:,1] / max(cube_data[:,1])
    
    return cube_data, params


def normalise(z_data):
    
    scaler = preproc.StandardScaler(with_mean=True).fit(z_data[:, np.newaxis])
    
    norm_z_data = z_data[:, np.newaxis]
    norm_z_data = scaler.transform(norm_z_data)
    norm_z_data = norm_z_data[:,0]
    
    return norm_z_data, scaler


# Create (fitted) 'gp' object (the gaussian process), used for predicting, calculating LML etc.
def emulate(data,kernel,restarts,no_fit=False):
    
    if no_fit:
        gp = sklgp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=restarts,optimizer=None).fit(data[:,0:2],data[:,2],alpha=0.001)
    else:
        gp = sklgp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=restarts).fit(data[:,0:2],data[:,2])
        
    return gp


def LML_plot(gp,iteration,params='13',bounds=([-3.0,0.0],[-7.0,-4.0]),meshes=(50,50),num_lines=10):
    # By default plot LML landscape for first length scale and noise level
    
    plt.figure(100)
    # Range of hyperparameters to plot
    par1 = np.logspace(bounds[0][0],bounds[0][1],meshes[0])
    par2 = np.logspace(bounds[1][0],bounds[1][1],meshes[1])
    par1_grid, par2_grid = np.meshgrid(par1,par2)
    
    # Calculate LML for range of hyperparmeters par1 and par2
    LML = np.zeros(meshes)
    for i in range(par1.shape[0]):
        for j in range(par2.shape[0]):
            if params == '12':
                #hyperparams = np.concatenate(([gp.kernel_.theta[0]], [np.log(par1[i])], [np.log(par2[j])], [gp.kernel_.theta[3]]))
                hyperparams = np.concatenate(([gp.kernel_.theta[0]], [np.log(par1[i])], [np.log(par2[j])])) # No noise
            elif params == '13':
                hyperparams = np.concatenate(([gp.kernel_.theta[0]], [np.log(par1[i])], [gp.kernel_.theta[2]], [np.log(par2[j])]))
            elif params == '23':
                hyperparams = np.concatenate(([gp.kernel_.theta[0]], [gp.kernel_.theta[1]], [np.log(par1[i])], [np.log(par2[j])]))
            
            LML[i,j] = gp.log_marginal_likelihood(hyperparams)
    
    LML = np.array(LML).T

    # Levels
    vmin, vmax = (-LML).min(), (-LML).max()
    level_min = np.floor(np.log10(vmin))
    level_max = np.ceil(np.log10(vmax))
    level = np.logspace(level_min, level_max, (level_max-level_min)*num_lines + 1)
    # Black contour lines
    plt.contour(par1_grid, par2_grid, -LML, levels=level, norm=LogNorm(vmin=10**level_min, vmax=10**level_max), colors='k', linewidths=0.2)
    # Filled contour plot
    plt.contourf(par1_grid, par2_grid, -LML, levels=level, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.jet)

    plt.xscale("log")
    plt.yscale("log")
    if params == '12':
        plt.xlabel("Length-scale 1")
        plt.ylabel("Length-scale 2")
    elif params == '13':
        plt.xlabel("Length-scale 1")
        plt.ylabel("Noise level")
    elif params == '23':
        plt.xlabel("Length-scale 2")
        plt.ylabel("Noise level")
    plt.title("Log marginal likelihood landscape")

    # Color bar formatting
    formatter = LogFormatter(10, labelOnlyBase=True)
    plt.colorbar(ticks=level, format=formatter)

    plt.tight_layout()
    
    plt.savefig('LML'+str(iteration)+'.pdf', format='pdf', dpi=1000)
    
    # Write to file LML landscape
    np.savetxt('LML'+str(iteration)+'.dat',LML)    
    
    plt.close(100)