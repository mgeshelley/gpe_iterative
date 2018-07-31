from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')

from module_load import *
from gpe_routines import *
from neutron import *


# --------------------
# ---- PARAMETERS ----
# --------------------

# -- DATA --
# Select starting sample rate, or select starting number of points?
select_init_samp_rate = False
# Initial number of points
start_points = 200
# Initial sample rate
start_frac = 0.02
# Filename to read
f_name = 'data/sly4.dat'
# How to interpolate original data set
interp_Z = 21
interp_rho = 600 # > 600 --> can get memory errors when evaluating surface
# Which densities to look at
    # Spacing in original data is ~5e-6 up to 0.0031; above this, spacing is ~1e-4
rho_bounds = (0.0,0.023)
# Which Z values to look at
Z_bounds = (19,61)

# -- EMULATION --
# Maximum number of new points to add each iteration
max_new_points = 200
# Maximum sigma tolerated (CURRENTLY 2 SIGMA)
max_sigma = 0.005 # Make very large (>1) to prevent iteration
# Number of restarts for GP hyperparameter optimization
restarts = 5
# Kernel bounds
length_1_bounds = [1e-2,1e-0] # Z
length_2_bounds = [1e-3,1e-0] # rho
noise_bounds = [1e-7,1e-10]
fixed_noise = 2e-3 # Fixed value if not varying noise during optimization
# Specify kernel hyperparameters instead of fitting
specify_kernel = False

# -- PLOTTING --
# Error in HFB calculations
HFB_error = 0.004
# Extra plots? (standardized data, training set)
plot_extra = True

# Plot LML landscape?
plot_LML = False
# Which hyperparameters to explore in LML plot
LML_pars = '12'
# Mesh for LML plot
LML_meshes = (40,40)
# Number of lines per order of mag on LML plot
LML_line_density = 5

# Plotting limits
surf_plot_lims = (-2.0,6.0,0.5) # Plotting original data
diff_plot_lims = (0.0,0.01,0.001) # Plotting difference
conf_plot_lims = (0.0,0.01,0.001) # Plotting confidence intervals
stand_plot_lims = (-2.0,2.0,0.1) # Plotting standardized surface


# -----------------------
# ---- PROGRAM START ----
# -----------------------

# Get mode to run program
mode = sys.argv[1]

if (mode=='start_samp'):
    # Print to screen number of points to be used for initial LHS sample
    # (this value is read by 'create_lhs.R')
    if select_init_samp_rate:
        all_data, num_all_points = read_data_neutron(filename=f_name)
        num_start = int(round(num_all_points*start_frac))
        print(num_start)
    else:
        print(start_points)
    
elif (mode=='iterative'):
    # Main program mode - iterative procedure
    
    print ('Number of restarts for optimization = %4d' % restarts)
    print ('Maximum number of new points per iteration = %4d ' % max_new_points)
    print ('Maximum 2*sigma tolerated = %5.3f' % max_sigma)
    print ('Kernel bounds:')
    print ('{0:>25s} = ({1:.0e},{2:.0e})'.format('Z length-scale bounds',length_1_bounds[0],length_1_bounds[1]))
    print ('{0:>25s} = ({1:.0e},{2:.0e})'.format('rho length-scale bounds',length_2_bounds[0],length_2_bounds[1]))
    print ('{0:>25s} = ({1:.0e},{2:.0e})'.format('Noise bounds',noise_bounds[0],noise_bounds[1]))
    print ('Error in HFB calculations = {0:5.3f}'.format(HFB_error))

    # Read data from file
    print ('Reading from file "'+f_name+'"...')
    all_data, num_all_points = read_data_neutron(filename=f_name)
    # Select densities to study
    print ('Selecting densities in range ({0:8.6f},{1:8.6f})fm^-3...'.format(rho_bounds[0],rho_bounds[1]))
    subset_data, subset_dims = subset(all_data,Z_bounds,rho_bounds)
    num_all_points = subset_data.shape[0]

    if select_init_samp_rate:
        print ('Percentage of original data set = {0:.1%}'.format(start_frac))
    else:
        print ('Percentage of original data set = {0:.1%}'.format(start_points/num_all_points))
    print ('Size of original data set = %6d' % num_all_points)
    
    # Interpolate all_data onto grid of 21 Z values, and \rho values spaced roughly 0.00004 apart; remove densities containing NaNs after interpolation
    print ('Interpolating...')
    final_data, trimmed_dims = interp_surface(subset_data,interp_Z,interp_rho)
    print ('Interpolated onto {0:2d}x{1:4d} grid'.format(trimmed_dims[0],trimmed_dims[1]))
    num_interp = np.size(final_data,0)
    print ('Size of interpolated data set = %6d' % num_interp)

    # Arrays for grid for emulating and plotting
    stand_x_pred = np.linspace(0.0,1.0,trimmed_dims[0])
    stand_y_pred = np.linspace(0.0,1.0,trimmed_dims[1])
    
    # Create standardized prediction grid {[0:1],[0:1]}
    stand_subset_grid, stand_subset_params = unit_cube(final_data[:,0:2])
    
    # Create non-standardized prediction grid for plotting later
    x_pred = (stand_x_pred*(stand_subset_params[1]-stand_subset_params[0])) + stand_subset_params[0]
    y_pred = (stand_y_pred*(stand_subset_params[3]-stand_subset_params[2])) + stand_subset_params[2]
    
    
    
    ### CODE TO GET MINIMUMS FROM DATA BEFORE EMULATION ###
    
    #raw_data = np.copy(final_data[:,2])
    #raw_data = raw_data.reshape(trimmed_dims)
    #raw_data = raw_data.T
    
    #### Get minimum energy for each density in prediction grid
    #print ('Extracting minimum energy values for raw (interpolated) data...')
    #mins, within_HFB_error = extract_mins(raw_data,HFB_error)
    #print (mins)
    #print (within_HFB_error)
    ## Minima for each density
    #mins_data = np.vstack((y_pred,x_pred[mins])).T
    #np.savetxt('Z_minE_raw.dat',mins_data)
    ## All points within HFB error of min, at each density
    #points_within_HFB_error = np.vstack((y_pred[within_HFB_error[:,0]],x_pred[within_HFB_error[:,1]])).T
    #np.savetxt('Z_E_within_error_raw.dat',points_within_HFB_error)
    ## Plot
    #plot_mins(mins_data,points_within_HFB_error,'raw')
    
    
    
    # Check if there are NaNs in array 'final_data' (usually created at boundaries by interpolation routine)
    if (np.sum(np.isnan(final_data))!=0):
        print ('ERROR: NaNs in array "final_data"')
    
    # Take LHS sample using 'lhs_samp.dat'
    print ('Creating LHS sample...')
    samp_data = lhs_sample(final_data)
    
    ### Plot standardized original data
    if plot_extra:
        # Remove square-root trend from data
        whole_fit_data, whole_fit_params = fit_trend(np.copy(final_data))
        fit_subset_z = whole_fit_data[:,2] - lin_reg(whole_fit_data[:,1],whole_fit_params[0],whole_fit_params[1],whole_fit_params[2],whole_fit_params[3])
        # Normalise data
        whole_norm_z, whole_scaler = normalise(fit_subset_z)        
        norm_subset_z = whole_scaler.transform(fit_subset_z[:, np.newaxis])
        # Plot
        norm_subset_z = norm_subset_z.reshape(trimmed_dims)
        norm_subset_z = norm_subset_z.T
        limits = stand_plot_lims
        neutron_plot(stand_x_pred,stand_y_pred,norm_subset_z,'Standardized original data','stand_data.pdf',limits,'Energy per particle')
    
    # Original data ready for plotting
    z_orig = np.copy(final_data[:,2])
    z_orig = z_orig.reshape(trimmed_dims)
    z_orig = z_orig.T
    
    
    ### Kernel used for GP regression
    ## Start optimization from small length-scale, low-noise hyperparameters
    #kernel = kerns.ConstantKernel(1.0, (1.0,1.0)) * kerns.RBF([length_1_bounds[0],length_2_bounds[0]], [length_1_bounds,length_2_bounds]) + kerns.WhiteKernel(noise_bounds[0], noise_bounds)
    
    ## Start optimization from harmonic mean of bounds
    #kernel = kerns.ConstantKernel(1.0, (1.0,1.0)) * kerns.RBF([stats.hmean(length_1_bounds),stats.hmean(length_2_bounds)], [length_1_bounds,length_2_bounds]) + kerns.WhiteKernel(noise_bounds[0], noise_bounds)
    
    # Kernel with fixed noise component
    kernel = kerns.ConstantKernel(1.0, (1.0,1.0)) * kerns.RBF([stats.hmean(length_1_bounds),stats.hmean(length_2_bounds)], [length_1_bounds,length_2_bounds]) + kerns.WhiteKernel(fixed_noise, "fixed")
    
    ## Kernel with no noise component
    #kernel = kerns.ConstantKernel(1.0, (1.0,1.0)) * kerns.RBF([stats.hmean(length_1_bounds),stats.hmean(length_2_bounds)], [length_1_bounds,length_2_bounds])
    
    print ('\nStart iterating...')
    variance_too_high = True
    iteration = 1
    #while iteration <= 1:
    while variance_too_high:
        print ('\n\nIteration %2d' % iteration)
        
        training_set_size = np.size(samp_data,0)
        print ('Size of training set = %5d' % training_set_size)
        print ('Percentage of original data = {0:.1%}'.format(training_set_size/num_all_points))
        
        # Remove square-root trend from data
        print ('Removing trend from data...')
        fit_data, fit_params = fit_trend(samp_data)
        #print (fit_params)
        
        # Normalise data
        print ('Normalising data...')
        norm_z, scaler = normalise(fit_data[:,2])
        norm_data = np.hstack((fit_data[:,0:2],norm_z[:, np.newaxis]))
        #print (scaler.scale_,scaler.mean_)
        
        # Rescale Z and rho to unit cube
        print ('Rescaling Z, rho...')
        #stand_norm_data, stand_params = unit_cube(norm_data)
        stand_norm_data = norm_data
        stand_norm_data[:,0] = stand_norm_data[:,0] - stand_subset_params[0]
        stand_norm_data[:,0] = stand_norm_data[:,0] / (stand_subset_params[1] - stand_subset_params[0])
        stand_norm_data[:,1] = stand_norm_data[:,1] - stand_subset_params[2]
        stand_norm_data[:,1] = stand_norm_data[:,1] / (stand_subset_params[3] - stand_subset_params[2])
        
        if specify_kernel:
            #kernel.theta = np.log([1.0,0.0697,0.0201,1e-10])
            kernel.theta = np.log([1.0,0.0697,0.0201])
            #gp = sklgp.GaussianProcessRegressor(kernel=kernel)
            gp = emulate(stand_norm_data,kernel,restarts,no_fit=True)
        else:
            print('Fitting kernel parameter(s) using data...')
            gp = emulate(stand_norm_data,kernel,restarts)
        
        print (gp.kernel_)
        
        #if specify_kernel:
            #LML = 1.0
        #else:
        LML = gp.log_marginal_likelihood()
        print ('Log marginal likelihood for hyperparameters = %6.2f' % LML)
        if (LML > 0.0):
            print ('WARNING: LML > 0')
        
        # Mean, two_sig
        print ('Performing regression over prediction grid...')
        ''' CHANGE TO LOOP TO AVOID MEMORY ERROR????? '''
        raw_gp_pred, raw_gp_var = gp.predict(stand_subset_grid, return_cov=True)
        
        # Plots
        print ('Plotting emulation results...')
        
        
        #### Plot emulation of standardized data
        #stand_gp_pred = np.copy(raw_gp_pred)
        #stand_gp_pred = stand_gp_pred.reshape(trimmed_dims)
        #stand_gp_pred = stand_gp_pred.T
        #limits = (-2.0,2.0,0.1)
        #neutron_plot(stand_x_pred,stand_y_pred,stand_gp_pred,'Standardized emulation','stand_emul'+str(iteration)+'.pdf',limits)
        
        
        ### Plot emulation of original surface        
        # De-normalise
        gp_pred = scaler.inverse_transform(raw_gp_pred)
        # Re-add trend
        #gp_pred = gp_pred + lin_reg(final_data[:,1],fit_params[0],fit_params[1],fit_params[2])
        gp_pred = gp_pred + lin_reg(final_data[:,1],fit_params[0],fit_params[1],fit_params[2],fit_params[3])
        
        gp_pred = gp_pred.reshape(trimmed_dims)
        gp_pred = gp_pred.T

        if plot_extra:
            # Plot
            limits = surf_plot_lims        
            neutron_plot(x_pred,y_pred,gp_pred,'Original surface emulation','emul'+str(iteration)+'.pdf',limits,'Energy per particle [MeV/A]')
        
        
        # Write to file prediction grids
        np.savetxt('x_pred.dat',x_pred)
        np.savetxt('y_pred'+str(iteration)+'.dat',y_pred)
        
        # Write to file GPE prediction
        np.savetxt('emul'+str(iteration)+'.dat',gp_pred)
        
        # Plot difference between original surface and emulation
        gp_diff = abs(gp_pred-z_orig)
        limits = diff_plot_lims
        neutron_plot(x_pred,y_pred,gp_diff,'Original surface difference','diff'+str(iteration)+'.pdf',limits,'Energy per particle [MeV/A]')
        
        # Write to file difference
        np.savetxt('diff'+str(iteration)+'.dat',gp_diff)       
        
        # Make variance array
        gp_var = np.copy(raw_gp_var)
        gp_var = np.diag(gp_var)
        gp_var = gp_var.reshape(trimmed_dims)
        
        # Find local maxima in variance array
        print ('Adding new points to training set...')
        maxima = np.empty([0,4])
        for i,j in np.ndindex(gp_var.shape):
            value = gp_var[i,j]
            neighbours = []
            if i!=0:
                neighbours.append(gp_var[i-1,j])
            if j!=0:
                neighbours.append(gp_var[i,j-1])
            if i!=(gp_var.shape[0]-1):
                neighbours.append(gp_var[i+1,j])
            if j!=(gp_var.shape[1]-1):
                neighbours.append(gp_var[i,j+1])
            if value > max(neighbours):
                new_point = find_nearest(stand_subset_grid,[stand_x_pred[i],stand_y_pred[j]])
                # Add new point to list of maxima
                maxima = np.append(maxima, [np.append(final_data[new_point,:],value)], axis=0)
                #maxima[-1,3] = value
        
        # If too many maxima are found, only keep those with the highest variance value
        if maxima.shape[0] > max_new_points:
            maxima = maxima[maxima[:,3].argsort()][::-1]
            #maxima = np.argsort(-maxima)[:3]
            maxima = maxima[0:max_new_points,:]
        
        # Add new points to training set
        samp_data = np.append(samp_data, maxima[:,0:3], axis=0)
        
        # Plot confidence intervals (2*sigma)
        gp_var = gp_var * (scaler.scale_)**2 # Rescale (variance scales quadratically with data)
        gp_2_sig = 2*np.sqrt(gp_var) # Calculate 2*sigma
        gp_2_sig = gp_2_sig.T
        limits = conf_plot_lims
        neutron_plot(x_pred,y_pred,gp_2_sig,r'$2\sigma$ confidence interval','conf'+str(iteration)+'.pdf',limits,'Energy per particle [MeV/A]',plot_points=False,points=samp_data[training_set_size:-1,0:2])
        
        # Write to file confidence intervals (2*sigma)
        np.savetxt('conf'+str(iteration)+'.dat',gp_2_sig)        
        
        if plot_extra:
            ### Plot training set (+ new points in next iteration)
            limits = conf_plot_lims
            plot_training_set(x_pred,y_pred,gp_2_sig,limits,iteration,samp_data[0:training_set_size,0:2],plot_new=True,new_points=samp_data[training_set_size:-1,0:2])
        
        # Write to file training set
        np.savetxt('training_set'+str(iteration)+'.dat',samp_data[0:training_set_size,0:2])
        
        
        ### Get minimum energy for each density in prediction grid
        print ('Extracting minimum energy values...')
        mins, within_HFB_error = extract_mins(gp_pred,HFB_error)
        # Minima for each density
        mins_data = np.vstack((y_pred,x_pred[mins])).T
        np.savetxt('Z_minE_'+str(iteration)+'.dat',mins_data)
        # All points within HFB error of min, at each density
        points_within_HFB_error = np.vstack((y_pred[within_HFB_error[:,0]],x_pred[within_HFB_error[:,1]])).T
        np.savetxt('Z_E_within_error_'+str(iteration)+'.dat',points_within_HFB_error)
        # Plot
        plot_mins(mins_data,points_within_HFB_error,iteration)
        
        
        # Mean value of (2 SIGMA) sigma over surface
        print ('Mean sigma = {0:6.4f}'.format(np.mean(gp_2_sig)))
        # Check if sigma (CURRENTLY 2 SIGMA) low enough
        print ('Maximum sigma = %6.4f' % np.max(gp_2_sig))
        if np.max(gp_2_sig) <= max_sigma:
            variance_too_high = False
        
        
        ### Plot LML landscape
        if plot_LML:
            if LML < 0:
                print ('Plotting log marginal likelihood landscape...')
                if LML_pars == '12':
                    LML_bounds = (np.log10(length_1_bounds),np.log10(length_2_bounds))
                elif LML_pars == '13':
                    LML_bounds = (np.log10(length_1_bounds),np.log10(noise_bounds))
                elif LML_pars == '23':
                    LML_bounds = (np.log10(length_2_bounds),np.log10(noise_bounds))
                LML_plot(gp,iteration,params=LML_pars,bounds=LML_bounds,meshes=LML_meshes,num_lines=LML_line_density)
            else:
                print ('Log marginal likelihood >0 --> no plot')
        
        iteration = iteration + 1
    
    print ('\n\nMaximum sigma < {0:6.4f}: iteration stopped\n'.format(max_sigma))
        
else:
    print('ERROR: incorrect mode specified')
