# Module for routines for neutron star data
from __future__ import print_function, division

from module_load import *


# Read in neutron star data
def read_data_neutron(filename):
    all_data = np.loadtxt(filename)
    num_points = np.size(all_data,0)
    
    return all_data,num_points


# Interpolate surface, plot
def interp_surface(content,num_x,num_y):
	# Separate x,y,z data
	x = content[:,0]
	y = content[:,1]
	z = content[:,2]
	
	# Interpolate
	grid_x, grid_y = np.mgrid[min(x):max(x):complex(num_x), min(y):max(y):complex(num_y)]
	interpolated = interp.griddata(np.vstack((x,y)).T,z,(grid_x,grid_y),method='linear')
	
	x_pred = np.linspace(min(x),max(x),num_x)
	y_pred = np.linspace(min(y),max(y),num_y)
	
	
	# Remove densities with NaNs
	while np.isnan(interpolated[:,0]).any():
	    num_y = num_y-1
	    interpolated = np.delete(interpolated,0,axis=1)
	    y_pred = np.delete(y_pred,0)
	
	while np.isnan(interpolated[:,-1]).any():
	    num_y = num_y-1
	    interpolated = np.delete(interpolated,-1,axis=1)
	    y_pred = np.delete(y_pred,-1)
	
	# Put 'interpolated' data in table layout
	interpolated = interpolated.reshape(num_x*num_y,1)
	interp_data = np.array([]).reshape(0,3)
	iter = 0
	for i in x_pred:
		for j in y_pred:
			interp_data = np.vstack((interp_data, [i,j,interpolated[iter]]))
			iter = iter + 1
	
	# Save to file
	filename = 'interp_'+str(num_x)+'x'+str(num_y)+'.dat'
	np.savetxt(filename,interp_data)
	
	return interp_data, (num_x,num_y)


# Function to remove unwanted values from data
def subset(in_data,dim_1_bounds,dim_2_bounds):
	
	data = np.copy(in_data)
	
	# Remove values outisde area bounded by dim_1_bounds
	data = data[(data[:,0] > dim_1_bounds[0]) & (data[:,0] < dim_1_bounds[1])]
	
	# Remove data points outside area bounded by dim_2_bounds
	data = data[(data[:,1] > dim_2_bounds[0]) & (data[:,1] < dim_2_bounds[1])]
	
	size1 = np.size(data,0)
	size2 = np.size(np.unique(data[:,0]))
	data_dims = (size2, int(round(size1/size2)))
    
	return data, data_dims


def neutron_plot(x_data,y_data,z_data,title_string,file_string,limits,cbar_label,plot_points=False,points=1):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    contour_levs = np.linspace( limits[0], limits[1], np.around((limits[1]-limits[0])/limits[2],decimals=1)+1 )
    plt.contour(x_data,y_data,z_data,colors='k',linewidths=0.2,levels=contour_levs)
    plt.contourf(x_data,y_data,z_data,cmap=plt.cm.jet,levels=contour_levs)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(cbar_label)
    if plot_points:
        plt.plot(points[:,0],points[:,1],'bx',markersize=3)
    plt.title(title_string)
    plt.xlabel('Z')
    plt.ylabel(r'$\rho$ [fm$^{-3}$]')
    plt.tight_layout()
    plt.savefig(file_string,format='pdf',dpi=1000)
    plt.close(1)


def extract_mins(data,error):
    mins = np.zeros(data.shape[0],dtype='int')
    points_within_error = np.zeros((0,2))
    for i in range(0,data.shape[0]):
        mins[i] = np.argmin(data[i,:])
        minZ = np.min(data[i,:])
        for j in range(0,data.shape[1]):
            if (abs(minZ - data[i,j]) <= error):
                points_within_error = np.append(points_within_error, [[i,j]], axis=0)
    
    
    return mins.astype(int), points_within_error.astype(int)


def plot_mins(mins,within_error,iteration):
    fig = plt.figure(10)
    ax = fig.add_subplot(111)
    plt.plot(within_error[:,0],within_error[:,1],'bx',markersize=2)
    plt.plot(mins[:,0],mins[:,1],'r')
    plt.title('Z values with minimum energy')
    plt.xlabel(r'$\rho$ [fm$^{-3}$]')
    plt.ylabel('Z')
    ax.set_ylim([20,60])
    plt.tight_layout()
    plt.savefig('min_energies'+str(iteration)+'.pdf',format='pdf',dpi=1000)
    plt.close(10)
    
    
def plot_training_set(x_data,y_data,z_data,limits,iteration,points,plot_new=False,new_points=1):
    fig = plt.figure(20)
    ax = fig.add_subplot(111)
    contour_levs = np.linspace( limits[0], limits[1], np.around((limits[1]-limits[0])/limits[2],decimals=1)+1 )
    plt.contourf(x_data,y_data,z_data,cmap=plt.cm.jet,levels=contour_levs)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('MeV/A')
    plt.plot(points[:,0],points[:,1],'ro',markersize=1)
    if plot_new:
        plt.plot(new_points[:,0],new_points[:,1],'kx',markersize=2)
    plt.title('Training set for iteration ' + str(iteration))
    plt.xlabel('Z')
    plt.ylabel(r'$\rho$ [fm$^{-3}$]')
    plt.tight_layout()    
    plt.savefig('training_set'+str(iteration)+'.pdf',format='pdf',dpi=1000)
    plt.close(20)
