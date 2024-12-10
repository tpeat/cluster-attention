# for non-identity QK
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import interp1d
from datetime import datetime
from tqdm import trange

# TODO: change to experiment name
experiment = "baseline"

n = 64
T = 15
dt = 0.1
num_steps = int(T/dt) + 1
d = 3
beta = 1
denominator = True
half_sphere = False

V = np.eye(d)
A = np.eye(d)

Q = np.eye(d)
K = np.eye(d)

# TODO: uncomment to randomize Q,K
# Q = np.random.randn(d, d)
# K = np.random.randn(d, d)

project_residual = True
residual_strength = 0.0 # TODO: set to > 0 to add skip connections

x0 = np.random.randn(n, d)
x0 /= np.linalg.norm(x0, axis=1)[:, np.newaxis]

z0 = np.random.randn(n, d)
z0 /= np.linalg.norm(z0, axis=1)[:, np.newaxis]

if half_sphere:
    x0[x0[:, 2] < 0] *= -1

z = np.zeros(shape=(n, num_steps, d)) 
z[:, 0, :] = x0
integration_time = np.linspace(0, T, num_steps)

def projection(x, v):
    """Project v onto the tangent space at x"""
    return v - (np.sum(x * v, axis=1) / np.sum(x * x, axis=1))[:, np.newaxis] * x

for l, t in enumerate(integration_time[:-1]):
    Qz = np.matmul(Q, z[:, l, :].T)  # Q(t)x_k(t)
    Kz = np.matmul(K, z[:, l, :].T)

    attention_scores = np.matmul(Qz.T, Kz)
    exp_beta_dot = np.exp(beta * attention_scores)
    
    causal_mask = np.tril(np.ones((n, n)))
    exp_beta_dot = exp_beta_dot * causal_mask
    
    if denominator:
        attention = exp_beta_dot / exp_beta_dot.sum(axis=1)[:, np.newaxis]
    else:
        attention = exp_beta_dot / n
    
    dlst = np.matmul(attention, np.matmul(V, z[:, l, :].T).T)

    if project_residual:
        combined = z[:, l, :] + residual_strength * dlst
        dynamics = projection(z[:, l, :], combined)
    else:
        dynamics = projection(z[:, l, :], dlst)
        dynamics = dynamics + residual_strength * z[:, l, :]
    
    dynamics = projection(z[:, l, :], dlst)
    
    z[:, l+1, :] = z[:, l, :] + dt * dynamics
    z[:, l+1, :] = z[:, l+1, :] / np.linalg.norm(z[:, l+1, :], axis=1)[:, np.newaxis]

movie = False
color = '#3658bf'
now = datetime.now() 
if d == 2:
    dir_path = f'./circle/{experiment}/beta={beta}'
else:
    dir_path = f'./sphere/{experiment}/beta={beta}'
dt_string = now.strftime("%H-%M-%S")
filename = dt_string + "movie.gif"
base_filename = dt_string
        
if not os.path.exists(dir_path):
     os.makedirs(dir_path)

# set viz bounds
x_min, x_max = z[:, :, 0].min(), z[:, :, 0].max()
if d>1:
    y_min, y_max = z[:, :, 1].min(), z[:, :, 1].max()
    if d == 3:
        z_min, z_max = z[:, :, 2].min(), z[:, :, 2].max()

margin = 0.1
x_range = x_max - x_min
x_min -= margin * x_range
x_max += margin * x_range

if d>1:
    y_range = y_max - y_min
    y_min -= margin * y_range
    y_max += margin * y_range
    if d == 3:
        z_range = z_max - z_min
        z_min -= margin * z_range
        z_max += margin * z_range
            
font = {'size'   : 18}
rc('font', **font)
        
interp_x = []
interp_y = []
interp_z = []

for i in range(n):
    interp_x.append(interp1d(integration_time, z[i, :, 0], 
                             kind='cubic', 
                             fill_value='extrapolate'))
    if d>1:
        interp_y.append(interp1d(integration_time, z[i, :, 1], 
                                 kind='cubic', 
                                 fill_value='extrapolate'))
        if d==3:
            interp_z.append(interp1d(integration_time, z[i, :, 2], 
                                     kind='cubic', 
                                     fill_value='extrapolate'))

# controls num perticles in frame
particles_per_frame = max(1, n // num_steps)
current_particles = particles_per_frame

for t in trange(num_steps):
    if d == 2:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1)
        label_size = 16
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.title(r'$t={t}$'.format(t=str(round(t*dt,2))), fontsize=16)
            
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        plt.rc('font', family='serif')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # plot visible particles (some still are hidden by mask)
        current_particles = min(n, current_particles + particles_per_frame)
        visible_range = range(current_particles)
        
        plt.scatter([x(integration_time)[t] for i, x in enumerate(interp_x) if i in visible_range], 
                    [y(integration_time)[t] for i, y in enumerate(interp_y) if i in visible_range], 
                    c=color, 
                    alpha=1, 
                    marker='o', 
                    linewidth=0.75, 
                    edgecolors='black', 
                    zorder=3)
        
        if t > 0:
            for i in visible_range:
                x_traj = interp_x[i](integration_time)[:t+1]
                y_traj = interp_y[i](integration_time)[:t+1]
                plt.plot(x_traj, 
                         y_traj, 
                         c=color, 
                         alpha=1, 
                         linewidth=0.25, 
                         linestyle='dashed',
                         zorder=1)
        
        plt.savefig(os.path.join(dir_path, base_filename + "{}.pdf".format(t)), 
                    format='pdf', 
                    bbox_inches='tight')

    elif d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        label_size = 16
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size

        phi = np.linspace(0, 2*np.pi, 100)
        
        plt.title(r'$t={t}$'.format(t=str(round(t*dt,2))), fontsize=16)
        
        current_particles = min(n, current_particles + particles_per_frame)
        visible_range = range(current_particles)
            
        ax.scatter([x(integration_time)[t] for i, x in enumerate(interp_x) if i in visible_range], 
                    [y(integration_time)[t] for i, y in enumerate(interp_y) if i in visible_range],
                    [z(integration_time)[t] for i, z in enumerate(interp_z) if i in visible_range],
                    c=color, 
                    alpha=1, 
                    marker='o', 
                    linewidth=0.75, 
                    edgecolors='black')
        
        plt.rc('font', family='serif')
        ax.axis('off')
        
        if t > 0:
            for i in visible_range:
                x_traj = interp_x[i](integration_time)[:t+1]
                y_traj = interp_y[i](integration_time)[:t+1]
                z_traj = interp_z[i](integration_time)[:t+1]
                ax.plot(x_traj, 
                        y_traj, 
                        z_traj, 
                        c=color, 
                        alpha=0.75, 
                        linestyle='dashed',
                        linewidth=0.25)
        
        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        
        default_azim = 30  
        rotation_range = 30
        total_frames = num_steps
        
        angle = default_azim + rotation_range * (t/total_frames - 0.5)
        ax.view_init(elev=10, azim=angle)
    
        ax.grid(False)
        plt.locator_params(nbins=4)
        
        plt.savefig(os.path.join(dir_path, base_filename + "{}.png".format(t)), 
                    format='png', 
                    bbox_inches='tight',
                    dpi=500)
     
    if movie:           
        plt.savefig(os.path.join(dir_path, base_filename + "{}.png".format(t)),
                    format='png', dpi=250, bbox_inches='tight')
    plt.clf()
    plt.close()