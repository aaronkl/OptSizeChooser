'''
Created on Apr 17, 2014

@author: Aaron Klein
'''
import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from mpl_toolkits.mplot3d import Axes3D


from gp_model import GPModel, fetchKernel, getNumberOfParameters
from support import compute_expected_improvement

def getint(name):
    basename = name.partition('.')
    alpha, num = basename[0].split('_')
    return int(num)

points_per_axis = 50
num_of_samples = 500
index = 0
path = "/home/kleinaa/experiments/candidates/exp5/run7/"

for root, dirs, files in os.walk(path):

    files.sort(key=getint)

    for f in files:

        if f.endswith(".pkl"):

            #fig = plt.figure(figsize=(20, 8))
            num_plots_x = 1
            num_plots_y = 3
            axis_index = 1

            fig, (ax1, ax2, ax3) = plt.subplots(num_plots_x, num_plots_y)
            fig.set_size_inches(20, 8)
            plt.hold(True)

            ax1.axis([0, 1, 0, 1])
            ax2.axis([0, 1, 0, 1])
            ax3.axis([0, 1, 0, 1])

            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            ax3.set_aspect('equal')

#             ax1.set_xlabel('S')
#             ax1.set_ylabel('X')
#             ax2.set_xlabel('S')
#             ax2.set_ylabel('X')
#             ax3.set_xlabel('S')
#             ax3.set_ylabel('X')

            ax1.text(0.2, 1.05, "Gaussian Process")
            ax2.text(0.28, 1.05, "Expected Improvement")
            ax3.text(0.46, 1.05, "Pmin")

            pkl_file = open(root + str("/") + f, 'rb')
            (comp, vals, hyper_samples, incumbent, selected_candidates, rep) = pickle.load(pkl_file)

            covarname = "Matern52"
            mean = hyper_samples[0][0]
            noise = hyper_samples[0][1]
            amp2 = hyper_samples[0][2]
            cov_func, _ = fetchKernel(covarname)
            ls = hyper_samples[0][3]

            print "hyperparameter samples of run " + str(index)
            for h in hyper_samples:
                print "mean: " + str(h[0]) + " noise: " + str(h[1]) + " amp2: " + str(h[2]) + " ls: " + str(h[3])

            model = GPModel(comp, vals, mean, noise, amp2, ls, covarname)

            x_idx = np.linspace(0, 1, points_per_axis)[:, np.newaxis]
            y_idx = np.linspace(0, 1, points_per_axis)[:, np.newaxis]

            x, y = np.meshgrid(x_idx, y_idx)
            grid = np.zeros((points_per_axis * points_per_axis, 2))
            for i in xrange(0, points_per_axis):
                for j in xrange(0, points_per_axis):
                    grid[i * points_per_axis + j, 0] = x[j][i]
                    grid[i * points_per_axis + j, 1] = y[j][i]

            mean = model.predict(grid, False)

            z = np.zeros([points_per_axis, points_per_axis])
            ground_truth = np.zeros([points_per_axis, points_per_axis])
            for i in xrange(0, points_per_axis):
                for j in xrange(0, points_per_axis):
                    z[i][j] = mean[i * points_per_axis + j]
                    x_2 = y_idx[i] * 15
                    x_1 = (x_idx[j] * 15) - 5
                    ground_truth[i][j] = np.square(x_2 - (5.1/(4*np.square(math.pi)))*np.square(x_1) + (5/math.pi)*x_1 - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x_1) + 10

            ax1.pcolor(x, y, z, cmap='hot')

            ax2.hold(True)
            ax2.plot(comp[:, 0], comp[:, 1], 'ro', label="Points")
            ax2.plot(selected_candidates[:, 0], selected_candidates[:, 1], 'b+', label="Candidates")

            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

            ei_values = np.zeros([points_per_axis, points_per_axis])

            for i in xrange(0, points_per_axis):
                for j in xrange(0, points_per_axis):
                    ei_values[j][i] = compute_expected_improvement(np.array([x[i][j], y[i][j]]), model)

            ax2.pcolor(x, y, ei_values, cmap='Greens')

            ax3.hold(True)

            pmin = np.zeros(grid.shape[0])
            mean, L = model.getCholeskyForJointSample(grid)

            Omega = np.asfortranarray(np.random.normal(0, 1, (500,
                                                 grid.shape[0])))

            for omega in Omega:
                vals = model.drawJointSample(mean, L, omega)
                mins = np.where(vals == vals.min())[0]
                number_of_mins = len(mins)
                for m in mins:
                    pmin[m] += 1. / (number_of_mins)
                    pmin += pmin / 500

            hist = np.zeros([points_per_axis, points_per_axis])

            for i in xrange(0, points_per_axis):
                for j in xrange(0, points_per_axis):
                    hist[j][i] = pmin[i * points_per_axis + j]

            xopt = np.array([-np.pi, np.pi, 9.42478])
            yopt = np.array([12.275, 2.275, 2.475])

            xopt = (xopt + 5) / 15
            yopt = yopt / 15

            ax3.contour(x, y, hist, colors=('blue'), label="Pmin")

            ax3.plot(rep[:, 0], rep[:, 1], 'mo', label="Representers")
            ax3.plot(rep[-1, 0], rep[-1, 1], 'gs', label="Incumbent")
            ax3.pcolor(x, y, ground_truth, cmap=CM.gray)
            ax3.scatter(xopt, yopt, marker="o", facecolor='w', edgecolor='w', s=20 * 5)

            ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
            filename = path + "/plot3D_" + str(index) + ".png"
            index += 1
            
            print comp

            plt.savefig(filename)


print "Finished"
