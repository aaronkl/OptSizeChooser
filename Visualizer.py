'''
Created on 10.11.2013

@author: Aaron Klein
'''

import matplotlib.pyplot as plt
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from acquisition_functions.entropy_search import EntropySearch, NUMBER_OF_PMIN_SAMPLES, NUMBER_OF_CANDIDATE_SAMPLES, NUMBER_OF_REPRESENTER_POINTS
from acquisition_functions.expected_improvement import ExpectedImprovement
from acquisition_functions.entropy_search_big_data import EntropySearchBigData


class Visualizer():

    def __init__(self, index):

        self._index = index
        self._costs = index

    def plot(self, X, y, model, cost_model, cands):

        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        plt.hold(True)

        self.plot_gp(X, y, model)
        num_of_cands = 101
        entropy_estimator = EntropySearch(X, y, model, cost_model)
        entropy_plot = np.zeros([num_of_cands])
        points_plot = np.linspace(0, 1, num_of_cands)

        for i in xrange(0, num_of_cands):
            cand = np.array([points_plot[i]])
            entropy_plot[i] = entropy_estimator.compute(cand)

        self.plot_entropy_one_dim(points_plot, entropy_plot)

        ei = ExpectedImprovement(X, y, model)
        ei_values = np.zeros([num_of_cands])
        ei_points = np.linspace(0, 1, num_of_cands)

        for i in xrange(0, num_of_cands):
            ei_values[i] = ei.compute(np.array([ei_points[i]]))

        self.plot_expected_improvement(ei_points, ei_values)

        self.plot_points(entropy_estimator._func_sample_locations)
        #self.plot_points(cands)

        sample_locs = entropy_estimator._func_sample_locations

        pmin = entropy_estimator._compute_pmin_bins(model)

        self.plot_pmin(pmin, sample_locs)

        ax.axis([0, 1, -2, 2])
        #plt.legend((self._p_ei[0], self._p_ent[0], self._p_gp[0],
        #            self._p_pmin[0], self._p_rp[0], self._p_comp[0]),
        #           ('Expected Improvement', 'Entropy', 'Performance',
        #               'Pmin', 'Selected Candidates', 'Comp'), loc=0)

        filename = "/home/kleinaa/plots/plot_" + str(self._index) + ".png"
        self._index += 1

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

        ax.legend((self._p_ei[0], self._p_ent[0], self._p_gp[0],
                  self._p_pmin[0], self._p_rp[0], self._p_comp[0]),
                    ('Expected Improvement', 'Entropy', 'Performance',
                        'Pmin', 'Selected Candidates', 'Comp'),
                        loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        plt.savefig(filename)

    def plot_expected_improvement(self, cands, ei_values):

        self._p_ei = plt.plot(cands, ei_values, 'g')

    def plot_points(self, points):

        self._p_rp = plt.plot(points, np.ones(points.shape[0]) * (-2), 'ro')

    def plot_entropy_one_dim(self, cands, entropy_values):
        entropy_values = (entropy_values - np.mean(entropy_values)) / np.sqrt(np.var(entropy_values))

        self._p_ent = plt.plot(cands, entropy_values, 'b')

    def plot_pmin(self, pmin, sample_locs):

        self._p_pmin = plt.bar(sample_locs, pmin, bottom=-2,
                               width=0.04, color='b')

    def plot_gp(self, X, y, model):

        plt.hold(True)
        self._p_comp = plt.plot(X, y, 'k+')
        x = np.linspace(0, 1, 100)[:, np.newaxis]

        test_inputs = np.ones((100, 1))
        for i in range(0, 100):
            test_inputs[i] = x[i]

        (mean, variance) = model.predict(test_inputs, True)
        lower_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            lower_bound[i] = mean[i] - math.sqrt(variance[i])

        upper_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            upper_bound[i] = mean[i] + math.sqrt(variance[i])

        self._p_gp = plt.plot(x, mean, 'r')
        plt.fill_between(test_inputs[:, 0], upper_bound[:, 0],
                         lower_bound[:, 0], facecolor='red', alpha=0.6)

    def plot3D(self, X, y, model, cost_model, incumbent, cand):

        self._fig = plt.figure()
        fig = plt.figure(figsize=plt.figaspect(0.5))

        self._ax = fig.add_subplot(2, 2, 1, projection='3d')
        self._ax.text2D(0.0, 0.1, "Model")

        self._ax_2 = fig.add_subplot(2, 2, 2, projection='3d')
        self._ax_2.text2D(0.0, 0.1, "Entropy")

        self._ax_3 = fig.add_subplot(2, 2, 3)
#         self._ax_3.text2D(0.05, 1.95, "Candidates")

        self._ax_4 = fig.add_subplot(2, 2, 4)
#         self._ax_4.text2D(0.15, 1.95, "Pmin")

        self._ax_3.axis([0, 1, 0, 1])
        self._ax_4.axis([0, 1, 0, 1])

        plt.hold(True)
        self._ax_3.set_xlabel('S')
        self._ax_3.set_ylabel('X')

        self.plot3D_gp(model)

        entropy_estimator = EntropySearchBigData(X, y, model, cost_model)
        x = np.linspace(0, 1, 100)[:, np.newaxis]
        y = np.linspace(0, 1, 100)[:, np.newaxis]

        x, y = np.meshgrid(x, y)

        test_inputs = np.zeros((100 * 100, 2))
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                test_inputs[i * 100 + j, 0] = x[j][i]
                test_inputs[i * 100 + j, 1] = y[j][i]

        cost = cost_model.predict(test_inputs, False)

        entropy = np.zeros([100, 100])

        for i in xrange(0, 100):
            for j in xrange(0, 100):
                entropy[j][i] = entropy_estimator.compute(np.array([x[i][j],
                                                                    y[i][j]]))

                entropy[j][i] /= cost[i * 100 + j]

        self.plot3D_entropy(x, y, entropy)

        ei = ExpectedImprovement(X, y, model)
        x = np.linspace(0, 1, 100)[:, np.newaxis]
        y = np.linspace(0, 1, 100)[:, np.newaxis]

        x, y = np.meshgrid(x, y)

        ei_values = np.zeros([100, 100])
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                ei_values[j][i] = ei.compute(np.array([x[i][j], y[i][j]]))

        self._ax_3.hold(True)
        self._ax_3.plot(X[:, 0], X[:, 1], 'ro', label="Points")
        self._ax_3.plot(cand[:, 0], cand[:, 1], 'b+',
                                  label="Candidates")
        self._ax_3.contour(x, y, ei_values, colors=('green'),
                                  label="Expected Improvement")

        self._ax_3.legend()

        mu = model.predict(np.array([incumbent]))
        y = np.random.normal(loc=mu, scale=1.0)
        updated_model = model.copy()
        updated_model.update(incumbent, y)

        pmin_new = entropy_estimator._compute_pmin_bins(updated_model)
        #self.plot3D_expected_improvement(x, y, ei_values)

        #self.plot3D_representer_points(entropy_estimator._func_sample_locations)
        self._ax_4.hold(True)
        self.plot_old_pmin(entropy_estimator._pmin_old)

        self.plot_new_pmin(pmin_new)

        self._ax_4.legend()
        filename = "/home/kleinaa/plots/plot3D_" + str(self._index) + ".png"
        self._index += 1

        plt.savefig(filename)

    def plot3D_expected_improvement(self, x, y, ei_values):

        self._ax.plot_surface(x, y, ei_values, cmap='Greens_r')

    def plot3D_entropy(self, x, y, entropy):
        entropy = (entropy - np.mean(entropy)) / np.sqrt(np.var(entropy))
        self._ax_2.plot_surface(x, y, entropy, cmap='Blues_r')

    def plot3D_gp(self, model):

        x = np.linspace(0, 1, 100)[:, np.newaxis]
        y = np.linspace(0, 1, 100)[:, np.newaxis]

        x, y = np.meshgrid(x, y)
        test_inputs = np.zeros((100 * 100, 2))
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                test_inputs[i * 100 + j, 0] = x[j][i]
                test_inputs[i * 100 + j, 1] = y[j][i]

        mean = model.predict(test_inputs, False)

        z = np.zeros((100, 100))
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                z[j][i] = mean[i * 100 + j]

        self._ax.plot_surface(x, y, z, cmap='Reds_r')

    def plot_new_pmin(self, pmin):

        ind = np.linspace(0, 1, pmin.shape[0])
        #self._ax_4.bar(ind, pmin, 0.04, color='b')
        self._ax_4.plot(ind, pmin, color='b', label="Pmin_new")

    def plot_old_pmin(self, pmin):

        ind = np.linspace(0, 1, pmin.shape[0])
        #self._ax_4.bar(ind, pmin, 0.04, color='r')
        self._ax_4.plot(ind, pmin, color='r', label="Pmin_old")
