'''
Created on 22.10.2013

@author: Aaron Klein, Simon Bartels

Chooser that uses adapts EI criterion for big data.

IMPORTANT: Assumes that the first variable is the data set size.
'''

import cPickle
import copy
import multiprocessing
import shutil

from Locker import *
from chooser.GPEIOptChooser import GPEIOptChooser
import gp
from helpers import *
import numpy          as np
import numpy.random   as npr
import scipy.linalg   as spla
import scipy.optimize as spo
import scipy.stats    as sps
import util
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d

#plt.ion()

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return GPEIOptSizeChooser(expt_dir, **args)

def optimize_pt(candidate, b, comp, vals, model):
    def wrap_grad_opt_ei_hyper(cand, comp, vals):
        cand = np.reshape(cand, (-1, comp.shape[1]))
        # cs = np.array([cand])
        (ei, grad_ei) = model.grad_optimize_ei_over_hypers(cand, comp, vals)
        sum_grad_ei = [grad_ei[0, d, :].sum() for d in xrange(0, comp.shape[1])]
        sum_grad_ei_np = np.array(sum_grad_ei)
        return (ei[0].sum(), sum_grad_ei_np)
    ret = spo.fmin_l_bfgs_b(wrap_grad_opt_ei_hyper,
                            candidate.flatten(), args=(comp, vals),
                            bounds=b, disp=0)
    return ret[0]

def _restrict_point_array(array):
    """
    Sets the first entry of each point to 1.
    """
    for n in xrange(0, array.shape[0] - 1):
            c = array[n]
            c[0] = 1
            array[n] = c
    return array


class GPEIOptSizeChooser:
            
    def compare(self, f, *args):
        randomstate = npr.get_state()
        res = self.f(args)
        if self.compare is None:
            print "compare object is null"
        else:
            self.compare.randomstate = randomstate
            res2 = self.compare.f(args)
            print res
            print res2
        return res

    # Given a set of completed 'experiments' in the unit hypercube with
    # corresponding objective 'values', pick from the next experiment to
    # run according to the acquisition function.
    def next(self, grid, values, durations,
             candidates, pending, complete):

        # Don't bother using fancy GP stuff at first.
        if complete.shape[0] < 2:
            return int(candidates[0])

        # Perform the real initialization.
        if self.D == -1:
            self._real_init(grid.shape[1], values[complete])

        # Grab out the relevant sets.
        comp = grid[complete, :]
        cand = grid[candidates, :]
        # the number of candidates
        numcand = cand.shape[0]
        # restrict first dimension to max value (data set size)
        cand = _restrict_point_array(cand)
        
        vals = values[complete]
        

        # Spray a set of candidates around the min so far
        # and put them together with the candidates into cand2
        best_comp = np.argmin(vals)
        c2 = np.random.randn(10, comp.shape[1]) * 0.001 + comp[best_comp, :]
        c2 = _restrict_point_array(c2)
        cand2 = np.vstack((c2, cand))

        # Possibly burn in.
        # TODO: remove
        self.needs_burnin = False
        if self.needs_burnin:
            for mcmc_iter in xrange(self.burnin):
                self.sample_hypers(comp, vals)
                log("BURN %d/%d] mean: %.2f  amp: %.2f "
                                 "noise: %.4f  min_ls: %.4f  max_ls: %.4f"
                                 % (mcmc_iter + 1, self.burnin, self.mean,
                                    np.sqrt(self.amp2), self.noise,
                                    np.min(self.ls), np.max(self.ls)))
            self.needs_burnin = False

        # Sample from hyperparameters.
        # Adjust the candidates to hit ei peaks
        # Delete hyper samples from burnin
        self.hyper_samples = []
        for mcmc_iter in xrange(self.mcmc_iters):
            self.sample_hypers(comp, vals)
            log("%d/%d] mean: %.2f  amp: %.2f  noise: %.4f "
                             "min_ls: %.4f  max_ls: %.4f"
                             % (mcmc_iter + 1, self.mcmc_iters, self.mean,
                                np.sqrt(self.amp2), self.noise,
                                np.min(self.ls), np.max(self.ls)))
        self.dump_hypers()

        overall_ei = self.grad_optimize_ei_over_hypers(cand2, comp, vals, False)
        # Sort all candidates and the ones around the current min by their average EI.
        inds = np.argsort(np.mean(overall_ei, axis=1))[-self.grid_subset:]
        cand2 = cand2[inds, :]
        
        # the first variable is supposed to be the data set size
        # we examine EI on the projected GP where s=smax
        # b = [(grid.vmap.variables[0]['max'], grid.vmap.variables[0]['max'])]  # optimization bounds
        b = [(1,1)]
        for i in xrange(1, cand.shape[1]):
            b.append((0,1))
            #b.append((grid.vmap.variables[i]['min'], grid.vmap.variables[i]['max']))
            

        # pool = multiprocessing.Pool(self.grid_subset)
        # Locally optimize all candidates.
        # results = [pool.apply_async(optimize_pt, args=(
        #            c, b, comp, vals, copy.copy(self))) for c in cand2]
        results = [optimize_pt(c, b, comp, vals, copy.copy(self)) for c in cand2]
        
        # Seems like this appends the optimized candidates to the candidates.
        for res in results:
            # res = res.get(1e8)
            # res[0] = 
            cand = np.vstack((cand, res))
        # pool.close()

        overall_ei = self.grad_optimize_ei_over_hypers(cand, comp, vals, False)
        best_cand = np.argmax(np.mean(overall_ei, axis=1))

        # If the index of the best candidate is larger than the actual number of candidates
        # we also augment and return the candidate set.
        final_candidate = cand[best_cand, :]
        (ms, cs) = self._predict_over_hypers(np.array([final_candidate]), 
                                               comp, vals)
        confidence = cs[0].mean()
        final_candidate[0] = 1 / max(1, confidence)
        self.plot_gp(comp, vals)
        return (int(numcand), final_candidate)

        
    def _predict_over_hypers(self, Xstar, X, y):
        """
        Returns a tuple. First entry is an array of mean predictions 
        for each x* for each hypersample. Second entry is of the same
        form but contains confidence predictions.
        """
        def _predict():
            # The primary covariances for prediction.
            comp_cov = self.cov(X)
            cand_cross = self.cov(X, Xstar)
    
            # Compute the required Cholesky.
            obsv_cov = comp_cov + self.noise * np.eye(X.shape[0])
            obsv_chol = spla.cholesky(obsv_cov, lower=True)
    
            # Predictive things.
            # Solve the linear systems.
            alpha = spla.cho_solve((obsv_chol, True), y - self.mean)
            beta = spla.solve_triangular(obsv_chol, cand_cross, lower=True)
    
            # Predict the marginal means and variances at candidates.
            func_m = np.dot(cand_cross.T, alpha) + self.mean
            func_v = self.amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)
            return (func_m, func_v)
        
        ls = self.ls.copy()
        amp2 = self.amp2
        mean = self.mean
        noise = self.noise
        
        m_predictions = np.zeros((Xstar.shape[0], len(self.hyper_samples)))
        c_predictions = np.zeros((Xstar.shape[0], len(self.hyper_samples)))
        
        for h in range(0, len(self.hyper_samples) - 1):
            hyper = self.hyper_samples[h]
            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]
            (m,c) = _predict()
            m_predictions[:, h] = m
            c_predictions[:, h] = c

        self.mean = mean
        self.amp2 = amp2
        self.noise = noise
        self.ls = ls.copy()
        return (m_predictions,c_predictions)
    
    def plot_gp(self, X, y):
            
        x = np.linspace(0,2,100)[:,np.newaxis]
        s = np.linspace(0,500,100)[:,np.newaxis]

        
        x2 = np.zeros((100,2))
        for i in range(0,100):
            #x2[i,0] = s[i]
            x2[i,0] = 1
            x2[i,1] = x[i]
            #x2[i] = np.array([s[i]],[x[i]])

        
        (m,v) = self._predict_over_hypers(x2, X, y)
        
        mean = np.zeros((100,1))
        for i in range(0,m.shape[0]):
            mean[i] = m[i,:].mean()

        variance = np.zeros((100,1))
        for i in range(0,v.shape[0]):
            variance[i] = v[i,:].mean()
        
        lower_bound = np.zeros((100,1))
        for i in range(0,m.shape[0]):
            lower_bound[i] = m[i,:].mean() - math.sqrt(variance[i])         
        
        
        upper_bound = np.zeros((100,1))
        for i in range(0,m.shape[0]):
            upper_bound[i] = m[i,:].mean() + math.sqrt(variance[i])
        
       
        plt.plot(x, mean)
        plt.fill_between(x[:,0],  upper_bound[:,0], lower_bound[:,0],facecolor='red')
        plt.grid(True)
        
        #for i in range(0,m.shape[1]):
        #    plt.plot(x, m[:,i])
        #plt.show()
        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        ax = fig.gca(projection='3d')
       
        #x_values = np.array(x2[:,0])
        #y_values = np.array(x2[:,1])
        #z_values = np.array(mean[:,0])
        
        ax.plot(X[:,0],X[:,1],y,'r+')
        
        #for i in range(2,X.shape[0]):
        #    ax.plot(X[i,1],y[i], 'ro', label='points')
        
        #X, Y, Z = axes3d.get_test_data(0.05)
        #print X.shape
        #print Y.shape
        #print Z.shape
        #ax.plot(x_values, y_values, z_values, label='mean_curve')
        ax.plot_wireframe(X[:,0],X[:,1],y, rstride=10, cstride=10)

#        z_values = np.array(lower_bound[:,0])
 #       ax.plot(x_values, y_values, z_values, 'r', label='lower_bound')
    
        
        
  #      z_values = np.array(upper_bound[:,0])
   #     ax.plot(x_values, y_values, z_values, 'r', label='upper_bound')
        
        #ax.legend()

        plt.show()

    
    # Adjust points by optimizing EI over a set of hyperparameter samples
    def grad_optimize_ei_over_hypers(self, cands, comp, vals,
                                     compute_grad=True):
        ls = self.ls.copy()
        amp2 = self.amp2
        mean = self.mean
        noise = self.noise

        overall_ei = np.zeros((cands.shape[0], len(self.hyper_samples)))
        if compute_grad:
            overall_grad_ei = np.zeros((cands.shape[0], cands.shape[1], len(self.hyper_samples)))
        for h in range(0, len(self.hyper_samples) - 1):
            hyper = self.hyper_samples[h]
            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]
            if compute_grad:
                (ei, g_ei) = self.grad_optimize_ei(cands, comp, vals, compute_grad)
                overall_grad_ei[:, :, h] = g_ei
            else:
                ei = self.grad_optimize_ei(cands, comp, vals, compute_grad)
            overall_ei[:, h] = ei

        self.mean = mean
        self.amp2 = amp2
        self.noise = noise
        self.ls = ls.copy()

        if compute_grad:
            return (overall_ei, overall_grad_ei)
        else:
            return overall_ei
        

    # Adjust points based on optimizing their ei
    def grad_optimize_ei(self, cand, comp, vals, compute_grad=True):
        """
        No support for pending values.
        """
        best = np.min(vals)

        # The primary covariances for prediction.
        comp_cov = self.cov(comp)
        cand_cross = self.cov(comp, cand)

        # Compute the required Cholesky.
        obsv_cov = comp_cov + self.noise * np.eye(comp.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)

        # Predictive things.
        # Solve the linear systems.
        alpha = spla.cho_solve((obsv_chol, True), vals - self.mean)
        beta = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        # Predict the marginal means and variances at candidates.
        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)

        # Expected improvement
        func_s = np.sqrt(func_v)
        u = (best - func_m) / func_s
        ncdf = sps.norm.cdf(u)
        npdf = sps.norm.pdf(u)
        ei = func_s * (u * ncdf + npdf)

        if not compute_grad:
            return ei
        
        
        cov_grad_func = getattr(gp, 'grad_' + self.cov_func.__name__)
        cand_cross_grad = cov_grad_func(self.ls, comp, cand)

        # Gradients of ei w.r.t. mean and variance
        g_ei_m = -ncdf
        g_ei_s2 = 0.5 * npdf / func_s

        # Apply covariance function
        grad_cross = np.squeeze(cand_cross_grad)

        grad_xp_m = np.dot(alpha.transpose(), grad_cross)
        grad_xp_v = np.dot(-2 * spla.cho_solve(
                (obsv_chol, True), cand_cross).transpose(), grad_cross)

        grad_xp = 0.5 * self.amp2 * (grad_xp_m * g_ei_m + grad_xp_v * g_ei_s2)

        return ei, grad_xp
    
    def sample_hypers(self, comp, vals):
        if self.noiseless:
            self.noise = 1e-3
        self._sample_noisy(comp, vals, self.noiseless)
        self._sample_ls(comp, vals)
        self.hyper_samples.append((self.mean, self.noise, self.amp2, self.ls))

    def _sample_ls(self, comp, vals):
        def logprob(ls):
            if np.any(ls < 0) or np.any(ls > self.max_ls):
                return -np.inf

            cov = (self.amp2 * (self.cov_func(ls, comp, None) + 
                1e-6 * np.eye(comp.shape[0])) + self.noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - self.mean)
            lp = (-np.sum(np.log(np.diag(chol))) - 
                      0.5 * np.dot(vals - self.mean, solve))
            return lp

        self.ls = util.slice_sample(self.ls, logprob, compwise=True)

    def _sample_noisy(self, comp, vals, noisy=True):
        def logprob(hypers):
            mean = hypers[0]
            amp2 = hypers[1]
            if noisy:
                noise = hypers[2]
            else:
                noise = 1e-3

            # This is pretty hacky, but keeps things sane.
            if mean > np.max(vals) or mean < np.min(vals):
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf

            cov = (amp2 * (self.cov_func(self.ls, comp, None) + 
                1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve)

            if noisy:
                # Roll in noise horseshoe prior.
                lp += np.log(np.log(1 + (self.noise_scale / noise) ** 2))

            # Roll in amplitude lognormal prior
            lp -= 0.5 * (np.log(amp2) / self.amp2_scale) ** 2

            return lp

        hypers = util.slice_sample(np.array(
                [self.mean, self.amp2, self.noise]), logprob, compwise=False)
        self.mean = hypers[0]
        self.amp2 = hypers[1]
        if noisy:
            self.noise = hypers[2]
        else:
            self.noise = 1e-3
            
    #===========================================
    #===This part is hopefully only temporary===
    #===========================================
    def __init__(self, expt_dir, covar="Matern52", mcmc_iters=10,
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20):
        self.cov_func = getattr(gp, covar)
        self.locker = Locker()
        self.state_pkl = os.path.join(expt_dir, self.__module__ + ".pkl")
        self.stats_file = os.path.join(expt_dir,
                                   self.__module__ + "_hyperparameters.txt")
        self.mcmc_iters = int(mcmc_iters)
        self.burnin = int(burnin)
        self.needs_burnin = True
        self.pending_samples = int(pending_samples)
        self.D = -1
        self.hyper_iters = 1
        # Number of points to optimize EI over
        self.grid_subset = int(grid_subset)
        self.noiseless = bool(int(noiseless))
        self.hyper_samples = []

        self.noise_scale = 0.1  # horseshoe prior
        self.amp2_scale = 1  # zero-mean log normal prior
        self.max_ls = 10  # top-hat prior on length scales
        try:
            self.compare = GPEIOptChooser(expt_dir + "/comp", covar, mcmc_iters, pending_samples, noiseless, burnin, grid_subset)
        except:
            self.compare = None
            
    def dump_hypers(self):
        self.locker.lock_wait(self.state_pkl)

        # Write the hyperparameters out to a Pickle.
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
        cPickle.dump({ 'dims'   : self.D,
                       'ls'     : self.ls,
                       'amp2'   : self.amp2,
                       'noise'  : self.noise,
                       'mean'   : self.mean },
                     fh)
        fh.close()

        shutil.move(fh.name, self.state_pkl)

        self.locker.unlock(self.state_pkl)

        # Write the hyperparameters out to a human readable file as well
        fh = open(self.stats_file, 'w')
        fh.write('Mean Noise Amplitude <length scales>\n')
        fh.write('-----------ALL SAMPLES-------------\n')
        meanhyps = 0 * np.hstack(self.hyper_samples[0])
        for i in self.hyper_samples:
            hyps = np.hstack(i)
            meanhyps += (1 / float(len(self.hyper_samples))) * hyps
            for j in hyps:
                fh.write(str(j) + ' ')
            fh.write('\n')

        fh.write('-----------MEAN OF SAMPLES-------------\n')
        for j in meanhyps:
            fh.write(str(j) + ' ')
        fh.write('\n')
        fh.close()

    def _real_init(self, dims, values):
        self.locker.lock_wait(self.state_pkl)

        self.randomstate = npr.get_state()
        if os.path.exists(self.state_pkl):
            fh = open(self.state_pkl, 'r')
            state = cPickle.load(fh)
            fh.close()

            self.D = state['dims']
            self.ls = state['ls']
            self.amp2 = state['amp2']
            self.noise = state['noise']
            self.mean = state['mean']

            self.needs_burnin = False
        else:

            # Input dimensionality.
            self.D = dims

            # Initial length scales.
            self.ls = np.ones(self.D)

            # Initial amplitude.
            self.amp2 = np.std(values) + 1e-4

            # Initial observation noise.
            self.noise = 1e-3

            # Initial mean.
            self.mean = np.mean(values)

            # Save hyperparameter samples
            self.hyper_samples.append((self.mean, self.noise, self.amp2,
                                       self.ls))

        self.locker.unlock(self.state_pkl)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.amp2 * (self.cov_func(self.ls, x1, None)
                               + 1e-6 * np.eye(x1.shape[0]))
        else:
            return self.amp2 * self.cov_func(self.ls, x1, x2)