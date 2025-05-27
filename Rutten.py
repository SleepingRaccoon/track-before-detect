'''
title: A Comparison of Particle Filters for Recursive Track-before-detect
link: https://ieeexplore.ieee.org/document/1591851
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

class Rutten:
    def __init__(self, 
                 ts, q1, q2, 
                 Nx, Ny, dx, dy, std_psf, SNR,
                 v1, v2, k1, k2, 
                 f1, f2, fk):
        
        self.init_motion_model(ts, q1, q2)        
        self.init_measurement_model(Nx, Ny, dx, dy, std_psf, SNR)
        self.init_prior(v1, v2, k1, k2)

        self.f1 = f1 # begin frame
        self.f2 = f2 # end frame
        self.fk = fk # number of frames

        self.real_sk = None
        self.presence = None
        
        self.measurements = None

        self.num_proposal = None
        self.top_bins = None

        self.ssir_sk = None
        self.ssir_pe = None
        self.ssir_mse = None

        self.epec_sk = None
        self.epec_pe = None
        self.epec_mse = None

    def init_motion_model(self, ts, q1, q2):
        self.d = 5
        self.ts = ts
        self.q1 = q1
        self.q2 = q2

        F_block = np.array([[1, ts], [0, 1]])
        self.F = block_diag(F_block, F_block, 1) 

        Q_block = np.array([[ts**3 / 3, ts**2 / 2], [ts**2 / 2, ts]])
        self.Q = block_diag(Q_block * q1, Q_block * q1, ts * q2) 

    def init_measurement_model(self, Nx, Ny, dx, dy, std_psf, SNR):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.std_psf = std_psf
        self.SNR = SNR

        self.alpha = (self.dx * self.dy) / (2 * np.pi * self.std_psf**2)
        self.std_n = 1
        self.intensity = self.std_n * (10**(SNR / 20)) / self.alpha

    def init_prior(self, v1, v2, k1, k2):
        self.v1 = v1
        self.v2 = v2
        self.k1 = k1
        self.k2 = k2
        if not(0 <= k1 <= 1 and 1 <= k2):
            raise ValueError("k1, k2 error.")

    def is_valid_states(self, states):
        if states.ndim != 2 or states.shape[1] != self.d:
            raise ValueError(f"'states' must be of shape (N, {self.d})")
        # s = [x, vx, y, vy, i]
        xk, yk = states[:, 0], states[:, 2]  
        x_valid = np.all((xk >= 0) & (xk <= (self.Nx - 1) * self.dx))
        y_valid = np.all((yk >= 0) & (yk <= (self.Ny - 1) * self.dy))
        return x_valid and y_valid

    def new_states(self, init_state):
        self.real_sk = np.zeros((self.fk, self.d))
        self.presence = np.zeros(self.fk, dtype=bool)

        init_state = np.asarray(init_state)
        if init_state.shape != (self.d,):
            raise ValueError(f"'init_state' must be of shape ({self.d},)")
        init_state[self.d - 1] = self.intensity
        
        for _ in range(100):
            self.real_sk[self.f1] = init_state.copy()
            for k in range(self.f1 + 1, self.f2 + 1):
                noise = np.random.multivariate_normal(np.zeros(self.d), self.Q)
                self.real_sk[k] = np.dot(self.F, self.real_sk[k - 1].T).T + noise
            if self.is_valid_states(self.real_sk[self.f1: self.f2 + 1]):
                self.presence[self.f1: self.f2 + 1] = True
                return

        raise ValueError("Failed to generate valid track.")

    def psf(self, i, j, state):
        if state.shape != (self.d,):
            raise ValueError(f"'state' must be of shape ({self.d},)")
        # s = [x, vx, y, vy, i]
        xk = state[0]
        yk = state[2]
        ik = state[4]
        return ik * self.alpha * \
            np.exp(-((i * self.dx - xk)**2 + (j * self.dy - yk)**2) / (2 * self.std_psf**2))

    def get_measurements(self):
        self.measurements = np.random.normal(
            loc=0, 
            scale=self.std_n, 
            size=(self.fk, self.Nx, self.Ny)
        )
        for k in range(self.f1, self.f2 + 1):
            for i in range(0, self.Nx):
                for j in range(0, self.Ny):
                    self.measurements[k, i, j] += self.psf(i, j, self.real_sk[k])

    def new_particles_batch(self, num_particles, x1, x2, y1, y2):
        pars = np.zeros((num_particles, self.d))
        pars[:, 0] = np.random.randint(x1, x2 + 1, num_particles) * self.dx
        pars[:, 1] = np.random.uniform(self.v1, self.v2, num_particles)
        pars[:, 2] = np.random.randint(y1, y2 + 1, num_particles) * self.dy
        pars[:, 3] = np.random.uniform(self.v1, self.v2, num_particles)
        pars[:, 4] = np.random.uniform(self.k1 * self.intensity, 
                                       self.k2 * self.intensity, num_particles)
        return pars

    def get_top_bins(self, k, n):
        flat_indices = np.argpartition(self.measurements[k].ravel(), -n)[-n:]
        return np.column_stack(np.unravel_index(flat_indices, self.measurements[k].shape))

    def get_top_bins_in_all_frames(self, n):
        self.num_proposal = n
        self.top_bins = np.zeros((self.fk, self.num_proposal, 2), dtype=np.int8)
        self.k_pq = self.num_proposal / (self.Nx * self.Ny)
        for k in range(self.fk):
            self.top_bins[k] = self.get_top_bins(k, self.num_proposal)

    def proposal_particles_batch(self, num_particles, k):
        q = np.random.randint(0, self.num_proposal, num_particles)
        xq = self.top_bins[k, q, 0] * self.dx
        yq = self.top_bins[k, q, 1] * self.dy
        pars = np.zeros((num_particles, self.d))
        pars[:, 0] = xq
        pars[:, 1] = np.random.uniform(self.v1, self.v2, num_particles)
        pars[:, 2] = yq
        pars[:, 3] = np.random.uniform(self.v1, self.v2, num_particles)
        pars[:, 4] = np.random.uniform(self.k1 * self.intensity, 
                                       self.k2 * self.intensity, num_particles)
        return pars

    def likelihood_ratio_batch(self, states, k, cx, cy):
        if states.ndim != 2 or states.shape[1] != self.d:
            raise ValueError(f"'states' must be of shape (N, {self.d})")
        
        N = states.shape[0]
        log_ans = np.zeros(N)

        xk = states[:, 0]  # (N,)
        yk = states[:, 2]  # (N,)
        ik = states[:, 4]

        # (N,)
        xlow = np.clip(np.round(xk / self.dx) - cx, 0, self.Nx - 1).astype(int)
        xhigh = np.clip(np.round(xk / self.dx) + cx, 0, self.Nx - 1).astype(int)
        ylow = np.clip(np.round(yk / self.dy) - cy, 0, self.Ny - 1).astype(int)
        yhigh = np.clip(np.round(yk / self.dy) + cy, 0, self.Ny - 1).astype(int)

        for n in range(N):
            if xlow[n] > xhigh[n] or ylow[n] > yhigh[n]:
                continue
            i_grid, j_grid = np.meshgrid(np.arange(xlow[n], xhigh[n] + 1), 
                                         np.arange(ylow[n], yhigh[n] + 1), 
                                         indexing='ij')
            h = ik[n] * self.alpha * \
                np.exp(-((i_grid * self.dx - xk[n])**2 + (j_grid * self.dy - yk[n])**2) 
                       / (2 * self.std_psf**2))
            z = self.measurements[k, i_grid, j_grid]

            log_lr = -h * (h - 2 * z) / (2 * self.std_n**2)
            log_ans[n] = np.sum(log_lr)

        return np.exp(log_ans)
        
    def ssir_run(self, num_particles, p0, pb, pd, cx, cy):
        
        t1 = time.perf_counter()

        self.ssir_sk = np.zeros((self.fk, self.d))
        self.ssir_pe = np.zeros(self.fk)
        self.ssir_mse = np.zeros(self.fk)

        particles = self.new_particles_batch(num_particles, 0, self.Nx - 1, 0, self.Ny - 1)
        weights = np.ones(num_particles) / num_particles
        mask = np.zeros(num_particles, dtype=bool)        
        mask[0: int(num_particles * p0)] = True

        for k in range(0, self.fk):
            
            mask_backup = mask.copy()

            death_count = int(np.sum(mask_backup) * pd)
            if death_count > 0:
                death_idx = np.random.choice(np.where(mask_backup)[0], death_count, replace=False)
                mask[death_idx] = False

            birth_count = int(np.sum(~mask_backup) * pb)
            if birth_count > 0:
                birth_idx = np.random.choice(np.where(~mask_backup)[0], birth_count, replace=False)
                particles[birth_idx] = self.proposal_particles_batch(birth_count, k)
                mask[birth_idx] = True

            remaining_alive = mask_backup & mask
            particles[remaining_alive] = (self.F @ particles[remaining_alive].T).T + \
                np.random.multivariate_normal(np.zeros(self.d), self.Q, np.sum(remaining_alive))
                    
            weights = self.likelihood_ratio_batch(particles, k, cx, cy)
            weights[~mask] = 1
            weights[mask & (~mask_backup)] *= self.k_pq
            weights = self.normalize_weights(weights)
            
            # Resample
            resampled_indices = self.systematic_resample(weights, num_particles)
            particles = particles[resampled_indices]
            mask = mask[resampled_indices]
            weights = np.ones(num_particles) / num_particles
            
            self.ssir_pe[k] = np.mean(mask)
            
            if self.f1 <= k <= self.f2:
                self.ssir_sk[k] = np.mean(particles[mask], axis=0)
                delta_x = self.real_sk[k, 0] - self.ssir_sk[k, 0]
                delta_y = self.real_sk[k, 2] - self.ssir_sk[k, 2]
                self.ssir_mse[k] = delta_x**2 + delta_y**2
        
        return time.perf_counter() - t1

    def epec_run(self, Nc, Nb, p0, pb, pd, cx, cy):
        
        t1 = time.perf_counter()

        self.epec_sk = np.zeros((self.fk, self.d))
        self.epec_pe = np.zeros(self.fk)
        self.epec_mse = np.zeros(self.fk)

        par_continuing = self.new_particles_batch(Nc, 0, self.Nx - 1, 0, self.Ny - 1)
        wc = np.ones(Nc) / Nc     

        par_birth = np.zeros((Nb, self.d))
        wb = np.ones(Nb) / Nb    

        p1 = p0

        for k in range(0, self.fk):

            par_birth = self.proposal_particles_batch(Nb, k)
            wb = self.likelihood_ratio_batch(par_birth, k, cx, cy) * self.k_pq / Nb
            wb_norm = self.normalize_weights(wb)
           
            par_continuing = (self.F @ par_continuing.T).T + \
                np.random.multivariate_normal(np.zeros(self.d), self.Q, Nc)
            wc =  self.likelihood_ratio_batch(par_continuing, k, cx, cy) / Nc
            wc_norm = self.normalize_weights(wc)


            mb = pb * (1 - p1) * np.sum(wb)
            mc = (1 - pd) * p1 * np.sum(wc)

            mb_norm = mb / (mb + mc)
            mc_norm = mc / (mb + mc)

            p2 = (mb + mc) / (mb + mc + pd * p1 + (1 - pb) * (1 - p1))

            mixed_wb = mb_norm * wb_norm
            mixed_wc = mc_norm * wc_norm

            weight = np.concatenate([mixed_wb, mixed_wc])
            particles = np.concatenate([par_birth, par_continuing], axis=0)
            
            resampled_indices = self.systematic_resample(weight, Nc)
            par_continuing = particles[resampled_indices]
            wc = np.ones(Nc) / Nc
            
            self.epec_pe[k] = p2

            if self.f1 <= k <= self.f2:
                self.epec_sk[k] = np.mean(par_continuing, axis=0)
                delta_x = self.real_sk[k, 0] - self.epec_sk[k, 0]
                delta_y = self.real_sk[k, 2] - self.epec_sk[k, 2]
                self.epec_mse[k] = delta_x**2 + delta_y**2

            p1 = p2

        return time.perf_counter() - t1


    def normalize_weights(self, weights):

        if not isinstance(weights, (list, tuple, np.ndarray)):
            raise TypeError("Input weights must be a list, tuple, or NumPy array.")    
        weights = np.asarray(weights, dtype=np.float64)

        if weights.ndim != 1:
            raise ValueError("Weights must be of shape (N,).")
        if len(weights) == 0:
            raise ValueError("Weight list can't be empty.")
        if np.any(weights < 0):
            raise ValueError("Weights can't contain negative values.")
        if np.all(weights == 0):
            raise ValueError("All weights are zero and can't be normalized.")

        weights = np.where(weights == 0, 1e-300, weights)

        return weights / np.sum(weights)


    def systematic_resample(self, weights, num_samples):
        
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
    
        weights = self.normalize_weights(weights)

        u0 = np.random.uniform(0, 1 / num_samples)
        u = u0 + np.arange(num_samples) / num_samples

        cum_weights = np.cumsum(weights)
        cum_weights[-1] = 1.0

        sampled_indices = np.searchsorted(cum_weights, u, side='left')
        sampled_indices = np.clip(sampled_indices, 0, len(weights) - 1)
    
        return sampled_indices

    def plot_pe(self, t0, t1, t2):

        plt.figure()

        plt.axhline(y=0.6, color='r', linestyle='--', linewidth=1, label='Threshold')

        if t0:
            real = self.presence
            plt.plot(np.arange(self.fk), real, c='r', linestyle='-', label='Real Target')
            plt.scatter(np.arange(self.fk), real, c='r', s=20, marker='o')

        if t1:
            ssir = self.ssir_pe
            plt.plot(np.arange(self.fk), ssir, c='g', linestyle='--', label='SSIR Pe')
            plt.scatter(np.arange(self.fk), ssir, c='g', s=20, marker='x')

        if t2:
            epec = self.epec_pe
            plt.plot(np.arange(self.fk), epec, c='b', linestyle='--', label='EPEC Pe')
            plt.scatter(np.arange(self.fk), epec, c='b', s=20, marker='*')

        plt.title('Probability of Existence')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_track(self, t0, t1, t2):

        plt.figure()

        if t0:
            real = self.real_sk[self.f1: self.f2 + 1]
            plt.plot(real[:, 0], real[:, 2], c='r', linestyle='-', label='Real Path')
            plt.scatter(real[:, 0], real[:, 2], c='r', s=20, marker='o')
        
        if t1:
            ssir = self.ssir_sk[self.f1: self.f2 + 1]
            plt.plot(ssir[:, 0], ssir[:, 2], c='g', linestyle='--', label='SSIR Path')
            plt.scatter(ssir[:, 0], ssir[:, 2], c='g', s=20, marker='x')

        if t2:
            epec = self.epec_sk[self.f1: self.f2 + 1]
            plt.plot(epec[:, 0], epec[:, 2], c='b', linestyle='--', label='EPEC Path')
            plt.scatter(epec[:, 0], epec[:, 2], c='b', s=30, marker='*')

        plt.title("Track Comparison")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_rmse(self, t1, t2):

        plt.figure()
        plt.xticks(np.arange(self.f1, self.f2 + 1, 1))

        if t1:
            ssir = np.sqrt(self.ssir_mse[self.f1: self.f2 + 1])
            plt.plot(np.arange(self.f1, self.f2 + 1, dtype=int), ssir, c='g', linestyle='--', label='SSIR RMSE')
            plt.scatter(np.arange(self.f1, self.f2 + 1, dtype=int), ssir, c='g', s=20, marker='x')

        if t2:
            epec = np.sqrt(self.epec_mse[self.f1: self.f2 + 1])
            plt.plot(np.arange(self.f1, self.f2 + 1, dtype=int), epec, c='b', linestyle='--', label='EPEC RMSE')
            plt.scatter(np.arange(self.f1, self.f2 + 1, dtype=int), epec, c='b', s=20, marker='*')

        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    
    test = Rutten(
        ts=1, q1=0.002, q2=0.01, 
        Nx=20, Ny=20, dx=1, dy=1, std_psf=0.7, SNR=3,
        v1=-1, v2=1, k1=0.5, k2=1.5, 
        f1=7, f2=22, fk=30
    )
    
    test.new_states([4, 0.76, 4, 0.45, 20])
    test.get_measurements()

    test.get_top_bins_in_all_frames(20)

    print(test.std_n, test.intensity, test.intensity * test.alpha)

    t1 = test.ssir_run(num_particles=4000, p0=0, pb=0.05, pd=0.05, cx=3, cy=3)
    print(t1)
    print(test.ssir_pe)
    print(test.ssir_mse)

    t2 = test.epec_run(Nc=2000, Nb=2000, p0=0, pb=0.05, pd=0.05, cx=3, cy=3)
    print(t2)
    print(test.epec_pe)
    print(test.epec_mse)

    test.plot_pe(1, 1, 1)
    test.plot_track(1, 1, 1)
    test.plot_rmse(1, 1)

    pass
