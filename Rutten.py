'''
A Comparison of Particle Filters for Recursive Track-before-detect
reference: https://ieeexplore.ieee.org/document/1591851
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import block_diag

class Rutten:
    def __init__(self, 
                 Nx, Ny, dx, dy, 
                 ts, q1, q2, 
                 std_x, std_y, SNR,
                 v1, v2, i1, i2, 
                 begin_frame, end_frame, num_frames):
        # 传感器
        self.init_sensor(Nx, Ny, dx, dy)
        # 运动模型
        self.init_motion_model(ts, q1, q2)
        # 测量模型
        self.init_measurement_model(std_x, std_y, SNR)
        # 初始化先验信息
        self.init_prior(v1, v2, i1, i2)

        self.begin_frame = begin_frame
        self.end_frame = end_frame
        self.num_frames = num_frames

        self.track_real = None
        self.pe_real = None
        
        self.measurements = None

        self.num_proposal = None
        self.top_bins = None

        self.track_ssir = None
        self.pe_ssir = None
        self.mse_ssir = None

        self.track_epec = None
        self.pe_epec = None
        self.mse_epec = None

    def init_sensor(self, Nx, Ny, dx, dy):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy

    def init_motion_model(self, ts, q1, q2):
        self.d = 5
        self.ts = ts
        self.q1 = q1
        self.q2 = q2

        # 状态转移矩阵
        F_block = np.array([[1, ts], [0, 1]])
        self.F = block_diag(F_block, F_block, 1) 

        # 过程噪声协方差矩阵
        self.Q_block = np.array([[ts**3 / 3, ts**2 / 2], [ts**2 / 2, ts]])
        self.Q = block_diag(self.Q_block * q1, self.Q_block * q1, ts * q2) 

    def init_measurement_model(self, std_x, std_y, SNR):
        self.std_x = std_x
        self.std_y = std_y
        self.SNR = SNR
        # 点扩散函数前面的系数        
        self.alpha = (self.dx * self.dy) / (2 * np.pi * self.std_x * self.std_y)

        self.intensity = 20
        self.std_n = self.alpha * self.intensity / (10**(SNR / 20))

    def init_prior(self, v1, v2, i1, i2):
        self.v1 = v1
        self.v2 = v2
        self.i1 = i1
        self.i2 = i2
        if not (i1 <= self.intensity <= i2):
            raise ValueError(f"'self.intensity' must be in [{i1}, {i2}].")

    def is_valid_track(self, track):
        if track.ndim != 2 or track.shape[1] != self.d:
            raise ValueError(f"'track' must be of shape (N, {self.d})")
        # x = [x, vx, y, vy, i]
        xk, yk = track[:, 0], track[:, 2]  
        x_valid = np.all((xk >= 0) & (xk <= (self.Nx - 1) * self.dx))
        y_valid = np.all((yk >= 0) & (yk <= (self.Ny - 1) * self.dy))
        return x_valid and y_valid

    def new_track(self, init_state):
        self.track_real = np.zeros((self.num_frames, self.d))
        self.pe_real = np.zeros(self.num_frames)

        init_state = np.asarray(init_state)
        if init_state.shape != (self.d,):
            raise ValueError(f"'init_state' must be of shape ({self.d},)")
        
        init_state[self.d - 1] = self.intensity
        
        for _ in range(100):
            self.track_real[self.begin_frame] = init_state.copy()
            for k in range(self.begin_frame + 1, self.end_frame + 1):
                noise = np.random.multivariate_normal(np.zeros(self.d), self.Q)
                self.track_real[k] = np.dot(self.F, self.track_real[k - 1].T).T + noise
        
            if self.is_valid_track(self.track_real[self.begin_frame: self.end_frame + 1]):
                self.pe_real[self.begin_frame: self.end_frame + 1] = True
                return

        raise ValueError("Failed to generate valid track.")

    def psf(self, i, j, state):
        if state.shape != (self.d,):
            raise ValueError(f"'state' must be of shape ({self.d},)")
        xk = state[0]
        yk = state[2]
        ik = state[4]
        return ik * self.alpha * \
            np.exp(-(i * self.dx - xk)**2 / (2 * self.std_x**2)) * \
                np.exp(-(j * self.dy - yk)**2 / (2 * self.std_y**2))

    def get_measurements(self):
        
        self.measurements = np.random.normal(
            loc=0, 
            scale=self.std_n, 
            size=(self.num_frames, self.Nx, self.Ny)
        )

        for k in range(self.begin_frame, self.end_frame + 1):
            for i in range(0, self.Nx):
                for j in range(0, self.Ny):
                    self.measurements[k, i, j] += self.psf(i, j, self.track_real[k])

    def new_particles_batch(self, num_particles, x1, x2, y1, y2):
        pars = np.zeros((num_particles, self.d))
        pars[:, 0] = np.random.randint(x1, x2 + 1, num_particles) * self.dx
        pars[:, 1] = np.random.uniform(self.v1, self.v2, num_particles)
        pars[:, 2] = np.random.randint(y1, y2 + 1, num_particles) * self.dy
        pars[:, 3] = np.random.uniform(self.v1, self.v2, num_particles)
        pars[:, 4] = np.random.uniform(self.i1, self.i2, num_particles)
        return pars
    
    # 在第k帧，得到强度最高的n个单元
    def get_top_bins(self, k, n):
        flat_indices = np.argpartition(self.measurements[k].ravel(), -n)[-n:]
        return np.column_stack(np.unravel_index(flat_indices, self.measurements[k].shape))

    # 在所有帧中，得到强度最高的n个单元
    def get_top_bins_in_all_frames(self, n):
        self.num_proposal = n
        self.top_bins = np.zeros((self.num_frames, self.num_proposal, 2), dtype=np.int8)
        self.k_pq = self.num_proposal / (self.Nx * self.Ny)
        for k in range(self.num_frames):
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
        pars[:, 4] = np.random.uniform(self.i1, self.i2, num_particles)
        return pars

    def likelihood_ratio_batch(self, states, k, cx, cy):
        
        if states.ndim !=2 or states.shape[1] != self.d:
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
            
            h = ik[n] * self.alpha *\
                np.exp(-(i_grid * self.dx - xk[n])**2 / (2 * self.std_x**2)) * \
                np.exp(-(j_grid * self.dy - yk[n])**2 / (2 * self.std_y**2))
            
            z = self.measurements[k, i_grid, j_grid]
            
            log_lr = -h * (h - 2 * z) / (2 * self.std_n**2)
            log_ans[n] = np.sum(log_lr)

        return np.exp(log_ans)
        
    def ssir_pf(self, num_particles, p0, pb, pd, cx, cy):
        
        t1 = time.perf_counter()

        self.track_ssir = np.zeros((self.num_frames, self.d))
        self.pe_ssir = np.zeros(self.num_frames)
        self.mse_ssir = np.zeros(self.num_frames)

        # 生成粒子
        particles = self.new_particles_batch(num_particles, 0, self.Nx - 1, 0, self.Ny - 1)
        weights = np.ones(num_particles) / num_particles
        
        mask = np.zeros(num_particles, dtype=bool)        
        # 这些粒子标记为存活
        mask[0: int(num_particles * p0)] = True

        for k in range(0, self.num_frames):
            
            # 备份上个时刻的粒子状态
            mask_backup = mask.copy()

            # 在Na个活粒子中杀死NaPd个粒子
            death_count = int(np.sum(mask_backup) * pd)
            if death_count > 0:
                death_idx = np.random.choice(np.where(mask_backup)[0], death_count, replace=False)
                mask[death_idx] = False

            # 在N-Na个死粒子中复活(N - Na)Pb个活粒子
            birth_count = int(np.sum(~mask_backup) * pb)
            if birth_count > 0:
                birth_idx = np.random.choice(np.where(~mask_backup)[0], birth_count, replace=False)
                particles[birth_idx] = self.proposal_particles_batch(birth_count, k)
                mask[birth_idx] = True

            # 仍然活着的粒子，提议分布q就是运动模型
            remaining_alive = mask_backup & mask
            particles[remaining_alive] = (self.F @ particles[remaining_alive].T).T + \
                np.random.multivariate_normal(np.zeros(self.d), self.Q, np.sum(remaining_alive))
                    
            # 计算似然，上一时刻存活的粒子正常计算权重，上一时刻死亡的粒子在正常
            # 计算权重的条件下还要乘以 (p/q)，也就是高强度单元格与所有单元格的比值
            weights = self.likelihood_ratio_batch(particles, k, cx, cy)
            weights[~mask] = 1
            weights[mask & (~mask_backup)] *= self.k_pq
            
            # 权重归一化
            weights = self.normalize_weights(weights)
            
            # 重采样
            resampled_indices = self.systematic_resample(weights, num_particles)
            particles = particles[resampled_indices]
            mask = mask[resampled_indices]
            weights = np.ones(num_particles) / num_particles

            x_est = np.mean(particles[mask], axis=0)
            pe_est = np.mean(mask)

            self.track_ssir[k] = x_est
            self.pe_ssir[k] = pe_est
            
            if self.begin_frame <= k <= self.end_frame:
                dx = self.track_real[k, 0] - x_est[0]
                dy = self.track_real[k, 2] - x_est[2]
                self.mse_ssir[k] = dx**2 + dy**2

            # p0 = pe_est
        
        return time.perf_counter() - t1

    def epec_pf(self, Nc, Nb, p1, pb, pd, cx, cy):
        
        t1 = time.perf_counter()

        self.track_epec = np.zeros((self.num_frames, self.d))
        self.pe_epec = np.zeros(self.num_frames)
        self.mse_epec = np.zeros(self.num_frames)

        # p1 = 0 # 先验存在概率

        # 继续粒子
        par_continuing = self.new_particles_batch(Nc, 0, self.Nx - 1, 0, self.Ny - 1)
        wc = np.ones(Nc) / Nc     

        # 出生粒子
        par_birth = np.zeros((Nb, self.d))
        wb = np.ones(Nb) / Nb    

        for k in range(0, self.num_frames):
            
            # 采样新生粒子
            par_birth = self.proposal_particles_batch(Nb, k)
            # 计算新生粒子权重
            wb = self.likelihood_ratio_batch(par_birth, k, cx, cy) * self.k_pq / Nb
            # 新生粒子权重归一化
            wb_norm = self.normalize_weights(wb)

            # 继续粒子按运动模型采样            
            par_continuing = (self.F @ par_continuing.T).T + \
                np.random.multivariate_normal(np.zeros(self.d), self.Q, Nc)

            # 更新继续粒子权重
            wc =  self.likelihood_ratio_batch(par_continuing, k, cx, cy) / Nc
            # 继续粒子权重归一化
            wc_norm = self.normalize_weights(wc)

            # 计算混合概率
            mb = pb * (1 - p1) * np.sum(wb)
            mc = (1 - pd) * p1 * np.sum(wc)

            mb_norm = mb / (mb + mc)
            mc_norm = mc / (mb + mc)

            # 得到后验存在概率
            p2 = (mb + mc) / (mb + mc + pd * p1 + (1 - pb) * (1 - p1))

            mixed_wb = mb_norm * wb_norm
            mixed_wc = mc_norm * wc_norm

            weight = np.concatenate([mixed_wb, mixed_wc])
            particles = np.concatenate([par_birth, par_continuing], axis=0)
            
            resampled_indices = self.systematic_resample(weight, Nc)
            par_continuing = particles[resampled_indices]
            wc = np.ones(Nc) / Nc
            
            self.track_epec[k] = np.mean(par_continuing, axis=0)
            self.pe_epec[k] = p2

            if self.begin_frame <= k <= self.end_frame:
                dx = self.track_real[k, 0] - self.track_epec[k, 0]
                dy = self.track_real[k, 2] - self.track_epec[k, 2]
                self.mse_epec[k] = dx**2 + dy**2

            p1 = p2

        return time.perf_counter() - t1

    # 权重归一化
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

        # 仅替换零权重为极小值
        weights = np.where(weights == 0, 1e-300, weights)

        weights_sum = np.sum(weights)
        
        return weights / weights_sum

    # 系统重采样
    def systematic_resample(self, weights, num_samples):
        
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
    
        weights = self.normalize_weights(weights)

        u0 = np.random.uniform(0, 1 / num_samples)
        u = u0 + np.arange(num_samples) / num_samples

        # 计算累积权重并强制归一化
        cum_weights = np.cumsum(weights)
        cum_weights[-1] = 1.0 # 确保严格归一化

        # 确定采样索引
        sampled_indices = np.searchsorted(cum_weights, u, side='left')
        sampled_indices = np.clip(sampled_indices, 0, len(weights) - 1)
    
        return sampled_indices

    def plot_pe(self, t0, t1, t2):

        plt.figure()

        plt.axhline(y=0.6, color='r', linestyle='--', linewidth=1, label='Threshold')

        if t0:
            real = self.pe_real
            plt.plot(np.arange(self.num_frames), real, c='r', linestyle='-', label='Real Pe')
            plt.scatter(np.arange(self.num_frames), real, c='r', s=30, marker='o', label='Real Points')
        
        if t1:
            ssir = self.pe_ssir
            plt.plot(np.arange(self.num_frames), ssir, c='g', linestyle='--', label='SSIR Pe')
            plt.scatter(np.arange(self.num_frames), ssir, c='g', s=30, marker='^', label='SSIR Points')

        if t2:
            epec = self.pe_epec
            plt.plot(np.arange(self.num_frames), epec, c='b', linestyle='--', label='EPEC Pe')
            plt.scatter(np.arange(self.num_frames), epec, c='b', s=30, marker='d', label='EPEC Points')

        plt.legend()
        plt.grid()
        plt.show()

    def plot_mse(self, t1, t2):

        plt.figure()
        plt.xticks(np.arange(self.begin_frame, self.end_frame + 1, 1))

        if t1:
            # 用sqrt让量纲一致
            ssir = np.sqrt(self.mse_ssir[self.begin_frame: self.end_frame + 1])
            plt.plot(np.arange(self.begin_frame, self.end_frame + 1, dtype=int), ssir, c='g', linestyle='--', label='SSIR MSE')
            plt.scatter(np.arange(self.begin_frame, self.end_frame + 1, dtype=int), ssir, c='g', s=30, marker='^', label='SSIR Points')

        if t2:
            epec = np.sqrt(self.mse_epec[self.begin_frame: self.end_frame + 1])
            plt.plot(np.arange(self.begin_frame, self.end_frame + 1, dtype=int), epec, c='b', linestyle='--', label='EPEC MSE')
            plt.scatter(np.arange(self.begin_frame, self.end_frame + 1, dtype=int), epec, c='b', s=30, marker='d', label='EPEC Points')

        plt.legend()
        plt.grid()
        plt.show()
    

    def plot_track(self, t0, t1, t2):

        plt.figure()

        if t0:
            real = self.track_real[self.begin_frame: self.end_frame + 1]
            plt.plot(real[:, 0], real[:, 2], c='r', linestyle='-', label='Real Path')
            plt.scatter(real[:, 0], real[:, 2], c='r', s=30, marker='o', label='Real Points')
        
        if t1:
            ssir = self.track_ssir[self.begin_frame: self.end_frame + 1]
            plt.plot(ssir[:, 0], ssir[:, 2], c='g', linestyle='--', label='SSIR Path')
            plt.scatter(ssir[:, 0], ssir[:, 2], c='g', s=30, marker='^', label='SSIR Points')

        if t2:
            epec = self.track_epec[self.begin_frame: self.end_frame + 1]
            plt.plot(epec[:, 0], epec[:, 2], c='b', linestyle='--', label='EPEC Path')
            plt.scatter(epec[:, 0], epec[:, 2], c='b', s=30, marker='d', label='EPEC Points')

        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    
    test = Rutten(
        Nx=20, dx=1, Ny=20, dy=1, 
        ts=1, q1=0.001, q2=0.01, 
        std_x=0.7, std_y=0.7, SNR=3,
        v1=-1, v2=1, i1=15, i2=25, 
        begin_frame=7, end_frame=22, num_frames=30
    )
    
    test.new_track([3, 0.81, 3, 0.54, 20])
    test.get_measurements()

    test.get_top_bins_in_all_frames(30)

    print(test.std_n, test.intensity)

    t1 = test.ssir_pf(num_particles=5000, p0=0, pb=0.05, pd=0.05, cx=3, cy=3)
    print(t1)
    print(test.pe_ssir)
    print(test.mse_ssir)
    
    t2 = test.epec_pf(Nc=3000, Nb=2000, p1=0, pb=0.05, pd=0.05, cx=3, cy=3)
    print(t2)
    print(test.pe_epec)
    print(test.mse_epec)
    
    test.plot_pe(1, 1, 1)
    test.plot_mse(1, 1)
    test.plot_track(1, 1, 1)

    pass
