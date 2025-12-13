import math
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim.optimizer import Optimizer

# ============== TJUv4 (优化版) ==============
class TJU_v4(Optimizer):
    def __init__(self, params, lr=6e-2, betas=(0.9, 0.999), eps=1e-8,
                 warmup=10, hessian_scale=0.05, total_steps=3000):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        warmup=warmup, init_lr=lr / 1000.0, base_lr=lr,
                        hessian_scale=hessian_scale, total_steps=total_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st['step'] = 0
                    st['exp_avg'] = torch.zeros_like(p)
                    st['exp_avg_sq'] = torch.zeros_like(p)

                st['step'] += 1
                step = st['step']
                lr = self._lr(group, step)

                m, v = st['exp_avg'], st['exp_avg_sq']
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                denom = (v.sqrt() / math.sqrt(bc2)).add_(1e-8).add_(1e-3)  # 防止分母为零
                p.add_(-m / denom * lr / bc1)
        return None

    def _lr(self, g, step):
        if step <= g['warmup']:
            return g['init_lr'] + (g['base_lr'] - g['init_lr']) * step / g['warmup']
        return g['base_lr']

# ============== 数据加载与处理 ==============
baseMVA = 100.0
SEP1 = np.array([171.0, 221.0])  # 目标 SEP
SEP2 =  np.array([565.46, 170.9])
UEP = np.array([569.84, 161.20])  # UEP

def load_feasible_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    num_cols = [c for c in df.columns if pd.to_numeric(df[c], errors='coerce').notna().mean() > 0.9]
    if len(num_cols) < 2:
        raise RuntimeError("CSV 至少需要两列数值")
    A = pd.to_numeric(df[num_cols[0]], errors='coerce').to_numpy(float)
    B = pd.to_numeric(df[num_cols[1]], errors='coerce').to_numpy(float)
    dA = np.min(np.hypot(A - UEP[0], B - UEP[1]))
    dB = np.min(np.hypot(B - UEP[0], A - UEP[1]))
    P = np.c_[B, A] if dB < dA else np.c_[A, B]
    P = P[~np.isnan(P).any(axis=1)]
    P = np.unique(P, axis=0)
    return P

def nearest_idx(P: np.ndarray, q: np.ndarray) -> int:
    return int(np.argmin(np.sum((P - q[None, :]) ** 2, axis=1)))

def build_knn_graph(P: np.ndarray, k: int = 24, max_edge: float = None):
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    d, idx = tree.query(P, k=k)
    graph = [[] for _ in range(len(P))]
    for i in range(len(P)):
        for j, dist in zip(idx[i, 1:], d[i, 1:]):  # 跳过自身
            if max_edge is not None and dist > max_edge:
                continue
            graph[i].append((int(j), float(dist)))
    return graph

def dijkstra_path(graph, src: int, dst: int):
    N = len(graph)
    dist = [float('inf')] * N
    prev = [-1] * N
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, i = heapq.heappop(pq)
        if d != dist[i]: continue
        if i == dst: break
        for j, w in graph[i]:
            nd = d + w
            if nd < dist[j]:
                dist[j] = nd
                prev[j] = i
                heapq.heappush(pq, (nd, j))

    if dist[dst] == float('inf'): return None
    path = []
    current = dst
    while current != -1:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path

def smooth_polyline(P: np.ndarray, win: int = 5, iters: int = 2):
    Q = P.copy()
    for _ in range(iters):
        for i in range(1, len(Q) - 1):
            L = max(0, i - win)
            R = min(len(Q), i + win + 1)
            Q[i] = Q[L:R].mean(axis=0)
    return Q

def resample_polyline(P: np.ndarray, ds: float = 0.4):
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    S = np.concatenate([[0.0], np.cumsum(seg)])
    L = S[-1]
    n = int(max(2, math.ceil(L / ds)))
    s_new = np.linspace(0.0, L, n)
    Q = []
    j = 1
    for s in s_new:
        while s > S[j] and j < len(S) - 1: j += 1
        s0, s1 = S[j - 1], S[j]
        w = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
        q = P[j - 1] + w * (P[j] - P[j - 1])
        Q.append(q)
    return np.array(Q)

def make_curve_torch(P: np.ndarray):
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    S = np.concatenate([[0.0], np.cumsum(seg)])
    S_t = torch.tensor(S, dtype=torch.float32)
    P_t = torch.tensor(P, dtype=torch.float32)

    def x_of_s(s_scalar: torch.Tensor):
        s = torch.clamp(s_scalar, S_t[0], S_t[-1])
        k = torch.searchsorted(S_t, s)
        k = torch.clamp(k, 1, len(S_t) - 1)
        s0 = S_t[k - 1]
        s1 = S_t[k]
        w = (s - s0) / torch.clamp(s1 - s0, min=1e-8)
        x = P_t[k - 1] + w * (P_t[k] - P_t[k - 1])
        return x

    return x_of_s, float(S[-1]), S, P

def f_radial(x: torch.Tensor):
    return torch.sum((x / baseMVA) ** 2)

def optimize_on_curve(x_of_s, s_init: float, sense: str,
                      steps: int, lr: float, s_min: float, s_max: float,
                      barrier: float = 1e-3):
    s = torch.nn.Parameter(torch.tensor([s_init], dtype=torch.float32))
    opt = TJU_v4([s], lr=lr, warmup=5, total_steps=steps)
    traj = []
    for _ in range(steps):
        opt.zero_grad()
        x = x_of_s(s[0])
        f = f_radial(x)
        loss = -f if sense == 'max' else f
        
        # 处理目标附近的约束
        if barrier > 0:
            loss += barrier * ((torch.relu(torch.tensor(s_min) - s[0])) ** 2 +
                               (torch.relu(s[0] - torch.tensor(s_max))) ** 2)
        
        # 引入扰动以便探索
        if sense == 'min' and s[0] > 0.0:
            loss += 0.03 * torch.tensor(np.random.randn())  # 添加随机扰动项，确保是标量
            
        # 动态调整学习率
        if sense == 'min' and s[0] < s_init:
            lr *= 0.5
        
        loss.backward()
        opt.step()
        with torch.no_grad():
            traj.append(x_of_s(s[0]).detach().cpu().numpy().copy())
    
    return np.vstack(traj), float(s.detach().cpu().numpy())

# ============== 主流程 ==============
def main():
    # 1) 读取点云数据
    P = load_feasible_csv("5节点数据.csv")
    # 2) 加入 UEP 和 SEP1
    P_all = np.vstack([P, UEP[None, :], SEP1[None, :]])
    idx_uep = len(P_all) - 2
    idx_sep1 = len(P_all) - 1

    # 3) 构建 kNN 图并找到 UEP 到 SEP1 的最短路径
    graph = build_knn_graph(P_all, k=24, max_edge=6.0)
    path_idx = dijkstra_path(graph, idx_uep, idx_sep1)
    if path_idx is None or len(path_idx) < 3:
        raise RuntimeError("点云图不连通：UEP 到 SEP1 没有路径。请提供充足数据，或增大 k/max_edge。")

    path_pts = P_all[path_idx]
    path_pts = smooth_polyline(path_pts, win=5, iters=2)
    path_pts = resample_polyline(path_pts, ds=0.4)

    # 4) 构造曲线 C(s)，选择扰动点 xp1
    x_of_s, Smax, _, _ = make_curve_torch(path_pts)
    s_eps = 0.01 * Smax
    xp1 = x_of_s(torch.tensor([s_eps], dtype=torch.float32)).detach().cpu().numpy()[0]

    # 5) 阶段 1：最大化 f，回到 UEP（s → 0）
    traj1, s1 = optimize_on_curve(x_of_s, s_init=s_eps, sense='max',
                                  steps=800, lr=0.05, s_min=0.0, s_max=Smax, barrier=1e-4)

    # 6) 阶段 2：从 UEP 微扰出发，最小化 f 到 SEP1
    s2_start = max(1e-3, s1 + s_eps)
    traj2, s2 = optimize_on_curve(x_of_s, s_init=s2_start, sense='min',
                                  steps=70000, lr=0.01, s_min=0.0, s_max=Smax, barrier=1e-4)

    x_fin = x_of_s(torch.tensor([s2], dtype=torch.float32)).detach().cpu().numpy()

    # 7) 打印结果
    def obj(p): return float(np.sum((p / baseMVA) ** 2))
    print(f"[Target UEP] {UEP}, obj={obj(UEP):.6f}")
    print(f"[Target SEP1] {SEP1}, obj={obj(SEP1):.6f}")
    print(f"[Stage1 end (≈UEP)] {x_fin}, obj={obj(x_fin):.6f}")

# 8) 画出轨迹 - 修改了SEP标签显示，并添加箭头
    plt.figure(figsize=(7.6, 5.8))
    plt.scatter(P[:, 0], P[:, 1], s=6, c='tab:blue', alpha=0.55, label='Feasible Region')
    plt.plot(path_pts[:, 0], path_pts[:, 1], c='deeppink', lw=1.4, alpha=0.9)
    plt.scatter([UEP[0]], [UEP[1]], s=160, marker='*', c='limegreen', edgecolors='k', linewidths=0.6, label='UEP')
    
    # 合并SEP标签 - 只显示一个SEP标签
    plt.scatter([SEP1[0], SEP2[0]], [SEP1[1], SEP2[1]], s=160, marker='*', c='crimson', 
                edgecolors='k', linewidths=0.6, label='SEP')
    
    # 绘制轨迹线并添加箭头
    plt.plot(traj1[:, 0], traj1[:, 1], c='purple', lw=2.2, label='Xp1 → UEP')
    plt.plot(traj2[:, 0], traj2[:, 1], c='deeppink', lw=2.2, label='Tragetcory')
    arrow_idx = len(traj1) // 2
    if arrow_idx > 0 and arrow_idx < len(traj1) - 1:
        # 计算箭头方向（切线方向）
        dx = traj2[arrow_idx + 1, 0] - traj2[arrow_idx - 1, 0]
        dy = traj2[arrow_idx + 1, 1] - traj2[arrow_idx - 1, 1]
        
        # 添加箭头
        plt.annotate('', xytext=traj2[arrow_idx], 
                    xy=(traj1[arrow_idx, 0] + dx * 0.1, traj1[arrow_idx, 1] + dy * 0.1),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2, 
                                  shrinkA=0, shrinkB=0, mutation_scale=15))
    # 在UEP → SEP1轨迹上添加箭头
    # 选择轨迹中间点作为箭头位置
    arrow_idx = len(traj2) // 2
    if arrow_idx > 0 and arrow_idx < len(traj2) - 1:
        # 计算箭头方向（切线方向）
        dx = traj2[arrow_idx + 1, 0] - traj2[arrow_idx - 1, 0]
        dy = traj2[arrow_idx + 1, 1] - traj2[arrow_idx - 1, 1]
        
        # 添加箭头
        plt.annotate('', xytext=traj2[arrow_idx], 
                    xy=(traj2[arrow_idx, 0] + dx * 0.1, traj2[arrow_idx, 1] + dy * 0.1),
                    arrowprops=dict(arrowstyle='->', color='deeppink', lw=2, 
                                  shrinkA=0, shrinkB=0, mutation_scale=15))
    
    plt.scatter([xp1[0]], [xp1[1]], s=90, marker='*', facecolors='none', edgecolors='purple', linewidths=2.0, label='Perturbation point Xp1')
    plt.xlabel("PG1 (MW)"); plt.ylabel("PG2 (MW)"); plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(loc='best', fontsize=9); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()