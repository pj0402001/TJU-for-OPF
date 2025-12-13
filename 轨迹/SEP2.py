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
SEP1 = np.array([171.0, 221.0])  # 目标 SEP1
SEP2 = np.array([565.46, 170.9])  # 新目标 SEP2
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
    
    # 2) 加入 UEP 和 SEP1、SEP2
    P_all = np.vstack([P, UEP[None, :], SEP1[None, :], SEP2[None, :]])
    idx_uep = len(P_all) - 3  # UEP的索引
    idx_sep1 = len(P_all) - 2  # SEP1的索引
    idx_sep2 = len(P_all) - 1  # SEP2的索引

    # 3) 构建 kNN 图，并包含 UEP 到 SEP2 的路径
    graph = build_knn_graph(P_all, k=24, max_edge=6.0)
    path_idx2 = dijkstra_path(graph, idx_uep, idx_sep2)

    if path_idx2 is None or len(path_idx2) < 3:
        raise RuntimeError("点云图不连通：UEP 到 SEP2 没有路径。请提供充足数据，或增大 k/max_edge。")

    path_pts2 = P_all[path_idx2]
    path_pts2 = smooth_polyline(path_pts2, win=5, iters=2)
    path_pts2 = resample_polyline(path_pts2, ds=0.4)

    # 4) 选择扰动点 xp1
    xp1 = np.array([569.8, 161.0])  # 指定的扰动点

    # 5) 构造从 xp1 到 UEP 的路径
    P_start = np.vstack([xp1, UEP])  # 从扰动点到 UEP 的路径
    x_of_s_start, Smax_start, _, _ = make_curve_torch(P_start)

    # 6) 从扰动点到 UEP （s → 0）
    s_eps_start = 0.01 * Smax_start
    traj1, s1 = optimize_on_curve(x_of_s_start, s_init=s_eps_start, sense='max',
                                   steps=800, lr=0.05, s_min=0.0, s_max=Smax_start, barrier=1e-4)

    # 7) 从 UEP 微扰出发，进行最小化到 SEP2
    s2_start = max(1e-3, s1 + s_eps_start)
    traj2, s2 = optimize_on_curve(x_of_s_start, s_init=s2_start, sense='min',
                                   steps=800, lr=0.01, s_min=0.0, s_max=Smax_start, barrier=1e-4)

    # 8) 打印结果与绘图
    x_fin = traj1[-1]  # 取到 UE 的结果
    
    def obj(p): return float(np.sum((p / baseMVA) ** 2))
    print(f"[Target UEP] {UEP}, obj={obj(UEP):.6f}")
    print(f"[Target SEP1] {SEP1}, obj={obj(SEP1):.6f}")
    print(f"[Target SEP2] {SEP2}, obj={obj(SEP2):.6f}")
    print(f"[Stage1 end (≈UEP)] {x_fin}, obj={obj(x_fin):.6f}")

    # 9) 绘制轨迹
    plt.figure(figsize=(7.6, 5.8))
    plt.scatter(P[:, 0], P[:, 1], s=6, c='tab:blue', alpha=0.55, label='Feasible Region')
    plt.plot(path_pts2[:, 0], path_pts2[:, 1], c='deeppink', lw=1.4, alpha=0.9)
    plt.scatter([UEP[0]], [UEP[1]], s=160, marker='*', c='limegreen', edgecolors='k', linewidths=0.6, label='UEP')
    plt.scatter([SEP1[0], SEP2[0]], [SEP1[1], SEP2[1]], s=160, marker='*', c='crimson', 
                edgecolors='k', linewidths=0.6, label='SEPs')
    plt.plot(traj1[:, 0], traj1[:, 1], c='purple', lw=2.2, label='Xp2 → UEP')
    plt.plot(traj2[:, 0], traj2[:, 1], c='deeppink', lw=2.2, label='Tragectory')
    plt.scatter([xp1[0]], [xp1[1]], s=90, marker='*', facecolors='none', edgecolors='purple', linewidths=2.0, label='Perturbation point Xp2')

    # 添加箭头，使用中间点作为箭头位置
    mid_point1 = (xp1 + UEP) / 2
    mid_point2 = (UEP + SEP2) / 2
    arrow_props = dict(arrowstyle='->', color='purple', lw=2)
    plt.annotate('', xy=mid_point1, xytext=xp1, arrowprops=arrow_props)  # 从 xp1 到 UEP 的箭头
    plt.annotate('', xy=mid_point2, xytext=UEP, arrowprops={'arrowstyle': '->', 'color': 'deeppink', 'lw': 2})  # 从 UEP 到 SEP2 的箭头

    plt.xlabel("PG1 (MW)"); plt.ylabel("PG2 (MW)"); plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(loc='best', fontsize=9); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()