import re
import math
import time
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# ========== 工具：解析 MATPOWER case .m 文件 ==========
def load_matpower_case(file_path: str):
    """
    轻量解析 MATPOWER .m 文件，提取 baseMVA、bus、gen、branch。
    要求：
      mpc.baseMVA = <num>;
      mpc.bus = [ ... ];
      mpc.gen = [ ... ];
      mpc.branch = [ ... ];
    返回: dict(baseMVA, bus(np.ndarray), gen(np.ndarray), branch(np.ndarray))
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    def extract_block(name):
        pattern = rf"mpc\.{name}\s*=\s*\[(.*?)\];"
        m = re.search(pattern, txt, flags=re.S | re.M)
        if not m:
            raise ValueError(f"无法在 {file_path} 中找到 {name} 数据块（请确认是 MATPOWER 案例文件）")
        block = m.group(1).strip()
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if '%' in line:
                line = line.split('%', 1)[0].strip()
            if not line:
                continue
            line = line.rstrip(';').strip()
            if not line:
                continue
            tokens = re.split(r"[\s\t]+", line)
            nums = [float(tok) for tok in tokens if tok]
            if nums:
                rows.append(nums)
        return np.array(rows, dtype=float)

    m = re.search(r"mpc\.baseMVA\s*=\s*([0-9.+\-Ee]+)\s*;", txt)
    if not m:
        raise ValueError("未找到 mpc.baseMVA（请确认提供的是 MATPOWER 案例文件）")
    baseMVA = float(m.group(1))

    bus = extract_block('bus')
    gen = extract_block('gen')
    branch = extract_block('branch')

    return {
        'baseMVA': baseMVA,
        'bus': bus,
        'gen': gen,
        'branch': branch
    }

# ========== 2736-bus AC 可行性扫描器（稀疏版） ==========
class ACOPF2736Analyzer:
    def __init__(self,
                 case_file='case2736sp.m',
                 n_points=8,
                 genA_bus=None,
                 genB_bus=None,
                 range_frac=(0.40, 0.70),
                 per_bus_frac=None,
                 warm_start=True,
                 solver_opts=None):
        """
        case_file : MATPOWER 案例文件路径（如 case2736sp.m）
        n_points  : 每个维度扫描点数（总求解数为 n_points^2）
        genA_bus, genB_bus : 两台扫描机组所在母线号（不指定则自动选择 Pmax 最大的两台在运机组，尽量避开参考母线且不同母线）
        range_frac: 默认扫描相对范围（相对于各自 Pmax），如 (0.40, 0.70)
        per_bus_frac: 针对特定母线自定义百分比区间，如 {191:(0.50,0.53), 119:(0.57,0.64)}
        warm_start: 是否使用上一可行解作为下一点初值
        solver_opts: Ipopt 求解器选项 dict
        """
        data = load_matpower_case(case_file)
        self.baseMVA = data['baseMVA']
        self.bus = data['bus']
        self.gen = data['gen']
        self.branch = data['branch']

        # 维度
        self.n_bus = self.bus.shape[0]
        self.n_gen = self.gen.shape[0]
        self.n_branch = self.branch.shape[0]

        # 母线数据
        # bus: [BUS_I, TYPE, PD, QD, GS, BS, AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN]
        self.bus_i = self.bus[:, 0].astype(int)
        self.bus_type = self.bus[:, 1].astype(int)
        self.Pd = self.bus[:, 2] / self.baseMVA
        self.Qd = self.bus[:, 3] / self.baseMVA
        self.Gs = self.bus[:, 4] / self.baseMVA
        self.Bs = self.bus[:, 5] / self.baseMVA
        self.Vm0 = self.bus[:, 7]
        self.Va0_deg = self.bus[:, 8]
        self.Vmax = self.bus[:, 11]
        self.Vmin = self.bus[:, 12]

        # 发电机数据
        # gen: [GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, STATUS, PMAX, PMIN, ...]
        self.gen_bus = self.gen[:, 0].astype(int)
        self.gen_status = self.gen[:, 7].astype(int)
        self.Pmax = self.gen[:, 8] / self.baseMVA
        self.Pmin = self.gen[:, 9] / self.baseMVA
        self.Qmax = self.gen[:, 3] / self.baseMVA
        self.Qmin = self.gen[:, 4] / self.baseMVA
        self.Vg_set = self.gen[:, 5]

        # 在运机组索引
        self.active_gen_idx = [i for i in range(self.n_gen) if self.gen_status[i] == 1]

        # 母线 -> 在运机组索引列表
        self.bus_to_gens = {int(b): [] for b in self.bus_i}
        for gi in self.active_gen_idx:
            self.bus_to_gens[int(self.gen_bus[gi])].append(gi)

        # 参考母线
        slack_candidates = self.bus_i[self.bus_type == 3]
        if len(slack_candidates) == 0:
            raise ValueError("未找到参考母线 (bus type == 3)")
        self.slack_bus = int(slack_candidates[0])
        self.busnum_to_idx = {int(self.bus_i[k]): k for k in range(self.n_bus)}

        # 构建 Ybus（稀疏）：邻接列表及 G/B 值
        self._build_Ybus_sparse()

        # 选择扫描机组
        if genA_bus is None or genB_bus is None:
            cand = sorted(self.active_gen_idx, key=lambda i: self.Pmax[i], reverse=True)
            pick = []
            seen_bus = set()
            for gi in cand:
                b = int(self.gen_bus[gi])
                if b == self.slack_bus:
                    continue
                if b in seen_bus:
                    continue
                seen_bus.add(b)
                pick.append(gi)
                if len(pick) == 2:
                    break
            if len(pick) < 2:
                pick = cand[:2]
            self.genA_idx, self.genB_idx = pick[0], pick[1]
            self.genA_bus = int(self.gen_bus[self.genA_idx])
            self.genB_bus = int(self.gen_bus[self.genB_idx])
        else:
            self.genA_bus = int(genA_bus)
            self.genB_bus = int(genB_bus)
            self.genA_idx = self._find_active_gen_index_by_bus(self.genA_bus)
            self.genB_idx = self._find_active_gen_index_by_bus(self.genB_bus)

        # 扫描范围设定（支持 per_bus_frac）
        self.per_bus_frac = per_bus_frac or {}
        def frac_range_for(bus_no, gi):
            if bus_no in self.per_bus_frac:
                lo, hi = self.per_bus_frac[bus_no]
            else:
                lo, hi = range_frac
            pmin = max(self.Pmin[gi], lo * self.Pmax[gi])
            pmax = min(self.Pmax[gi], hi * self.Pmax[gi])
            if pmin > pmax:
                pmin = pmax = self.Pmin[gi]
            return pmin, pmax

        self.pA_min_pu, self.pA_max_pu = frac_range_for(self.genA_bus, self.genA_idx)
        self.pB_min_pu, self.pB_max_pu = frac_range_for(self.genB_bus, self.genB_idx)

        self.n_points = int(n_points)
        self.warm_start = bool(warm_start)
        self.solver_opts = solver_opts or {
            'max_iter': 600,
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
            'print_level': 0,
            'linear_solver': 'mumps'
        }

        # 统计
        self.feasible_times = []
        self.avg100_reported = False
        self.last_solution = None  # 用于暖启动

        # 打印信息
        print("===== 2736 节点 AC-OPF 可行性扫描 =====")
        print(f"BaseMVA: {self.baseMVA}")
        print(f"母线数: {self.n_bus}, 发电机数: {self.n_gen}, 支路数: {self.n_branch}")
        print(f"参考母线: {self.slack_bus}")
        total_Pd = float(np.sum(self.bus[:, 2]))
        print(f"系统总有功负荷: {total_Pd:.1f} MW")

        # 打印 Pmax Top-10（在运）
        top = sorted(self.active_gen_idx, key=lambda i: self.Pmax[i], reverse=True)[:10]
        print("在运机组 Pmax Top-10: [gen_idx | bus | Pmax(MW)]")
        for gi in top:
            print(f"  #{gi+1:<4} | {int(self.gen_bus[gi]):<5} | {self.Pmax[gi]*self.baseMVA:>9.1f}")

        print(f"\n扫描机组A: gen#{self.genA_idx+1} @ bus {self.genA_bus} (Pmax={self.Pmax[self.genA_idx]*self.baseMVA:.1f} MW)")
        print(f"扫描机组B: gen#{self.genB_idx+1} @ bus {self.genB_bus} (Pmax={self.Pmax[self.genB_idx]*self.baseMVA:.1f} MW)")
        print("扫描范围（MW）:")
        print(f"  A: [{self.pA_min_pu*self.baseMVA:.2f}, {self.pA_max_pu*self.baseMVA:.2f}]"
              f" ({100*self.pA_min_pu/self.Pmax[self.genA_idx]:.1f}% ~ {100*self.pA_max_pu/self.Pmax[self.genA_idx]:.1f}% Pmax)")
        print(f"  B: [{self.pB_min_pu*self.baseMVA:.2f}, {self.pB_max_pu*self.baseMVA:.2f}]"
              f" ({100*self.pB_min_pu/self.Pmax[self.genB_idx]:.1f}% ~ {100*self.pB_max_pu/self.Pmax[self.genB_idx]:.1f}% Pmax)")
        print(f"网格密度: {self.n_points} × {self.n_points} = {self.n_points*self.n_points} 点\n")

    def _find_active_gen_index_by_bus(self, bus_no: int):
        idxs = [i for i in self.active_gen_idx if int(self.gen_bus[i]) == int(bus_no)]
        if not idxs:
            raise ValueError(f"未找到位于母线 {bus_no} 的在运发电机（或 STATUS=0）")
        return idxs[0]

    def _build_Ybus_sparse(self):
        """
        构建稀疏 Ybus：为每个母线 i 维护相邻母线列表 neighbors[i] 及对应 Gij/Bij。
        """
        nb = self.n_bus
        # 临时：字典形式存储复数导纳
        Y = {i: {} for i in range(nb)}

        # 母线并联导纳
        for i in range(nb):
            if abs(self.Gs[i]) > 0 or abs(self.Bs[i]) > 0:
                Y[i][i] = Y[i].get(i, 0+0j) + complex(self.Gs[i], self.Bs[i])

        # 逐支路累计
        for k in range(self.n_branch):
            fbus = int(self.branch[k, 0]); tbus = int(self.branch[k, 1])
            r, x, b = self.branch[k, 2:5]
            ratio = self.branch[k, 8]
            angle_deg = self.branch[k, 9]
            status = int(self.branch[k, 10])
            if status == 0:
                continue

            if r == 0 and x == 0:
                continue

            f = self.busnum_to_idx[fbus]
            t = self.busnum_to_idx[tbus]

            z = complex(r, x)
            y = 1 / z if abs(z) > 1e-12 else 0+0j
            jb = 1j * (b / 2.0)

            tap = ratio if ratio != 0 else 1.0
            shift_rad = math.radians(angle_deg)
            t_complex = tap * complex(math.cos(shift_rad), math.sin(shift_rad))

            # stamp
            Y[f][f] = Y[f].get(f, 0+0j) + (y + jb) / (t_complex * np.conj(t_complex))
            Y[t][t] = Y[t].get(t, 0+0j) + (y + jb)
            Y[f][t] = Y[f].get(t, 0+0j) - y / np.conj(t_complex)
            Y[t][f] = Y[t].get(f, 0+0j) - y / (t_complex)

        # 转换为邻接列表
        self.neigh = []
        self.G_lists = []
        self.B_lists = []
        for i in range(nb):
            # 确保包含对角元素
            if i not in Y[i]:
                Y[i][i] = 0+0j
            js = sorted(Y[i].keys())
            self.neigh.append(js)
            gij = [Y[i][j].real for j in js]
            bij = [Y[i][j].imag for j in js]
            self.G_lists.append(gij)
            self.B_lists.append(bij)

    def solve_feasibility(self, pA_pu, pB_pu):
        """
        给定两台机组(A,B)的 Pg（pu），建立并求解 AC 可行性判定模型（目标=0）。
        返回: (feasible(bool), solution(dict or None), time_sec(float))
        """
        t0 = time.time()
        try:
            m = pyo.ConcreteModel()
            m.BUS = pyo.RangeSet(1, self.n_bus)
            m.GEN = pyo.RangeSet(1, self.n_gen)

            # Vm/Va 变量
            m.Vm = pyo.Var(m.BUS, bounds=lambda _, i: (self.Vmin[i-1], self.Vmax[i-1]))
            m.Va = pyo.Var(m.BUS, bounds=(-math.pi, math.pi))

            # Pg/Qg 边界（停机机组上下界为(0,0)）
            def pg_bounds(_, g):
                gi = g - 1
                return (0.0, 0.0) if self.gen_status[gi] == 0 else (self.Pmin[gi], self.Pmax[gi])

            def qg_bounds(_, g):
                gi = g - 1
                return (0.0, 0.0) if self.gen_status[gi] == 0 else (self.Qmin[gi], self.Qmax[gi])

            m.Pg = pyo.Var(m.GEN, bounds=pg_bounds)
            m.Qg = pyo.Var(m.GEN, bounds=qg_bounds)

            # 初值：若有上一可行解则暖启动，否则用 case 值
            if self.last_solution and self.warm_start:
                Vm0 = self.last_solution['Vm']
                Va0 = [math.radians(va) for va in self.last_solution['Va_deg']]
                Pg0 = [p / self.baseMVA for p in self.last_solution['Pg_MW']]
                Qg0 = [q / self.baseMVA for q in self.last_solution['Qg_MVar']]
            else:
                Vm0 = [min(max(self.Vm0[i], self.Vmin[i]), self.Vmax[i]) for i in range(self.n_bus)]
                Va0 = [math.radians(self.Va0_deg[i]) for i in range(self.n_bus)]
                Pg0 = [min(max(self.gen[i, 1] / self.baseMVA, self.Pmin[i]), self.Pmax[i]) for i in range(self.n_gen)]
                Qg0 = [min(max(self.gen[i, 2] / self.baseMVA, self.Qmin[i]), self.Qmax[i]) for i in range(self.n_gen)]

            for i in m.BUS:
                m.Vm[i].value = Vm0[i-1]
                m.Va[i].value = Va0[i-1]
            for g in m.GEN:
                gi = g - 1
                if self.gen_status[gi] == 0:
                    m.Pg[g].fix(0.0)
                    m.Qg[g].fix(0.0)
                else:
                    m.Pg[g].value = Pg0[gi]
                    m.Qg[g].value = Qg0[gi]

            # 用机组 VG 覆盖对应母线电压初值（夹在界内）
            for gi in self.active_gen_idx:
                bno = int(self.gen_bus[gi])
                i = self.busnum_to_idx[bno] + 1
                vg = float(self.Vg_set[gi])
                vg = min(max(vg, self.Vmin[i-1]), self.Vmax[i-1])
                m.Vm[i].value = vg

            # 固定两台扫描机组的 Pg
            m.Pg[self.genA_idx + 1].fix(pA_pu)
            m.Pg[self.genB_idx + 1].fix(pB_pu)

            # 参考母线相角
            slack_pos = self.busnum_to_idx[self.slack_bus] + 1
            m.ref = pyo.Constraint(expr=m.Va[slack_pos] == 0.0)

            # 目标：0
            m.OBJ = pyo.Objective(expr=0.0, sense=pyo.minimize)

            # 功率平衡（按稀疏邻接求和）
            neigh = self.neigh
            G_lists = self.G_lists
            B_lists = self.B_lists

            def P_balance_rule(_m, i):
                idx = i - 1
                Pi = 0.0
                js = neigh[idx]
                gij = G_lists[idx]; bij = B_lists[idx]
                for k in range(len(js)):
                    j = js[k] + 1  # Pyomo 索引
                    Pi += _m.Vm[i] * _m.Vm[j] * (gij[k] * pyo.cos(_m.Va[i] - _m.Va[j]) + bij[k] * pyo.sin(_m.Va[i] - _m.Va[j]))
                pg_sum = 0.0
                bus_no = int(self.bus_i[idx])
                for gi in self.bus_to_gens.get(bus_no, []):
                    pg_sum += _m.Pg[gi + 1]
                return pg_sum - self.Pd[idx] == Pi

            def Q_balance_rule(_m, i):
                idx = i - 1
                Qi = 0.0
                js = neigh[idx]
                gij = G_lists[idx]; bij = B_lists[idx]
                for k in range(len(js)):
                    j = js[k] + 1
                    Qi += _m.Vm[i] * _m.Vm[j] * (gij[k] * pyo.sin(_m.Va[i] - _m.Va[j]) - bij[k] * pyo.cos(_m.Va[i] - _m.Va[j]))
                qg_sum = 0.0
                bus_no = int(self.bus_i[idx])
                for gi in self.bus_to_gens.get(bus_no, []):
                    qg_sum += _m.Qg[gi + 1]
                return qg_sum - self.Qd[idx] == Qi

            m.Pbal = pyo.Constraint(m.BUS, rule=P_balance_rule)
            m.Qbal = pyo.Constraint(m.BUS, rule=Q_balance_rule)

            # 求解
            solver = SolverFactory('ipopt')
            for k, v in self.solver_opts.items():
                solver.options[k] = v
            res = solver.solve(m, tee=False)
            t_used = time.time() - t0

            term = res.solver.termination_condition
            feasible = term in (pyo.TerminationCondition.optimal,
                                pyo.TerminationCondition.locallyOptimal,
                                pyo.TerminationCondition.feasible)
            if not feasible:
                return False, None, t_used

            # 安全取值
            def safe_val(x):
                try:
                    return float(pyo.value(x))
                except Exception:
                    return 0.0

            sol = {
                'Pg_MW': [safe_val(m.Pg[g]) * self.baseMVA for g in m.GEN],
                'Qg_MVar': [safe_val(m.Qg[g]) * self.baseMVA for g in m.GEN],
                'Vm': [safe_val(m.Vm[i]) for i in m.BUS],
                'Va_deg': [safe_val(m.Va[i]) * 180.0 / math.pi for i in m.BUS]
            }
            return True, sol, t_used

        except Exception:
            return False, None, time.time() - t0

    def run_scan(self):
        """
        二维网格扫描：Pg(genA_bus) × Pg(genB_bus)
        - 打印每个可行解用时
        - 累计 100 个可行解时，打印前 100 个平均用时
        - 结束打印整体平均用时
        """
        print("开始扫描 ...")
        pA_vals_pu = np.linspace(self.pA_min_pu, self.pA_max_pu, self.n_points)
        pB_vals_pu = np.linspace(self.pB_min_pu, self.pB_max_pu, self.n_points)

        feasible_count = 0
        total_points = len(pA_vals_pu) * len(pB_vals_pu)
        processed = 0

        for pA in pA_vals_pu:
            for pB in pB_vals_pu:
                processed += 1
                ok, sol, tsec = self.solve_feasibility(pA, pB)
                if ok:
                    feasible_count += 1
                    self.feasible_times.append(tsec)
                    self.last_solution = sol  # 暖启动更新
                    print(f"✅ 可行解 #{feasible_count:>4}: "
                          f"Pg(bus{self.genA_bus})={pA*self.baseMVA:.2f} MW, "
                          f"Pg(bus{self.genB_bus})={pB*self.baseMVA:.2f} MW, "
                          f"用时={tsec:.3f}s")
                    if (not self.avg100_reported) and len(self.feasible_times) >= 100:
                        avg100 = sum(self.feasible_times[:100]) / 100.0
                        self.avg100_reported = True
                        print(f"🎯 前100个可行解平均求解时间：{avg100:.3f}s")
                # 进度
                if processed % max(1, total_points // 10) == 0 or processed == total_points:
                    prog = 100.0 * processed / total_points
                    print(f"进度: {processed}/{total_points} ({prog:.1f}%) | 已获可行解: {feasible_count}")

        if self.feasible_times:
            overall_avg = sum(self.feasible_times) / len(self.feasible_times)
            print(f"\n⏱️ 共获得 {len(self.feasible_times)} 个可行解；整体平均求解时间 = {overall_avg:.3f}s")
        else:
            print("\n未获得可行解。可考虑：调整扫描范围、减小 n_points、提高迭代上限或更换线性求解器。")

# ========== 运行示例 ==========
if __name__ == "__main__":
    # 默认：自动选择两台在运容量最大的机组（尽量不同母线且不选参考母线）
    # 扫描区间：各自 Pmax 的 40%~70%；每维 8 点（64 次求解）
    analyzer = ACOPF2736Analyzer(
        case_file='case2736sp.m',
        n_points=20,
        genA_bus=None,   # 如需指定：填母线号，例如 191
        genB_bus=None,   # 如需指定：填母线号，例如 119
        per_bus_frac=None,  # 例如 {191:(0.50,0.53), 119:(0.57,0.64)}
        warm_start=True,
        solver_opts={
            'max_iter': 600,
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
            'print_level': 0,
            'linear_solver': 'mumps'  # 可换为 'ma57' 或 'pardiso'（若可用）
            # 'hessian_approximation': 'limited-memory'  # 内存吃紧时可启用，速度可能变慢
        }
    )
    analyzer.run_scan()