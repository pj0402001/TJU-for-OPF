# TJU-for-OPF

## Abstract
<p style="text-align: justify;">
The inherent non-convexity and disconnected feasible regions of the Alternating Current Optimal Power Flow (ACOPF) problem, exacerbated by the proliferation of high-dimensional saddle points, pose fundamental challenges to the robustness of conventional gradient-based solvers, which often suffer from extreme sensitivity to initialization and premature stagnation due to conflicting gradients. To address these systemic limitations, this paper proposes a Nested-Decoupling Adaptive Trajectory-Unified (NdaTJU) framework that reformulates the static optimization problem as a continuous trajectory evolution within a constructed nonlinear dynamical system. The core innovation lies in the introduction of a mathematically orthogonal separation between the energy dissipation mechanism (via decoupled weight decay) and the adaptive gradient driving force. Unlike traditional penalty methods where auxiliary terms interfere with the search direction, this structural decoupling effectively resolves intrinsic gradient conflicts, enabling the system state to navigate ill-conditioned manifolds and reliably escape the basins of attraction of saddle points. Theoretically, we establish a rigorous bijective mapping where the Stable Equilibrium Points (SEPs) of the dynamical system correspond strictly to the Karush-Kuhn-Tucker (KKT) points of the ACOPF, while the Unstable Equilibrium Points (UEPs) delineate the topological separatrices of the solution space. Extensive numerical simulations across a spectrum of benchmarks—ranging from systems exhibiting thermal-limit-induced fragmentation (LMBM3) and weak-coupling distortions (WB5) to large-scale networks (1888-bus)—demonstrate that NdaTJU achieves a deterministic 100\% convergence rate from arbitrary flat starts, confirming that the framework effectively regularizes the energy landscape and eliminates the reliance on high-quality warm-start initializations required by state-of-the-art solvers.

<p align="center">
  <img src="figures\diagram.png" alt="FSNet Diagram" width="800"/>
</p>

## 🚀 Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎓 Usage

```bash
python WB2.py \
```
