# TJU-for-OPF
Nested Decoupling Adaptive Trajectory Unified Framework for AC Optimal Power Flow Method


## Abstract
In the alternating current optimal power flow(ACOPF) calculations, the selection of the initial point is crucial for both the computational speed of the solver and the precision of the solution. However, existing methodologies are often constrained by their dependence on the initial point, posing significant challenges in tackling nonlinear problems. To address this issue, this paper proposes a unified optimal power flow computation method based on the trajectory of dynamical systems, referred to as the Trajectory Unified Framework (TJU). This approach reformulates the power flow problem into a process of searching for stable equilibrium points of a nonlinear dynamical system defined by power flow equations, voltage constraints, and reactive power limitations. By employing the TJU framework, the solver can swiftly identify the solution trajectory without reliance on the specification of an initial point, thereby significantly enhancing computational efficiency and the quality of optimization results.

## 🚀 Installation

Install dependencies:
```bash
pip install -r requirements.txt
```
