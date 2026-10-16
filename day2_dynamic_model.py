"""
day2_dynamic_model.py
=====================
Day 2: Microchannel Fluid Mechanics, Conjugate Heat Transfer,
Thermal Capacitance Network, and Dynamic ODE Simulation for Direct-to-Chip Cooling.

Author: Senior Mechanical Engineering Simulation Project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import parameters as p

cfg = p.DEFAULT_CONFIG


# =============================================================================
# PART 1: MICROCHANNEL FLUID MECHANICS & FRICTION
# =============================================================================

def compute_microchannel_hydraulics(m_dot_chip: float) -> dict:
    """
    Calculates detailed laminar microchannel fluid dynamics:
    - Hydraulic diameter: Dh = 4 * Ac / P_wetted = 2*w*h / (w+h)
    - Reynolds number: Re = (rho * v * Dh) / mu
    - Friction factor: Shah-London laminar rectangular duct correlation
    - Hydrodynamic and thermal entrance lengths
    """
    w = cfg.hydraulics.channel_width      # 200 um = 0.0002 m
    h = cfg.hydraulics.channel_height     # 2 mm = 0.002 m
    L = cfg.hydraulics.channel_length     # 30 mm = 0.030 m
    N = cfg.hydraulics.num_channels       # 75 channels
    
    Ac_channel = w * h
    P_wetted = 2.0 * (w + h)
    Dh = 4.0 * Ac_channel / P_wetted      # [m]
    
    rho = cfg.fluid.rho
    mu = cfg.fluid.mu
    cp = cfg.fluid.cp
    kf = cfg.fluid.k_fluid
    
    # Fluid velocity inside channels
    m_safe = np.maximum(m_dot_chip, 1e-5)
    V_dot_total = m_safe / rho
    v_channel = V_dot_total / (N * Ac_channel)  # [m/s]
    
    # Reynolds & Prandtl numbers
    Re = (rho * v_channel * Dh) / mu
    Pr = (cp * mu) / kf
    
    # Channel aspect ratio alpha_c = w / h (<= 1)
    alpha_c = w / h
    
    # Shah & London (1978) fully developed laminar friction factor for rectangular ducts:
    # f * Re = 24 * (1 - 1.3553*alpha_c + 1.9467*alpha_c^2 - 1.7012*alpha_c^3 + 0.9564*alpha_c^4 - 0.2537*alpha_c^5)
    f_Re = 24.0 * (1.0 - 1.3553 * alpha_c + 1.9467 * (alpha_c ** 2) - 1.7012 * (alpha_c ** 3) 
                   + 0.9564 * (alpha_c ** 4) - 0.2537 * (alpha_c ** 5))
    f_darcy = f_Re / np.maximum(Re, 1e-3)
    
    # Hydrodynamic and thermal entrance lengths
    x_fd_h = 0.05 * Re * Dh         # [m]
    x_fd_t = 0.05 * Re * Pr * Dh    # [m]
    
    # Total channel fluid mass & thermal capacitance
    V_fluid = N * Ac_channel * L
    M_fluid = rho * V_fluid
    C_fluid = M_fluid * cp
    
    return {
        'Dh_um': Dh * 1e6,
        'v_channel_m_s': v_channel,
        'Re': Re,
        'Pr': Pr,
        'f_darcy': f_darcy,
        'x_fd_h_mm': x_fd_h * 1000.0,
        'x_fd_t_mm': x_fd_t * 1000.0,
        'is_thermally_developing': x_fd_t > L,
        'M_fluid_g': M_fluid * 1000.0,
        'C_fluid': C_fluid
    }


# =============================================================================
# PART 2: DYNAMIC CONTINUOUS ODE SYSTEM
# =============================================================================

class DirectToChipDynamicModel:
    """
    3-State Lumped Thermal Capacitance Model of a Direct-to-Chip Liquid Cooling System.
    
    State Vector x = [T_die, T_cp, T_c_out]^T
    
    Derivation from First Law of Thermodynamics:
    1. Silicon Die:
       C_die * dT_die/dt = P_chip(t) - (T_die - T_cp) / R_tim
       
    2. Copper Cold Plate Base:
       C_cp * dT_cp/dt = (T_die - T_cp) / R_tim - (T_cp - T_c_mean) / R_conv(m_dot)
       where T_c_mean = 0.5 * (T_c_in + T_c_out)
       
    3. Coolant Fluid in Microchannels:
       C_fluid * dT_c_out/dt = (T_cp - T_c_mean) / R_conv(m_dot) - m_dot * cp * (T_c_out - T_c_in)
    """
    
    def __init__(self, cfg=p.DEFAULT_CONFIG):
        self.cfg = cfg
        self.C_die = cfg.C_chip
        self.C_cp = cfg.C_coldplate
        
        # Microchannel fluid mass and capacitance
        hyd = compute_microchannel_hydraulics(cfg.hydraulics.m_dot_chip_nominal)
        self.C_fluid = hyd['C_fluid']
        self.R_tim = cfg.chips.R_tim
        self.cp = cfg.fluid.cp
        
    def convective_resistance(self, m_dot_chip: float) -> float:
        m_safe = np.maximum(m_dot_chip, 1e-5)
        ratio = self.cfg.hydraulics.m_dot_chip_nominal / m_safe
        return self.cfg.hydraulics.R_conv_nominal * (ratio ** self.cfg.hydraulics.alpha_conv)
        
    def ode_derivatives(self, t: float, x: np.ndarray, 
                        P_chip_func, m_dot_func, T_c_in_func) -> list:
        """
        Evaluates dx/dt = f(x, u, w, t) for ODE numerical integration.
        """
        T_die, T_cp, T_cout = x
        
        # Time-dependent inputs
        P_chip = P_chip_func(t)
        m_dot = m_dot_func(t)
        T_c_in = T_c_in_func(t)
        
        # Arithmetic mean fluid temperature inside microchannel
        T_cmean = 0.5 * (T_c_in + T_cout)
        R_conv = self.convective_resistance(m_dot)
        
        # Heat transfer rates [W]
        Q_cond = (T_die - T_cp) / self.R_tim
        Q_conv = (T_cp - T_cmean) / R_conv
        Q_advect = m_dot * self.cp * (T_cout - T_c_in)
        
        # State derivatives
        dTdie_dt = (P_chip - Q_cond) / self.C_die
        dTcp_dt = (Q_cond - Q_conv) / self.C_cp
        dTcout_dt = (Q_conv - Q_advect) / self.C_fluid
        
        return [dTdie_dt, dTcp_dt, dTcout_dt]

    def compute_hotspot_junction_temperature(self, T_die: float, T_cout: float, 
                                             P_chip: float, m_dot_chip: float) -> float:
        """
        Worst-case local silicon junction hotspot temperature above the channel exit:
        T_j_hotspot = T_cout + P_chip * (R_conv + R_tim)
        """
        R_conv = self.convective_resistance(m_dot_chip)
        return T_cout + P_chip * (R_conv + self.R_tim)

    def simulate(self, t_span: tuple, x0: list, 
                 P_chip_func, m_dot_func, T_c_in_func, t_eval=None):
        """
        Integrates the dynamic ODE system using Radau stiff solver.
        """
        rhs = lambda t, x: self.ode_derivatives(t, x, P_chip_func, m_dot_func, T_c_in_func)
        sol = solve_ivp(rhs, t_span, x0, method='Radau', t_eval=t_eval, rtol=1e-6, atol=1e-8)
        return sol


# =============================================================================
# PART 3: TRANSIENT SIMULATION EXPERIMENTS & VALIDATION
# =============================================================================

def run_transient_experiments():
    model = DirectToChipDynamicModel()
    
    print('=' * 75)
    print('DAY 2: DYNAMIC ODE MODEL EXECUTION & EXPERIMENTS')
    print('=' * 75)
    
    # -------------------------------------------------------------------------
    # Experiment 1: Workload Step Change (200 W -> 500 W TDP)
    # -------------------------------------------------------------------------
    print('1. Simulating Workload Step (200 W -> 500 W at t=2.0s, fixed flow = 1.5 L/min)...')
    m_fixed = 0.025  # kg/s nominal
    T_in_fixed = 32.0 # C
    
    def P_step(t):
        return 200.0 if t < 2.0 else 500.0
        
    def m_step(t):
        return m_fixed
        
    def T_in_step(t):
        return T_in_fixed
        
    # Start at steady state of 200 W
    R_conv_nom = model.convective_resistance(m_fixed)
    dT_fluid_init = 200.0 / (m_fixed * model.cp)
    T_cout_init = T_in_fixed + dT_fluid_init
    T_cmean_init = T_in_fixed + 0.5 * dT_fluid_init
    T_cp_init = T_cmean_init + 200.0 * R_conv_nom
    T_die_init = T_cp_init + 200.0 * model.R_tim
    x0 = [T_die_init, T_cp_init, T_cout_init]
    
    t_eval = np.linspace(0.0, 8.0, 800)
    sol1 = model.simulate((0.0, 8.0), x0, P_step, m_step, T_in_step, t_eval=t_eval)
    
    # Calculate hotspot temperature trace
    T_hotspot1 = []
    for i, t in enumerate(sol1.t):
        P_val = P_step(t)
        T_die_val = sol1.y[0, i]
        T_cout_val = sol1.y[2, i]
        T_h = model.compute_hotspot_junction_temperature(T_die_val, T_cout_val, P_val, m_fixed)
        T_hotspot1.append(T_h)
    T_hotspot1 = np.array(T_hotspot1)
    
    # -------------------------------------------------------------------------
    # Experiment 2: Flow Step Change (0.010 -> 0.035 kg/s at fixed 500 W load)
    # -------------------------------------------------------------------------
    print('2. Simulating Flow Step (0.010 -> 0.035 kg/s at t=2.0s, fixed 500 W load)...')
    def P_const(t):
        return 500.0
        
    def m_flow_step(t):
        return 0.010 if t < 2.0 else 0.035
        
    # Initial steady state at 500 W with 0.010 kg/s flow
    R_conv_low = model.convective_resistance(0.010)
    dT_fluid_low = 500.0 / (0.010 * model.cp)
    x0_flow = [
        T_in_fixed + 0.5*dT_fluid_low + 500.0*R_conv_low + 500.0*model.R_tim,
        T_in_fixed + 0.5*dT_fluid_low + 500.0*R_conv_low,
        T_in_fixed + dT_fluid_low
    ]
    sol2 = model.simulate((0.0, 8.0), x0_flow, P_const, m_flow_step, T_in_step, t_eval=t_eval)
    
    T_hotspot2 = []
    for i, t in enumerate(sol2.t):
        m_val = m_flow_step(t)
        T_die_val = sol2.y[0, i]
        T_cout_val = sol2.y[2, i]
        T_h = model.compute_hotspot_junction_temperature(T_die_val, T_cout_val, 500.0, m_val)
        T_hotspot2.append(T_h)
    T_hotspot2 = np.array(T_hotspot2)

    # -------------------------------------------------------------------------
    # Experiment 3: Energy Conservation Closure Across Transient Heating
    # -------------------------------------------------------------------------
    print('3. Verifying Transient First Law Energy Balance Closure...')
    # Integrate: Integral(P_chip) dt vs Integral(Q_out) dt + Delta_E_stored
    dt_arr = np.diff(sol1.t)
    P_vals = np.array([P_step(t) for t in sol1.t])
    P_mid = 0.5 * (P_vals[:-1] + P_vals[1:])
    E_in_total = np.sum(P_mid * dt_arr)
    
    Q_out_arr = m_fixed * model.cp * (sol1.y[2, :] - T_in_fixed)
    Q_out_mid = 0.5 * (Q_out_arr[:-1] + Q_out_arr[1:])
    E_out_total = np.sum(Q_out_mid * dt_arr)
    
    Delta_E_die = model.C_die * (sol1.y[0, -1] - sol1.y[0, 0])
    Delta_E_cp = model.C_cp * (sol1.y[1, -1] - sol1.y[1, 0])
    Delta_E_fluid = model.C_fluid * (sol1.y[2, -1] - sol1.y[2, 0])
    Delta_E_stored = Delta_E_die + Delta_E_cp + Delta_E_fluid
    
    energy_residual = abs(E_in_total - (E_out_total + Delta_E_stored))
    rel_energy_err = energy_residual / E_in_total
    
    print(f'   Total Heat Input (E_in):    {E_in_total:.3f} Joules')
    print(f'   Total Heat Removed (E_out): {E_out_total:.3f} Joules')
    print(f'   Internal Energy Stored (dE): {Delta_E_stored:.3f} Joules')
    print(f'   Transient Energy Residual:  {energy_residual:.3e} Joules (Relative Error: {rel_energy_err:.2e})')
    if rel_energy_err < 1e-3:
        print('   [SUCCESS] Transient First Law Energy Balance verified (< 0.1% integration error)!')
    else:
        print('   [WARNING] High energy balance error!')
    print('=' * 75)
    
    # -------------------------------------------------------------------------
    # Generate Publication Plots
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Day 2: Conjugate Heat Transfer, Microchannels & Dynamic ODE Model',
                 fontsize=14, fontweight='bold')
                 
    # Plot 1: Workload Step Response
    ax1 = axes[0, 0]
    ax1.plot(sol1.t, sol1.y[0, :], 'r-', lw=2, label=r'Die Temperature {die}(t)$')
    ax1.plot(sol1.t, sol1.y[1, :], 'b--', lw=2, label=r'Cold Plate Base {cp}(t)$')
    ax1.plot(sol1.t, sol1.y[2, :], 'g-.', lw=2, label=r'Fluid Outlet {c,out}(t)$')
    ax1.plot(sol1.t, T_hotspot1, 'm:', lw=2.5, label=r'Hotspot Junction {j,hotspot}(t)$')
    ax1.axhline(85.0, color='red', ls='--', lw=1.2, label=r'Limit 5^\circ')
    ax1.axvline(2.0, color='gray', ls=':', label='Workload Step 200W->500W')
    ax1.set_xlabel('Time [seconds]')
    ax1.set_ylabel(r'Temperature [$^\circ]')
    ax1.set_title(r'1. Dynamic Response to Workload Step ($\dot{m} = 1.5$ L/min)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5, loc='upper left')
    
    # Plot 2: Flow Step Response (Active Cooling Modulation)
    ax2 = axes[0, 1]
    ax2.plot(sol2.t, sol2.y[0, :], 'r-', lw=2, label=r'{die}(t)$')
    ax2.plot(sol2.t, sol2.y[1, :], 'b--', lw=2, label=r'{cp}(t)$')
    ax2.plot(sol2.t, sol2.y[2, :], 'g-.', lw=2, label=r'{c,out}(t)$')
    ax2.plot(sol2.t, T_hotspot2, 'm:', lw=2.5, label=r'{j,hotspot}(t)$')
    ax2.axvline(2.0, color='gray', ls=':', label='Flow Step 0.6 -> 2.1 L/min')
    ax2.set_xlabel('Time [seconds]')
    ax2.set_ylabel(r'Temperature [$^\circ]')
    ax2.set_title(r'2. Dynamic Response to Flow Step (P = 500W TDP)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5, loc='upper right')
    
    # Plot 3: Hydrodynamic vs Thermal Developing Lengths in Microchannels
    ax3 = axes[1, 0]
    flow_rates_L_min = np.linspace(0.3, 3.5, 50)
    flows_kg_s = flow_rates_L_min * (cfg.fluid.rho / 1000.0) / 60.0
    x_fd_h_list = []
    x_fd_t_list = []
    for m in flows_kg_s:
        hyd = compute_microchannel_hydraulics(m)
        x_fd_h_list.append(hyd['x_fd_h_mm'])
        x_fd_t_list.append(hyd['x_fd_t_mm'])
        
    ax3.plot(flow_rates_L_min, x_fd_h_list, 'b-', lw=2.2, label=r'Hydrodynamic Entry Length {fd,h}$')
    ax3.plot(flow_rates_L_min, x_fd_t_list, 'r--', lw=2.2, label=r'Thermal Entry Length {fd,t}$')
    ax3.axhline(30.0, color='black', ls='-.', lw=1.8, label=r'Physical Channel Length  = 30$ mm')
    ax3.set_xlabel('Coolant Flow Rate per Cold Plate [L/min]')
    ax3.set_ylabel('Entrance Length [mm]')
    ax3.set_title('3. Microchannel Physics: Developing Graetz Flow Regime')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8.5)
    
    # Plot 4: Transient Energy Accumulation
    ax4 = axes[1, 1]
    cum_Ein = np.cumsum(P_mid * dt_arr)
    cum_Eout = np.cumsum(Q_out_mid * dt_arr)
    time_mid = sol1.t[:-1]
    
    ax4.plot(time_mid, cum_Ein, 'r-', lw=2.2, label=r'Cumulative Heat Input $\int P_{chip} dt$')
    ax4.plot(time_mid, cum_Eout, 'g--', lw=2.2, label=r'Cumulative Heat Removed $\int Q_{fluid} dt$')
    ax4.fill_between(time_mid, cum_Ein, cum_Eout, color='orange', alpha=0.25, 
                     label=r'Stored Thermal Energy $\Delta E_{stored}$')
    ax4.set_xlabel('Time [seconds]')
    ax4.set_ylabel('Energy [Joules]')
    ax4.set_title('4. Transient Energy Storage: The Thermal Capacitor Effect')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8.5, loc='upper left')
    
    plt.tight_layout()
    plot_path = '/home/nyx/data/day2_transient_dynamics.png'
    plt.savefig(plot_path, dpi=300)
    print(f'[SUCCESS] Day 2 transient dynamics plot saved to: {plot_path}')


if __name__ == '__main__':
    run_transient_experiments()