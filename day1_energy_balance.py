"""
day1_energy_balance.py
======================
Day 1: Steady-State Energy Balance, Thermal Resistor Network,
Pumping Power, Chiller COP, and PUE Analysis for Direct-to-Chip Liquid Cooling.

Author: Senior Mechanical Engineering Simulation Project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import parameters as p

# Load default physical and geometric parameters
cfg = p.DEFAULT_CONFIG


# =============================================================================
# PART 1: THERMAL RESISTANCE & TEMPERATURE CALCULATIONS
# =============================================================================

def compute_convective_resistance(m_dot_chip: float) -> float:
    """
    Calculates the convective thermal resistance R_conv [K/W] of a single microchannel cold plate.
    
    Physics:
    - Newton's Law of Cooling: Q_conv = h * A * (T_wall - T_fluid) = Delta_T / R_conv
    - As flow rate increases, boundary layer thins, increasing heat transfer coefficient h.
    - Empirical microchannel power law: R_conv(m_dot) = R_nom * (m_nom / m_dot)^alpha
    - alpha = 0.65 represents developing thermal boundary layers and entrance manifold mixing.
    """
    # Avoid division by zero with small lower bound epsilon = 1e-5 kg/s
    m_safe = np.maximum(m_dot_chip, 1e-5)
    ratio = cfg.hydraulics.m_dot_chip_nominal / m_safe
    return cfg.hydraulics.R_conv_nominal * (ratio ** cfg.hydraulics.alpha_conv)


def compute_steady_state_temperatures(P_chip: float, m_dot_chip: float, T_c_in: float) -> dict:
    """
    Solves the 1D steady-state thermal resistor network from silicon junction to coolant.
    
    Equations:
    1. Caloric fluid temperature rise:
       Delta_T_fluid = P_chip / (m_dot_chip * cp)
       T_c_out = T_c_in + Delta_T_fluid
       T_c_mean = T_c_in + 0.5 * Delta_T_fluid
       
    2. Convective jump from mean fluid to copper cold plate base:
       T_cp = T_c_mean + P_chip * R_conv(m_dot_chip)
       
    3. Conductive jump across Thermal Interface Material (TIM) to silicon junction:
       T_j_mean = T_cp + P_chip * R_tim
       
    4. Conservative hot-spot junction temperature directly above channel outlet:
       T_j_hotspot = T_c_out + P_chip * (R_conv + R_tim)
    """
    cp = cfg.fluid.cp
    m_safe = np.maximum(m_dot_chip, 1e-5)
    
    # 1st Law sensible heating of the coolant
    delta_T_fluid = P_chip / (m_safe * cp)
    T_c_out = T_c_in + delta_T_fluid
    T_c_mean = T_c_in + 0.5 * delta_T_fluid
    
    # Thermal resistances
    R_conv = compute_convective_resistance(m_safe)
    R_tim = cfg.chips.R_tim
    
    # Temperatures
    T_cp = T_c_mean + P_chip * R_conv
    T_j_mean = T_cp + P_chip * R_tim
    T_j_hotspot = T_c_out + P_chip * (R_conv + R_tim)
    
    # Total effective thermal resistance from junction to inlet fluid
    R_total = (T_j_hotspot - T_c_in) / np.maximum(P_chip, 1e-5)
    
    return {
        'T_c_out': T_c_out,
        'T_c_mean': T_c_mean,
        'T_cp': T_cp,
        'T_j_mean': T_j_mean,
        'T_j_hotspot': T_j_hotspot,
        'R_conv': R_conv,
        'R_tim': R_tim,
        'R_cal': delta_T_fluid / np.maximum(P_chip, 1e-5),
        'R_total': R_total,
        'delta_T_fluid': delta_T_fluid
    }


# =============================================================================
# PART 2: HYDRAULIC PRESSURE DROP & ELECTRICAL PUMP POWER
# =============================================================================

def compute_pump_power(m_dot_rack: float) -> tuple[float, float]:
    """
    Calculates hydraulic pressure drop and electrical pump power across the 40-server rack loop.
    
    Physics:
    - Darcy-Weisbach head loss: Delta_p = k_flow * m_dot_rack^2  [Pa]
    - Hydraulic power delivered to fluid: W_hyd = V_dot * Delta_p = (m_dot / rho) * Delta_p  [W]
    - Electrical power consumed by pump motor: P_pump = W_hyd / eta_pump
      => P_pump = (k_flow * m_dot_rack^3) / (rho * eta_pump)  [W] (Cubic scaling!)
    """
    k = cfg.loop.k_flow
    rho = cfg.fluid.rho
    eta = cfg.loop.eta_pump
    
    delta_p = k * (m_dot_rack ** 2)
    P_pump = (k * (m_dot_rack ** 3)) / (rho * eta)
    return P_pump, delta_p


# =============================================================================
# PART 3: CHILLER PLANT THERMODYNAMICS & COP
# =============================================================================

def compute_chiller_cop(T_c_in: float, T_ambient: float) -> tuple[float, bool]:
    """
    Calculates chiller Coefficient of Performance (COP) based on Carnot cycle
    and centrifugal compressor efficiency.
    
    Thermodynamics:
    - Carnot COP: COP_carnot = T_evap_K / (T_cond_K - T_evap_K)
    - T_evap = T_c_in - Delta_T_cdu - Delta_T_evap
    - T_cond = T_ambient + Delta_T_cond
    - Real COP = eta_carnot * COP_carnot
    
    Free Cooling (Economizer):
    - If T_ambient <= T_c_in - Delta_T_cdu - 5.0 C, chiller compressor is bypassed!
      Only cooling tower / dry cooler fans operate (COP ~ 30).
    """
    plant = cfg.plant
    
    # Check economizer free cooling condition
    if T_ambient <= (T_c_in - plant.delta_T_cdu_approach - plant.free_cooling_margin):
        return 30.0, True  # Free cooling active
        
    T_evap_C = T_c_in - plant.delta_T_cdu_approach - plant.delta_T_evap
    T_cond_C = T_ambient + plant.delta_T_cond
    
    T_evap_K = T_evap_C + 273.15
    T_cond_K = T_cond_C + 273.15
    
    lift = T_cond_K - T_evap_K
    if lift < 1.0:
        lift = 1.0
        
    cop_carnot = T_evap_K / lift
    cop_real = plant.eta_carnot * cop_carnot
    cop_real = np.clip(cop_real, 1.5, 12.0)
    return float(cop_real), False


# =============================================================================
# PART 4: FACILITY POWER USAGE EFFECTIVENESS (PUE)
# =============================================================================

def compute_facility_pue(P_IT: float, m_dot_rack: float, T_c_in: float, T_ambient: float) -> dict:
    """
    Calculates total facility power breakdown and Power Usage Effectiveness (PUE).
    
    P_total = P_IT + P_pump + P_chiller + P_aux
    PUE = P_total / P_IT = 1 + (P_pump + P_chiller + P_aux) / P_IT
    """
    P_pump, delta_p = compute_pump_power(m_dot_rack)
    cop, is_free_cooling = compute_chiller_cop(T_c_in, T_ambient)
    
    # Heat rejected to outdoor environment = IT heat + pump viscous dissipation
    Q_rejected = P_IT + P_pump
    
    if is_free_cooling:
        P_chiller = cfg.plant.P_free_cooling_aux_kW * 1000.0
    else:
        P_chiller = Q_rejected / cop
        
    P_aux = cfg.plant.P_aux_base_kW * 1000.0  # Networking, lighting, PDU losses
    P_cooling = P_pump + P_chiller
    P_total = P_IT + P_cooling + P_aux
    
    pue = P_total / np.maximum(P_IT, 1e-3)
    
    return {
        'P_IT': P_IT,
        'P_pump': P_pump,
        'P_chiller': P_chiller,
        'P_cooling': P_cooling,
        'P_aux': P_aux,
        'P_total': P_total,
        'PUE': pue,
        'COP': cop,
        'is_free_cooling': is_free_cooling,
        'delta_p_kPa': delta_p / 1000.0
    }


# =============================================================================
# PART 5: OPTIMAL MINIMUM FLOW RATE SOLVER
# =============================================================================

def find_minimum_flow_rate(P_chip: float, T_c_in: float, T_j_target: float = 75.0) -> float:
    """
    Uses Brent's root-finding algorithm to find the exact minimum mass flow rate [kg/s]
    that keeps T_j_hotspot <= T_j_target.
    """
    m_min = cfg.hydraulics.m_dot_chip_min
    m_max = cfg.hydraulics.m_dot_chip_max * 2.5
    
    def residual(m):
        res = compute_steady_state_temperatures(P_chip, m, T_c_in)
        return res['T_j_hotspot'] - T_j_target
        
    if residual(m_max) > 0:
        return np.nan  # Infeasible target
    if residual(m_min) <= 0:
        return m_min   # Already cool enough at minimum flow
        
    return brentq(residual, m_min, m_max)


# =============================================================================
# PART 6: FIRST LAW ENERGY BALANCE CLOSURE VERIFICATION
# =============================================================================

def verify_first_law_closure():
    """
    Verifies that Energy In equals Energy Out at steady state.
    P_IT must equal m_dot_rack * cp * (T_c_out - T_c_in) within machine precision.
    """
    print('=' * 75)
    print('VERIFICATION: FIRST LAW ENERGY BALANCE CLOSURE')
    print('=' * 75)
    
    test_powers = [4000.0, 10000.0, 16000.0, 20000.0]  # [W] (20% to 100% rack load)
    test_flows = [0.4, 0.8, 1.2, 1.6]                  # [kg/s] rack flow
    T_in = 32.0                                         # [C] (ASHRAE W32)
    
    max_error = 0.0
    for P_rack in test_powers:
        P_chip = P_rack / cfg.chips.num_chips_total
        for m_rack in test_flows:
            m_chip = m_rack / cfg.chips.num_chips_total
            res = compute_steady_state_temperatures(P_chip, m_chip, T_in)
            
            # Heat absorbed by fluid across all 40 parallel cold plates
            Q_fluid = m_rack * cfg.fluid.cp * (res['T_c_out'] - T_in)
            err = abs(P_rack - Q_fluid)
            max_error = max(max_error, err)
            
    print(f'Tested {len(test_powers) * len(test_flows)} combinations across full power and flow ranges.')
    print(f'Maximum Absolute Energy Residual: {max_error:.2e} Watts')
    rel_error = max_error / 20000.0
    print(f'Relative Energy Balance Residual: {rel_error:.2e} (IEEE 754 machine precision limit: ~1e-16)')
    if rel_error < 1e-12:
        print('[SUCCESS] First Law Energy Balance closes with machine precision (relative error < 1e-12)!')
    else:
        print('[FAILURE] Energy balance violated!')
    print('=' * 75)


# =============================================================================
# PART 7: GENERATE VALIDATION & TRADE-OFF PLOTS
# =============================================================================

def generate_tradeoff_plots():
    """
    Generates 4 publication-quality engineering plots showing the fundamental
    thermodynamic, hydraulic, and facility trade-offs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Day 1: Steady-State Thermal, Hydraulic & Energy Balances (Data Center Rack)',
                 fontsize=14, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Plot 1: Junction Temperature vs Coolant Flow per Chip
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    m_chip_array = np.linspace(0.005, 0.055, 100)  # kg/s
    liters_min = m_chip_array * 60.0 / (cfg.fluid.rho / 1000.0)
    powers_chip = [200.0, 350.0, 500.0]            # W/chip (40%, 70%, 100% TDP)
    T_c_in = 32.0                                  # ASHRAE W32
    
    for P in powers_chip:
        T_j_vals = [compute_steady_state_temperatures(P, m, T_c_in)['T_j_hotspot'] for m in m_chip_array]
        ax1.plot(liters_min, T_j_vals, lw=2.2, label=f'P = {P:.0f} W/chip ({P*40/1000:.1f} kW rack)')
        
    ax1.axhline(cfg.chips.T_j_max, color='red', ls='--', lw=1.8, label=r'Silicon Limit {j,max} = 85^\circ')
    ax1.axhline(cfg.chips.T_j_target, color='green', ls=':', lw=1.5, label=r'Target Setpoint {j,target} = 75^\circ')
    ax1.set_xlabel('Coolant Flow Rate per Cold Plate [L/min]')
    ax1.set_ylabel(r'Chip Hotspot Temperature $ [$^\circ]')
    ax1.set_title(r'1. Thermal Resistance: $ vs Coolant Flow Rate')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)
    
    # -------------------------------------------------------------------------
    # Plot 2: The Optimization Dilemma (Pump Power vs Thermal Resistance)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    m_rack_array = np.linspace(0.2, 2.0, 100)      # kg/s rack total
    p_pump_vals = [compute_pump_power(m)[0] for m in m_rack_array]
    r_conv_vals = [compute_convective_resistance(m / 40.0) for m in m_rack_array]
    
    color1 = 'tab:blue'
    ax2.set_xlabel(r'Total Rack Mass Flow Rate $\dot{m}_{rack}$ [kg/s]')
    ax2.set_ylabel(r'Pump Power [W] ($\propto \dot{m}^3$)', color=color1)
    ax2.plot(m_rack_array, p_pump_vals, color=color1, lw=2.5, label='Pump Power [W]')
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    color2 = 'tab:orange'
    ax2_twin.set_ylabel(r'Convective Resistance {conv}$ [K/W]', color=color2)
    ax2_twin.plot(m_rack_array, r_conv_vals, color=color2, lw=2.5, ls='-.', label=r'{conv}$ [K/W]')
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    ax2.set_title('2. The Dilemma: Diminishing Thermal Gain vs Cubic Pump Cost')
    
    # -------------------------------------------------------------------------
    # Plot 3: Chiller COP vs Coolant Supply Temperature (ASHRAE Classes)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    T_supply_array = np.linspace(17.0, 45.0, 100)
    ambients = [25.0, 32.0, 38.0]  # Mild, nominal, peak Mumbai ambient [C]
    
    for T_amb in ambients:
        cop_vals = [compute_chiller_cop(Ts, T_amb)[0] for Ts in T_supply_array]
        ax3.plot(T_supply_array, cop_vals, lw=2.2, label=rf'{{ambient}} = {T_amb}^\circ')
        
    ax3.set_xlabel(r'Coolant Supply Temp {c,in}$ [$^\circ] (ASHRAE Classes)')
    ax3.set_ylabel('Chiller Coefficient of Performance (COP) [-]')
    ax3.set_title(r'3. Chiller COP vs Supply Temp in Mumbai Weather')
    ax3.axvline(17, color='gray', ls=':', label='W17 (17°C)')
    ax3.axvline(27, color='gray', ls=':', label='W27 (27°C)')
    ax3.axvline(32, color='gray', ls=':', label='W32 (32°C)')
    ax3.axvline(45, color='gray', ls=':', label='W45 (45°C)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc='upper left')
    
    # -------------------------------------------------------------------------
    # Plot 4: Facility PUE vs Server Workload (Fixed Baseline vs Flow-Matched)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    workloads = np.linspace(0.10, 1.00, 20)
    P_IT_array = workloads * cfg.P_IT_max
    T_amb_mumbai = 35.0  # 35 C daytime ambient
    T_c_in_base = 25.0   # 25 C supply
    
    pue_baseline = []
    pue_flow_matched = []
    
    for P_it in P_IT_array:
        # 1. Baseline: Constant max flow (1.5 kg/s) at all times
        res_base = compute_facility_pue(P_it, 1.5, T_c_in_base, T_amb_mumbai)
        pue_baseline.append(res_base['PUE'])
        
        # 2. Flow-matched: Modulate flow rate to satisfy T_j = 75 C setpoint
        P_chip = P_it / cfg.chips.num_chips_total
        m_chip_opt = find_minimum_flow_rate(P_chip, T_c_in_base, 75.0)
        m_rack_opt = m_chip_opt * cfg.chips.num_chips_total
        res_opt = compute_facility_pue(P_it, m_rack_opt, T_c_in_base, T_amb_mumbai)
        pue_flow_matched.append(res_opt['PUE'])
        
    ax4.plot(workloads * 100, pue_baseline, 'r--o', lw=2.2, label='Fixed High Flow Baseline (1.5 kg/s)')
    ax4.plot(workloads * 100, pue_flow_matched, 'g-s', lw=2.2, label=r'Flow-Matched Cooling ( = 75^\circ)')
    ax4.set_xlabel('Server Workload Utilization [%]')
    ax4.set_ylabel('Power Usage Effectiveness (PUE) [-]')
    ax4.set_title(r'4. PUE Impact of Flow Control ({amb} = 35^\circ, {c,in} = 25^\circ)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8.5)
    
    plt.tight_layout()
    plot_path = '/home/nyx/data/day1_steady_state_tradeoffs.png'
    plt.savefig(plot_path, dpi=300)
    print(f'[SUCCESS] Generated trade-off plots saved to: {plot_path}')


if __name__ == '__main__':
    verify_first_law_closure()
    generate_tradeoff_plots()