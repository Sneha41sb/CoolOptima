"""
parameters.py
=============
Physical, thermal, hydraulic, and operational parameters for the direct-to-chip
liquid-cooled data center rack simulation.

Author: Senior ME Simulation Project
Reference Standards & Literature:
- ASHRAE TC 9.9 (2021): "Thermal Guidelines for Data Processing Environments", 5th Ed.
- Open Compute Project (OCP) Open Rack Standard V3 (ORv3) Liquid Cooling Specifications.
- Incropera, F. P., DeWitt, D. P., Bergman, T. L., & Lavine, A. S. (2007). "Fundamentals of Heat and Mass Transfer".
- Intel / AMD Enterprise Processor Thermal Design Guides (Sapphire Rapids / Genoa).
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class FluidProperties:
    """
    Thermodynamic and transport properties of the heat transfer fluid.
    Default fluid: Pure Deionized Water at nominal operating temperature (~35 °C / 308 K).
    """
    name: str = "Water (single-phase)"
    rho: float = 994.0          # Density [kg/m^3] at ~35 °C (Incropera Table A.6)
    cp: float = 4178.0          # Specific heat capacity [J/(kg*K)]
    k_fluid: float = 0.625      # Thermal conductivity [W/(m*K)]
    mu: float = 7.20e-4         # Dynamic viscosity [Pa*s = kg/(m*s)]
    prandtl: float = 4.81       # Prandtl number Pr = (cp * mu) / k_fluid [-]


@dataclass(frozen=True)
class ChipThermalParameters:
    """
    Thermal properties of the compute chips (silicon die + TIM + cold plate base).
    Based on high-performance enterprise server processor (e.g., Intel Xeon / AMD EPYC).
    """
    # Architecture
    num_servers_per_rack: int = 20       # Number of servers in the 42U rack (OCP ORv3 standard)
    chips_per_server: int = 2           # Dual-socket server architecture
    num_chips_total: int = 40           # Total cold plates / chips in rack = 20 * 2

    # Power per chip
    tdp_per_chip: float = 500.0         # Thermal Design Power (TDP) per chip [W]
    idle_power_per_chip: float = 100.0  # Idle power per chip [W] (approx 20% TDP)
    
    # Silicon die geometry & material (Silicon)
    die_length: float = 0.030           # 30 mm die width [m]
    die_width: float = 0.030            # 30 mm die length [m]
    die_thickness: float = 0.00075      # 750 um silicon wafer thickness [m]
    rho_si: float = 2330.0              # Density of silicon [kg/m^3]
    cp_si: float = 712.0                # Specific heat of silicon [J/(kg*K)]
    
    # Thermal Interface Material (TIM-1 / TIM-2)
    # Effective junction-to-cold-plate conduction resistance per chip
    # R_tim = t_tim / (k_tim * A_die) + contact resistances
    # High-performance phase change or indium sheet TIM: ~0.035 K/W (OCP spec)
    R_tim: float = 0.035                # Conductive thermal resistance [K/W] per chip
    
    # Cold plate base material (Copper C11000)
    cp_base_thickness: float = 0.003    # 3 mm copper base thickness [m]
    rho_cu: float = 8960.0              # Density of copper [kg/m^3]
    cp_cu: float = 385.0                # Specific heat of copper [J/(kg*K)]
    k_cu: float = 398.0                 # Thermal conductivity of copper [W/(m*K)]

    # Operating constraints
    T_j_max: float = 85.0               # Maximum allowable junction temperature [°C] (ASHRAE / Silicon reliability)
    T_j_target: float = 75.0            # Nominal target junction temperature [°C]


@dataclass(frozen=True)
class ColdPlateHydraulics:
    """
    Hydraulic and convective heat transfer parameters for microchannel cold plates.
    Reference: OCP Liquid Cooling Cold Plate Specification.
    """
    # Microchannel geometry
    channel_width: float = 0.0002       # 200 um microchannel width [m]
    fin_thickness: float = 0.0002       # 200 um copper fin thickness [m]
    channel_height: float = 0.002       # 2.0 mm fin height [m]
    channel_length: float = 0.030       # 30 mm flow length [m]
    num_channels: int = 75              # Number of parallel microchannels per cold plate
    
    # Convective heat transfer coefficient correlation:
    # h = h_base * (m_dot / m_dot_nominal)^alpha
    # In laminar microchannel flow (Re < 2000), Nu is nearly constant for fully developed,
    # but developing thermal length and manifold mixing give alpha ~ 0.5 - 0.7.
    # We use empirical microchannel fit: R_conv(m_dot) = 1 / (C_conv * m_dot^alpha)
    # At nominal 0.025 kg/s (1.5 L/min) per chip: R_conv ~ 0.040 K/W -> total R_th = 0.075 K/W
    m_dot_chip_nominal: float = 0.025   # Nominal mass flow rate per chip [kg/s] (~1.5 L/min)
    R_conv_nominal: float = 0.040       # Nominal convective thermal resistance per chip [K/W]
    alpha_conv: float = 0.65            # Power-law flow sensitivity exponent [-]

    # Flow limits per chip
    m_dot_chip_min: float = 0.005       # Minimum flow to prevent stagnation/boiling [kg/s] (0.3 L/min)
    m_dot_chip_max: float = 0.060       # Maximum flow to prevent erosion-corrosion [kg/s] (3.6 L/min)


@dataclass(frozen=True)
class HydraulicLoopParameters:
    """
    Hydraulic resistance and pump parameters for the secondary loop (TCS).
    Total pressure drop: Delta_p = k_flow * m_dot_rack^2
    Pump power: P_pump = (m_dot_rack * Delta_p) / (rho * eta_pump)
    """
    # Flow configuration: 40 chips in parallel across rack manifold
    # Rack pressure drop at nominal flow: Delta_p_nom ~ 75 kPa (cold plates + quick disconnects + manifold)
    # Rack nominal mass flow = 40 chips * 0.025 kg/s = 1.0 kg/s (~60 L/min)
    m_dot_rack_nominal: float = 1.0     # Nominal rack mass flow rate [kg/s]
    delta_p_nominal: float = 75000.0    # Nominal rack pressure drop [Pa] (75 kPa)
    
    # Hydraulic resistance coefficient: k_flow = Delta_p / m_dot^2 [Pa / (kg/s)^2]
    # Delta_p = k_flow * m_dot^2
    k_flow: float = 75000.0 / (1.0 ** 2) # 75,000 Pa/(kg/s)^2

    # Pump mechanical-electrical efficiency
    eta_pump: float = 0.65              # 65% combined wire-to-water efficiency (standard canned motor centrifugal pump)


@dataclass(frozen=True)
class CoolingPlantParameters:
    """
    Facility Water System (FWS) and chiller plant parameters.
    Reference: ASHRAE TC 9.9 5th Edition (Classes W17, W27, W32, W40, W45).
    """
    # ASHRAE Liquid Cooling Class limits for supply temperature T_c_in [°C]
    # W17: <= 17 °C | W27: <= 27 °C | W32: <= 32 °C | W40: <= 40 °C | W45: <= 45 °C
    T_c_in_min: float = 17.0            # Minimum allowable coolant supply temperature [°C]
    T_c_in_max: float = 45.0            # Maximum allowable coolant supply temperature [°C]
    T_c_in_nominal: float = 32.0        # Default ASHRAE W32 baseline supply temperature [°C]

    # CDU (Cooling Distribution Unit) Liquid-to-Liquid Heat Exchanger
    cdu_effectiveness: float = 0.85     # Plate heat exchanger thermal effectiveness [-]
    delta_T_cdu_approach: float = 3.0   # Minimum approach temperature difference [°C]

    # Chiller Plant Model
    # Real chiller COP = eta_carnot * COP_carnot
    # COP_carnot = T_evap_K / (T_cond_K - T_evap_K)
    # T_evap = T_c_in - delta_T_cdu - delta_T_evap
    # T_cond = T_ambient + delta_T_cond
    eta_carnot: float = 0.45            # 45% of theoretical Carnot COP (modern centrifugal chiller)
    delta_T_evap: float = 2.0           # Evaporator approach temperature [°C]
    delta_T_cond: float = 5.0           # Condenser approach temperature above ambient [°C]
    
    # Free cooling / economizer
    # When T_ambient <= T_c_in - delta_T_cdu_approach - 2.0 °C, chiller compressor can be bypassed!
    free_cooling_margin: float = 5.0    # Approach required for dry cooler / plate exchanger free cooling [°C]
    P_free_cooling_aux_kW: float = 0.25 # Auxiliary fan/spray power when in full free cooling mode [kW per rack]

    # Auxiliary data center overhead power (lighting, networking, UPS losses)
    P_aux_base_kW: float = 0.50         # Constant auxiliary power per rack [kW]


@dataclass
class SimulationConfig:
    """
    Complete consolidated simulation configuration.
    """
    fluid: FluidProperties = field(default_factory=FluidProperties)
    chips: ChipThermalParameters = field(default_factory=ChipThermalParameters)
    hydraulics: ColdPlateHydraulics = field(default_factory=ColdPlateHydraulics)
    loop: HydraulicLoopParameters = field(default_factory=HydraulicLoopParameters)
    plant: CoolingPlantParameters = field(default_factory=CoolingPlantParameters)

    # Derived rack quantities
    @property
    def P_IT_max(self) -> float:
        """Total maximum IT power for the rack [W]."""
        return self.chips.num_chips_total * self.chips.tdp_per_chip

    @property
    def P_IT_idle(self) -> float:
        """Total idle IT power for the rack [W]."""
        return self.chips.num_chips_total * self.chips.idle_power_per_chip

    @property
    def m_dot_rack_min(self) -> float:
        """Minimum allowable rack mass flow rate [kg/s]."""
        return self.chips.num_chips_total * self.hydraulics.m_dot_chip_min

    @property
    def m_dot_rack_max(self) -> float:
        """Maximum allowable rack mass flow rate [kg/s]."""
        return self.chips.num_chips_total * self.hydraulics.m_dot_chip_max

    # Thermal capacitance per chip [J/K]
    @property
    def C_chip(self) -> float:
        """
        Silicon die thermal capacitance [J/K].
        C = rho * V * cp = rho_si * (L * W * t) * cp_si
        """
        V_si = self.chips.die_length * self.chips.die_width * self.chips.die_thickness
        return self.chips.rho_si * V_si * self.chips.cp_si

    @property
    def C_coldplate(self) -> float:
        """
        Copper cold plate base thermal capacitance [J/K].
        C = rho_cu * (L * W * t_base) * cp_cu
        """
        V_cu = self.chips.die_length * self.chips.die_width * self.chips.cp_base_thickness
        return self.chips.rho_cu * V_cu * self.chips.cp_cu


# Instantiate a default global configuration instance
DEFAULT_CONFIG = SimulationConfig()

if __name__ == "__main__":
    cfg = DEFAULT_CONFIG
    print("=" * 70)
    print("DATA CENTER LIQUID COOLING SYSTEM PARAMETERS LOADED")
    print("=" * 70)
    print(f"Rack Architecture: {cfg.chips.num_servers_per_rack} servers, "
          f"{cfg.chips.chips_per_server} sockets/server = {cfg.chips.num_chips_total} chips")
    print(f"Rack IT Power: Idle = {cfg.P_IT_idle / 1000.0:.2f} kW | "
          f"Peak (TDP) = {cfg.P_IT_max / 1000.0:.2f} kW")
    print(f"Fluid: {cfg.fluid.name} (rho = {cfg.fluid.rho} kg/m^3, cp = {cfg.fluid.cp} J/(kg*K))")
    print(f"Thermal Capacitance: Silicon Die = {cfg.C_chip:.4f} J/K | "
          f"Copper Base = {cfg.C_coldplate:.4f} J/K")
    print(f"Flow Limits: Min = {cfg.m_dot_rack_min:.2f} kg/s ({cfg.m_dot_rack_min * 60 / (cfg.fluid.rho / 1000.0):.1f} L/min) | "
          f"Max = {cfg.m_dot_rack_max:.2f} kg/s ({cfg.m_dot_rack_max * 60 / (cfg.fluid.rho / 1000.0):.1f} L/min)")
    print(f"Nominal Pressure Drop: {cfg.loop.delta_p_nominal / 1000.0:.1f} kPa at {cfg.loop.m_dot_rack_nominal} kg/s")
    print("=" * 70)
