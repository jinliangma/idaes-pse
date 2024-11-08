###############################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
###############################################################################

# desorption step simulation using Joule heating
# wall boundary set as almost adiabatic

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    TransformationFactory,
    SolverFactory,
    Var,
    Param,
    value,
    units as pyunits,
)
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent
from idaes.models.unit_models import ValveFunctionType, Valve
from idaes.core import FlowsheetBlock, EnergyBalanceType
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.dyn_utils import copy_values_at_time, copy_non_time_indexed_values
from idaes.core.util.initialization import initialize_by_time_element, propagate_state
import idaes.logger as idaeslog
import logging

from idaes.models_extra.tvsa_1d.unit_models.fixed_bed_1D import FixedBed1D

from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models_extra.power_generation.properties.natural_gas_PR import (
    EosType,
    get_prop,
)

__author__ = "Jinliang Ma"


def get_model(dynamic=True, time_set=None, nstep=None, init=True):
    m = ConcreteModel()
    m.dynamic = dynamic
    if time_set is None:
        time_set = [0,990]
    if nstep is None:
        nstep = 33
    if m.dynamic:
        m.fs = FlowsheetBlock(
            dynamic=True, time_set=time_set, time_units=pyunits.s
        )
    else:
        m.fs = FlowsheetBlock(dynamic=False)

    gas_species = {"CO2", "H2O", "N2"}
    # modify the bounds of pressure, default lower bound is 5e4
    configuration = get_prop(gas_species, ["Vap"], EosType.IDEAL)
    pres_bounds = (1e4, 1e5, 1e6, pyunits.Pa)
    configuration["state_bounds"]["pressure"] = pres_bounds
    m.fs.gas_properties = GenericParameterBlock(
	    **configuration,
	    doc="gas property",
    )

    m.fs.gas_properties.set_default_scaling("enth_mol_phase", 1e-3)
    m.fs.gas_properties.set_default_scaling("pressure", 1e-5)
    m.fs.gas_properties.set_default_scaling("temperature", 1e-2)
    m.fs.gas_properties.set_default_scaling("flow_mol", 1e1)
    m.fs.gas_properties.set_default_scaling("flow_mol_phase", 1e1)
    m.fs.gas_properties.set_default_scaling("_energy_density_term", 1e-4)

    _mf_scale = {
        "CO2": 50,
        "H2O": 10,
        "N2": 1,
    }
    for comp, s in _mf_scale.items():
        m.fs.gas_properties.set_default_scaling("mole_frac_comp", s, index=comp)
        m.fs.gas_properties.set_default_scaling("mole_frac_phase_comp", s, index=("Vap", comp))
        m.fs.gas_properties.set_default_scaling("flow_mol_phase_comp", s * 1e1, index=("Vap", comp))

    nxfe = 20
    x_nfe_list = [0,1]
    m.fs.FB = FixedBed1D(
        dynamic=dynamic,
        finite_elements=nxfe,
        length_domain_set=x_nfe_list,
        transformation_method="dae.finite_difference",
        energy_balance_type=EnergyBalanceType.enthalpyTotal,
        pressure_drop_type="ergun_correlation",
        property_package=m.fs.gas_properties,
        adsorbent="Lewatit",
        coadsorption_isotherm="Mechanistic", #"None", "Stampi-Bombelli", #"WADST","Mechanistic"
        adsorbent_shape="particle",
        has_joule_heating=True,
    )

    m.fs.Inlet_Valve = Valve(
        dynamic=False,
        valve_function_callback= ValveFunctionType.linear,
        property_package=m.fs.gas_properties,
    )

    m.fs.Outlet_Valve = Valve(
        dynamic=False,
        valve_function_callback= ValveFunctionType.linear,
        property_package=m.fs.gas_properties,
    )

    m.fs.inlet_valve2bed = Arc(
        source=m.fs.Inlet_Valve.outlet, destination=m.fs.FB.gas_inlet
    )

    m.fs.bed2outlet_valve = Arc(
        source=m.fs.FB.gas_outlet, destination=m.fs.Outlet_Valve.inlet
    )

    # Call Pyomo function to apply above arc connections
    TransformationFactory("network.expand_arcs").apply_to(m.fs)

    if m.dynamic:
        m.discretizer = TransformationFactory("dae.finite_difference")
        m.discretizer.apply_to(m, nfe=nstep, wrt=m.fs.time, scheme="BACKWARD")
    m.fs.FB.kf["CO2"] = 0.005
    m.fs.FB.kf["H2O"] = 0.0025 # originally 0.0143, high Kf["H2O"] causes convergence issue if RH in air is high
    m.fs.FB.bed_diameter.fix(0.1)
    m.fs.FB.wall_diameter.fix(0.105)
    m.fs.FB.bed_height.fix(0.01)
    m.fs.FB.particle_dia.fix(5.2e-4)
    m.fs.FB.heat_transfer_coeff_gas_wall = 3.53 # original 35.3
    m.fs.FB.heat_transfer_coeff_fluid_wall = 0.22 # 220 in Young's paper, use very low value for adiabatic case
    m.fs.FB.fluid_temperature.fix(298.15)
    m.fs.FB.joule_heating_rate.fix(0)

    flow_mol_gas = 0.0681
    m.fs.Inlet_Valve.Cv.fix(0.003)  # Estimated to get the desired flow rates at 90% valve opening
    m.fs.Inlet_Valve.valve_opening.fix(0.9)
    m.fs.Inlet_Valve.inlet.flow_mol.fix(flow_mol_gas)
    m.fs.Inlet_Valve.inlet.temperature.fix(298.15)
    m.fs.Inlet_Valve.inlet.pressure.fix(114000)
    m.fs.Inlet_Valve.inlet.mole_frac_comp[:, "CO2"].fix(0.0004)
    m.fs.Inlet_Valve.inlet.mole_frac_comp[:, "H2O"].fix(0.01)
    m.fs.Inlet_Valve.inlet.mole_frac_comp[:,  "N2"].fix(0.9896)

    m.fs.Outlet_Valve.Cv.fix(0.003) # Estimated to get the desired flow rates at 90% valve opening
    m.fs.Outlet_Valve.outlet.pressure.fix(101325)

    iscale.set_scaling_factor(m.fs.FB.gas_phase.heat, 1e-2)
    iscale.set_scaling_factor(m.fs.FB.gas_phase.area, 1e3)
    iscale.set_scaling_factor(m.fs.Inlet_Valve.control_volume.work, 1e-3)
    iscale.set_scaling_factor(m.fs.Outlet_Valve.control_volume.work, 1e-3)
    iscale.calculate_scaling_factors(m)

    # initialize flowsheet model
    if m.dynamic:
        m.fs.FB.set_initial_condition()
        m.fs.Inlet_Valve.valve_opening.fix()
        m.fs.Outlet_Valve.valve_opening.fix()
        m.fs.Inlet_Valve.inlet.flow_mol.unfix()
    else:
        solver = get_solver("ipopt_v2")
        # initialize by fixing flow rate and changing inlet and outlet valve openings
        m.fs.Inlet_Valve.initialize()
        propagate_state(m.fs.inlet_valve2bed)
        m.fs.FB.initialize(outlvl=4)
        propagate_state(m.fs.bed2outlet_valve)
        m.fs.Outlet_Valve.valve_opening.unfix()
        m.fs.Outlet_Valve.initialize()
        print("flow_mol=", value(m.fs.Inlet_Valve.inlet.flow_mol[0]))
        print("Cvs of inlet and outlet valves=", value(m.fs.Inlet_Valve.Cv), value(m.fs.Outlet_Valve.Cv))
        print("openings of inlet and outlet valves=", value(m.fs.Inlet_Valve.valve_opening[0]), value(m.fs.Outlet_Valve.valve_opening[0]))
        print("bed inlet and outlet pressures = ", value(m.fs.FB.gas_inlet.pressure[0]), value(m.fs.FB.gas_outlet.pressure[0]))
        # unfix flow rate but fix two valve openings, calculate flow rate
        m.fs.Inlet_Valve.inlet.flow_mol.unfix()
        m.fs.Inlet_Valve.valve_opening.fix(0.9)
        m.fs.Outlet_Valve.valve_opening.fix(0.2079)
        solver.solve(m, tee=True)
        print("flow_mol=", value(m.fs.Inlet_Valve.inlet.flow_mol[0]))
        print("Cvs of inlet and outlet valves=", value(m.fs.Inlet_Valve.Cv), value(m.fs.Outlet_Valve.Cv))
        print("openings of inlet and outlet valves=", value(m.fs.Inlet_Valve.valve_opening[0]), value(m.fs.Outlet_Valve.valve_opening[0]))
        print("bed inlet and outlet pressures = ", value(m.fs.FB.gas_inlet.pressure[0]), value(m.fs.FB.gas_outlet.pressure[0]))
        assert_units_consistent(m)
        print("assert_units_consistent called")
    return m


def main_steady_state():
    m = get_model(dynamic=False)
    return m


def main_dynamic():
    m_ss = get_model(dynamic=False)
    m_dyn = get_model(dynamic=True)
    copy_non_time_indexed_values(
            m_dyn.fs, m_ss.fs, copy_fixed=True, outlvl=idaeslog.ERROR
        )
    for t in m_dyn.fs.time:
        copy_values_at_time(
            m_dyn.fs, m_ss.fs, t, 0.0, copy_fixed=True, outlvl=idaeslog.ERROR
        )
    optarg = {
    "max_iter": 100,
    "nlp_scaling_method": "user-scaling",
    #"halt_on_ampl_error": "yes",
    "linear_solver": "ma57",
    }
    solver = get_solver("ipopt_v2")
    solver.options = optarg
    #add disturbance and solve dynamic model
    yco2_1 = 0.00001
    yh2o_1 = 0.99998
    yn2_1  = 0.00001
    for t in m_dyn.fs.time:
        if t>=30:
            m_dyn.fs.FB.joule_heating_rate[t,:].fix(2.2e5)
            m_dyn.fs.FB.fluid_temperature[t].fix(333.15)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"CO2"].fix(yco2_1)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"H2O"].fix(yh2o_1)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t, "N2"].fix(yn2_1)
            m_dyn.fs.Inlet_Valve.inlet.temperature[t].fix(333.15)
            m_dyn.fs.Inlet_Valve.inlet.pressure[t].fix(20500)
            m_dyn.fs.Outlet_Valve.outlet.pressure[t].fix(20000)
        if t>=210 and t<450:
            qjoule = 2.2e5 -2.2e5/(450-210)*(t-210)
            m_dyn.fs.FB.joule_heating_rate[t,:].fix(qjoule)
        if t>=450:
            m_dyn.fs.FB.joule_heating_rate[t,:].fix(0)
       
    dof = degrees_of_freedom(m_dyn)
    print("dof of dynamic model=", dof)   
    print("inlet and outlet valve opening at t=0 are", value(m_dyn.fs.Inlet_Valve.valve_opening[0]), value(m_dyn.fs.Outlet_Valve.valve_opening[0]))

    # solve each time element one by one
    initialize_by_time_element(m_dyn.fs, m_dyn.fs.time, solver=solver, outlvl=4)
    solver.solve(m_dyn,tee=True)
    #write_dynamic_results_to_csv(m_dyn,"microwave_results.csv")
    # calculate total CO2 leaving the bed over the simulation time
    co2_desorbed = 0
    t_prev = 0
    for t in m_dyn.fs.config.time:
        if t>0:
            co2_desorbed += value(m_dyn.fs.FB.gas_phase.properties[t,1].flow_mol_comp["CO2"]*(t-t_prev))
            t_prev = t
    print("mole of CO2 desorbed:", co2_desorbed)
    # plot figures
    time = []
    xlabel = ["x=0.0", "x=0.2", "x=0.4", "x=0.6", "x=0.8", "x=1.0"]
    xpoint = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    solid_temp = {}
    gas_temp = {}
    wall_temp = {}
    co2_mf = {}
    h2o_mf = {}
    n2_mf = {}
    loading_co2 = {}
    loading_co2_eq = {}
    loading_h2o = {}
    loading_h2o_eq = {}
    pres = {}
    flow_mol = {}
    vel_sup = {}
    heat_fluid = {}
    heat_joule = {}
    for t in m_dyn.fs.config.time:
        time.append(t)
    ix = 0
    for x in xpoint:
        y_solid_temp = []
        y_gas_temp = []
        y_wall_temp = []
        y_co2_mf = []
        y_h2o_mf = []
        y_n2_mf = []
        y_loading_co2 = []
        y_loading_co2_eq = []
        y_loading_h2o = []
        y_loading_h2o_eq = []
        y_pres = []
        y_flow_mol = []
        y_vel_sup = []
        y_heat_fluid = []
        y_heat_joule = []
        for t in m_dyn.fs.config.time:
            y_solid_temp.append(value(m_dyn.fs.FB.solid_temperature[t,x]-273.15))
            y_gas_temp.append(value(m_dyn.fs.FB.gas_phase.properties[t,x].temperature-273.15))
            y_wall_temp.append(value(m_dyn.fs.FB.wall_temperature[t,x]-273.15))
            y_co2_mf.append(value(m_dyn.fs.FB.gas_phase.properties[t,x].mole_frac_comp["CO2"]))
            y_h2o_mf.append(value(m_dyn.fs.FB.gas_phase.properties[t,x].mole_frac_comp["H2O"]))
            y_n2_mf.append(value(m_dyn.fs.FB.gas_phase.properties[t,x].mole_frac_comp["N2"]))
            y_loading_co2.append(value(m_dyn.fs.FB.adsorbate_loading[t,x,"CO2"]))
            y_loading_co2_eq.append(value(m_dyn.fs.FB.adsorbate_loading_equil[t,x,"CO2"]))
            y_loading_h2o.append(value(m_dyn.fs.FB.adsorbate_loading[t,x,"H2O"]))
            y_loading_h2o_eq.append(value(m_dyn.fs.FB.adsorbate_loading_equil[t,x,"H2O"]))
            y_pres.append(value(m_dyn.fs.FB.gas_phase.properties[t,x].pressure))
            y_flow_mol.append(value(m_dyn.fs.FB.gas_phase.properties[t,x].flow_mol))
            y_vel_sup.append(value(m_dyn.fs.FB.velocity_superficial_gas[t,x]))
            y_heat_fluid.append(value(m_dyn.fs.FB.heat_fluid_to_wall[t,x]))
            y_heat_joule.append(value(m_dyn.fs.FB.joule_heating_rate[t,x]*m_dyn.fs.FB.bed_area))
        solid_temp[xlabel[ix]] = y_solid_temp
        gas_temp[xlabel[ix]] = y_gas_temp
        wall_temp[xlabel[ix]] = y_wall_temp
        co2_mf[xlabel[ix]] = y_co2_mf
        h2o_mf[xlabel[ix]] = y_h2o_mf
        n2_mf[xlabel[ix]] = y_n2_mf
        loading_co2[xlabel[ix]] = y_loading_co2
        loading_co2_eq[xlabel[ix]] = y_loading_co2_eq
        loading_h2o[xlabel[ix]] = y_loading_h2o
        loading_h2o_eq[xlabel[ix]] = y_loading_h2o_eq
        pres[xlabel[ix]] = y_pres
        flow_mol[xlabel[ix]] = y_flow_mol
        vel_sup[xlabel[ix]] = y_vel_sup
        heat_fluid[xlabel[ix]] = y_heat_fluid
        heat_joule[xlabel[ix]] = y_heat_joule
        ix += 1

    solid_temp_df = pd.DataFrame(solid_temp)
    gas_temp_df = pd.DataFrame(gas_temp)
    wall_temp_df = pd.DataFrame(wall_temp)
    co2_mf_df = pd.DataFrame(co2_mf)
    h2o_mf_df = pd.DataFrame(h2o_mf)
    n2_mf_df = pd.DataFrame(n2_mf)
    loading_co2_df = pd.DataFrame(loading_co2)
    loading_co2_eq_df = pd.DataFrame(loading_co2_eq)
    loading_h2o_df = pd.DataFrame(loading_h2o)
    loading_h2o_eq_df = pd.DataFrame(loading_h2o_eq)
    pres_df = pd.DataFrame(pres)
    flow_mol_df = pd.DataFrame(flow_mol)
    vel_sup_df = pd.DataFrame(vel_sup)
    heat_fluid_df = pd.DataFrame(heat_fluid)
    heat_joule_df = pd.DataFrame(heat_joule)

    plt.figure(1)
    plt.plot(time, solid_temp_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Solid Temperature [C]")
    plt.show(block=False)

    plt.figure(2)
    plt.plot(time, gas_temp_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Gas Temperature [C]")
    plt.show(block=False)

    plt.figure(3)
    plt.plot(time, wall_temp_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Wall Temperature [C]")
    plt.show(block=False)

    plt.figure(4)
    plt.plot(time, co2_mf_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("CO2 Mole Fraction []")
    plt.show(block=False)

    plt.figure(5)
    plt.plot(time, h2o_mf_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("H2O Mole Fraction []")
    plt.show(block=False)

    plt.figure(6)
    plt.plot(time, n2_mf_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("N2 Mole Fraction []")
    plt.show(block=False)

    plt.figure(7)
    plt.plot(time, loading_co2_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("CO2 Loading [mol/kg]")
    plt.show(block=False) 
    
    plt.figure(8)
    plt.plot(time, loading_co2_eq_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("CO2 Loading at Equilibrium [mol/kg]")
    plt.show(block=False)

    plt.figure(9)
    plt.plot(time, loading_h2o_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("H2O Loading [mol/kg]")
    plt.show(block=False) 

    plt.figure(10)
    plt.plot(time, loading_h2o_eq_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("H2O Loading at Equilibrium [mol/kg]")
    plt.show(block=False) 

    plt.figure(11)
    plt.plot(time, pres_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    plt.show(block=False)

    plt.figure(12)
    plt.plot(time, flow_mol_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Gas Flow [mol/s]")
    plt.show(block=False)

    plt.figure(13)
    plt.plot(time, vel_sup_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Bed Superficial Velocity [m/s]")
    plt.show(block=False)

    plt.figure(14)
    plt.plot(time, heat_fluid_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Heat From Fluid Per Bed Length [W/m]")
    plt.show(block=False)

    plt.figure(15)
    plt.plot(time, heat_joule_df, label=xlabel)
    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Joule Heat Duty [W/m]")
    plt.show(block=True)

    return m_dyn

def write_dynamic_results_to_csv(m_dyn,filename="dynamic_results.csv"):
    # row index and column heading for DataFrame
    col = []
    col.append("time_s")
    var_heading = ("solid_temperature_C_x",
                   "gas_tempreature_C_x",
                   "wall_temperature_C_x",
                   "mole_frac_CO2_x",
                   "mole_frac_H2O_x",
                   "mole_frac_N2_x",
                   "loading_CO2_mol/kg_x",
                   "eq_loading_CO2_mol/kg_x",
                   "loading_H2O_mol/kg_x",
                   "eq_loading_H2O_mol/kg_x",
                   "pressure_Pa_x",
                   "flow_mol/s_x",
                   "velocity_superficial_m/s_x",
                   "heat_fluid_to_wall_W/m_x"
                   "Joule_heat_input_W/m_x"
                   )
    for ivar in var_heading:
        for x in m_dyn.fs.FB.length_domain:
            col.append(f"{ivar}{x}")
    col.append("inlet_CO2_flow_mol/s")
    col.append("inlet_H2O_flow_mol/s")
    col.append("inlet_N2_flow_mol/s")
    col.append("outlet_CO2_flow_mol/s")
    col.append("outlet_H2O_flow_mol/s")
    col.append("outlet_N2_flow_mol/s")
    len_x = len(m_dyn.fs.FB.length_domain)
    nrow = len(m_dyn.fs.time)
    nvar = len(var_heading)  # number of variables to be written
    ncol = len_x*nvar + 7
    data = np.zeros([nrow,ncol])
    irow = 0
    ind = []
    for t in m_dyn.fs.time:
        data[irow,0] = t
        irow += 1
        ind.append(irow)
    ix = 0
    for x in m_dyn.fs.FB.length_domain:
        irow = 0
        for t in m_dyn.fs.time:
            ivar = 0
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.solid_temperature[t,x]-273.15)
            ivar = 1
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,x].temperature-273.15)
            ivar = 2
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.wall_temperature[t,x]-273.15)
            ivar = 3
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,x].mole_frac_comp["CO2"])
            ivar = 4
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,x].mole_frac_comp["H2O"])
            ivar = 5
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,x].mole_frac_comp["N2"])
            ivar = 6
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.adsorbate_loading[t,x,"CO2"])
            ivar = 7
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.adsorbate_loading_equil[t,x,"CO2"])
            ivar = 8
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.adsorbate_loading[t,x,"H2O"])
            ivar = 9
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.adsorbate_loading_equil[t,x,"H2O"])
            ivar = 10
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,x].pressure)
            ivar = 11
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,x].flow_mol)
            ivar = 12
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.velocity_superficial_gas[t,x])
            ivar = 13
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.heat_fluid_to_wall[t,x])
            ivar = 14
            data[irow,1+ix+len_x*ivar] = value(m_dyn.fs.FB.joule_heating_rate[t,x]*m_dyn.fs.FB.bed_area)
            irow += 1
        ix += 1
    irow = 0
    for t in m_dyn.fs.time:
        ivar = 0
        data[irow,1+len_x*nvar+ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,0].flow_mol_comp["CO2"])
        ivar = 1
        data[irow,1+len_x*nvar+ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,0].flow_mol_comp["H2O"])
        ivar = 2
        data[irow,1+len_x*nvar+ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,0].flow_mol_comp["N2"])
        ivar = 3
        data[irow,1+len_x*nvar+ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,1].flow_mol_comp["CO2"])
        ivar = 4
        data[irow,1+len_x*nvar+ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,1].flow_mol_comp["H2O"])
        ivar = 5
        data[irow,1+len_x*nvar+ivar] = value(m_dyn.fs.FB.gas_phase.properties[t,1].flow_mol_comp["N2"])
        irow += 1
    df = pd.DataFrame(data, index=ind, columns=col)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    # Main function to to run simulation
    # To run steady-state model, call main_steady()
    # to run dynamic model, call main_dyn()
    m_dyn = main_dynamic()
    #m_ss = main_steady_state()




