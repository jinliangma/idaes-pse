###############################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
###############################################################################

# Sript for the simulation of a TVSA cycle for DAC application

# ## Simulation details
# The simulation is set up for the verification of the model. The goal is to
# successfully compare the model predictions with the experimental breakthrough
# data published in the dissertation of Joss, L (2016), 
# DOI: https://doi.org/10.3929/ethz-a-010722158, shown in Figure 3.2, page 37. 
# Equipment and inlet stream properties: 
# - Bed height: 0.01 m
# - Bed diameter: 0.1 m
# - This example flowsheet is based on NETL polymer sorbent
# 


import numpy as np
import time
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
from idaes.models.unit_models import ValveFunctionType, Valve
from idaes.core import FlowsheetBlock, EnergyBalanceType
import idaes.core.solvers.petsc as petsc  # PETSc utilities module
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.dyn_utils import copy_values_at_time, copy_non_time_indexed_values
from idaes.core.util.initialization import initialize_by_time_element, propagate_state
import idaes.logger as idaeslog
import logging

# from idaes.models_extra.gas_solid_contactors.unit_models.fixed_bed_1D import FixedBed1D
from idaes.models_extra.tvsa_1d.unit_models.fixed_bed_1d import FixedBed1D  # fixed_bed_1D should be in the same directory

from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models_extra.power_generation.properties.natural_gas_PR import (
    EosType,
    get_prop,
)

__author__ = "Chinedu Okoli, Anca Ostace, Jinliang Ma"


def get_model(dynamic=True, time_set=None, nstep=None, init=True):
    m = ConcreteModel()
    m.dynamic = dynamic
    if time_set is None:
        time_set = [0,20,40,60,900,930,960,1800]
    if nstep is None:
        nstep = 50
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
        adsorbent="netl_sorbent",
        coadsorption_isotherm="None",
        adsorbent_shape="particle",
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
    m.fs.FB.kf["CO2"] = 0.01
    m.fs.FB.kf["H2O"] = 0.01
    m.fs.FB.bed_diameter.fix(0.1)
    m.fs.FB.wall_diameter.fix(0.105)
    m.fs.FB.bed_height.fix(0.01)
    m.fs.FB.particle_dia.fix(2e-3) #5.2e-4 for Lewatit
    m.fs.FB.heat_transfer_coeff_gas_wall = 35.3
    m.fs.FB.heat_transfer_coeff_fluid_wall = 220
    m.fs.FB.fluid_temperature.fix(348.15)
    #m.fs.FB.hd_monolith.fix(0.005)

    # Fix boundary values for gas for all time
    # Gas inlet, 0.1175 mol/s based on Ryan's Aspen model
    # corresponding to an interstitial velocity at 0.5229 m/s and superficial velocity at 0.3660 at 1 atm
    # assume steam flow at the end of desorption step is about 1/200 of the air flow when inlet valve is 10% open
    flow_mol_gas = 0.1175/200
    m.fs.Inlet_Valve.Cv.fix(0.003)  # Estimated to get the desired flow rates at 90% valve opening
    m.fs.Inlet_Valve.valve_opening.fix(0.1)
    m.fs.Inlet_Valve.inlet.flow_mol.fix(flow_mol_gas)
    m.fs.Inlet_Valve.inlet.temperature.fix(348.15)
    m.fs.Inlet_Valve.inlet.pressure.fix(20050) # about 0.2 inch water pressure higher than outlet
    m.fs.Inlet_Valve.inlet.mole_frac_comp[:, "CO2"].fix(0.00001)
    m.fs.Inlet_Valve.inlet.mole_frac_comp[:, "H2O"].fix(0.99998)
    m.fs.Inlet_Valve.inlet.mole_frac_comp[:,  "N2"].fix(0.00001)

    m.fs.Outlet_Valve.Cv.fix(0.003) # Estimated to get the desired flow rates at 90% valve opening
    m.fs.Outlet_Valve.outlet.pressure.fix(2e4)

    iscale.set_scaling_factor(m.fs.FB.gas_phase.heat, 1e-2)
    iscale.set_scaling_factor(m.fs.FB.gas_phase.area, 1e4)
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
        m.fs.Inlet_Valve.initialize(outlvl=4)
        propagate_state(m.fs.inlet_valve2bed)
        m.fs.FB.initialize(outlvl=4)
        propagate_state(m.fs.bed2outlet_valve)
        m.fs.Outlet_Valve.valve_opening.unfix()
        m.fs.Outlet_Valve.initialize(outlvl=4)
        print("flow_mol=", value(m.fs.Inlet_Valve.inlet.flow_mol[0]))
        print("Cvs of inlet and outlet valves=", value(m.fs.Inlet_Valve.Cv), value(m.fs.Outlet_Valve.Cv))
        print("openings of inlet and outlet valves=", value(m.fs.Inlet_Valve.valve_opening[0]), value(m.fs.Outlet_Valve.valve_opening[0]))
        print("bed inlet and outlet pressures = ", value(m.fs.FB.gas_inlet.pressure[0]), value(m.fs.FB.gas_outlet.pressure[0]))
        # unfix flow rate but fix two valve openings, calculate flow rate
        m.fs.Inlet_Valve.inlet.flow_mol.unfix()
        m.fs.Inlet_Valve.valve_opening.fix(0.05)
        m.fs.Outlet_Valve.valve_opening.fix(0.5)
        solver.solve(m, tee=True)
        print("flow_mol=", value(m.fs.Inlet_Valve.inlet.flow_mol[0]))
        print("Cvs of inlet and outlet valves=", value(m.fs.Inlet_Valve.Cv), value(m.fs.Outlet_Valve.Cv))
        print("openings of inlet and outlet valves=", value(m.fs.Inlet_Valve.valve_opening[0]), value(m.fs.Outlet_Valve.valve_opening[0]))
        print("bed inlet and outlet pressures = ", value(m.fs.FB.gas_inlet.pressure[0]), value(m.fs.FB.gas_outlet.pressure[0]))

    return m


def main_steady_state():
    m = get_model(dynamic=False)
    return m

def main_steady_state_steps():
    #steady state at end of desorption step
    m = get_model(dynamic=False)
    print_inputs_outputs(m.fs)
    #steady state at the end of pressurization
    m.fs.Inlet_Valve.inlet.mole_frac_comp[0,"CO2"].fix(0.0004)
    m.fs.Inlet_Valve.inlet.mole_frac_comp[0,"H2O"].fix(0.01516)
    m.fs.Inlet_Valve.inlet.mole_frac_comp[0, "N2"].fix(0.98444)
    m.fs.Inlet_Valve.inlet.pressure[0].fix(105000)
    m.fs.Inlet_Valve.inlet.temperature[0].fix(298.15)
    m.fs.Inlet_Valve.valve_opening[0].fix(0.9)
    m.fs.Outlet_Valve.valve_opening[0].fix(0.05)
    m.fs.Outlet_Valve.outlet.pressure[0].fix(101325)
    m.fs.FB.fluid_temperature.fix(298.15)
    solver = get_solver("ipopt")
    solver.solve(m, tee=True)
    print_inputs_outputs(m.fs)
    #steady state at the end of adsorption
    m.fs.Inlet_Valve.inlet.mole_frac_comp[0,"CO2"].fix(0.0004)
    m.fs.Inlet_Valve.inlet.mole_frac_comp[0,"H2O"].fix(0.01516)
    m.fs.Inlet_Valve.inlet.mole_frac_comp[0, "N2"].fix(0.98444)
    m.fs.Inlet_Valve.inlet.pressure[0].fix(105000)
    m.fs.Inlet_Valve.inlet.temperature[0].fix(298.15)
    m.fs.Inlet_Valve.valve_opening[0].fix(0.9)
    m.fs.Outlet_Valve.valve_opening[0].fix(0.9)
    m.fs.Outlet_Valve.outlet.pressure[0].fix(101325)
    m.fs.FB.fluid_temperature.fix(298.15)
    solver = get_solver("ipopt")
    solver.solve(m, tee=True)
    print_inputs_outputs(m.fs)
    return m

def print_inputs_outputs(fs):
    print("------------------------------------------------------")
    print("Inlet openning =", value(fs.Inlet_Valve.valve_opening[0]))
    print("Outlet openning =", value(fs.Outlet_Valve.valve_opening[0]))
    print("Inlet pressure =", value(fs.Inlet_Valve.inlet.pressure[0]))
    print("Outlet pressure =", value(fs.Outlet_Valve.outlet.pressure[0]))
    print("Bed inlet pressure =", value(fs.Inlet_Valve.outlet.pressure[0]))
    print("Bed outlet pressure =", value(fs.Outlet_Valve.inlet.pressure[0]))
    print("Inlet temperature =", value(fs.Inlet_Valve.inlet.temperature[0]))
    print("Outlet temperature =", value(fs.Outlet_Valve.outlet.temperature[0]))
    print("Mole flow rate =", value(fs.Inlet_Valve.inlet.flow_mol[0]))
    print("Y_CO2 =", value(fs.Inlet_Valve.inlet.mole_frac_comp[0,"CO2"]))
    print("Y_H2O =", value(fs.Inlet_Valve.inlet.mole_frac_comp[0,"H2O"]))
    print("Y_N2 =", value(fs.Inlet_Valve.inlet.mole_frac_comp[0,"N2"]))
    print("Fluid temperature =", value(fs.FB.fluid_temperature[0]))


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
    "linear_solver": "ma27",
    }
    solver = get_solver("ipopt_v2")
    solver.options = optarg
    #solve without disturbance
    dof = degrees_of_freedom(m_dyn)
    print("dof of dynamic model=", dof)
    print("inlet and outlet valve opening at t=0 are", value(m_dyn.fs.Inlet_Valve.valve_opening[0]), value(m_dyn.fs.Outlet_Valve.valve_opening[0]))
    #solver.solve(m_dyn,tee=True)
    #add disturbance and solve dynamic model
    for t in m_dyn.fs.time:
        yco2_0 = 0.00001
        yco2_1 = 0.0004
        yh2o_0 = 0.99998
        yh2o_1 = 0.01565
        yn2_0 = 0.00001
        yn2_1 = 0.98444
        pin_0 = 20050
        pin_1 = 105000
        pout_0 = 20000
        pout_1 = 101325
        Tin_0 = 348.15
        Tin_1 = 298.15
        Tfluid_0 = 348.15
        Tfluid_1 = 298.15
        openin_0 = 0.05
        openin_1 = 0.2
        openin_2 = 0.9
        openin_3 = 0.025 # for purge step
        openout_0 = 0.5
        openout_1 = 0.1
        openout_2 = 0.9
        openout_3 = 0.5
        if t<=40: # cooling step, switch to air inlet without vacuum, pressurization
            if t>10:
                dt = 20
                yco2 = yco2_0 + (t-20)*(yco2_1-yco2_0)/dt
                yh2o = yh2o_0 + (t-20)*(yh2o_1-yh2o_0)/dt
                yn2 = yn2_0 + (t-20)*(yn2_1-yn2_0)/dt
                pin = pin_0 + (t-20)*(pin_1-pin_0)/dt
                pout = pout_0 + (t-20)*(pout_1-pout_0)/dt
                Tin = Tin_0 + (t-20)*(Tin_1-Tin_0)/dt
                Tfluid = Tfluid_0 + (t-20)*(Tfluid_1-Tfluid_0)/dt
                openin = openin_0 + (t-20)*(openin_1-openin_0)/dt
                openout = openout_0 + (t-20)*(openout_1-openout_0)/dt
                m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"CO2"].fix(yco2)
                m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"H2O"].fix(yh2o)
                m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t, "N2"].fix(yn2)
                m_dyn.fs.Inlet_Valve.inlet.pressure[t].fix(pin)
                m_dyn.fs.Outlet_Valve.outlet.pressure[t].fix(pout)
                m_dyn.fs.Inlet_Valve.inlet.temperature[t].fix(Tin)
                m_dyn.fs.FB.fluid_temperature.fix(Tfluid_0)
                m_dyn.fs.Inlet_Valve.valve_opening[t].fix(openin)
                m_dyn.fs.Outlet_Valve.valve_opening[t].fix(openout)
        elif t<=60: # pressurization step, open inlet and outlet valve more
            dt = 20
            openin = openin_1 + (t-40)*(openin_2-openin_1)/dt
            openout = openout_1 + (t-40)*(openout_2-openout_1)/dt
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"CO2"].fix(yco2_1)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"H2O"].fix(yh2o_1)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t, "N2"].fix(yn2_1)
            m_dyn.fs.Inlet_Valve.inlet.pressure[t].fix(pin_1)
            m_dyn.fs.Outlet_Valve.outlet.pressure[t].fix(pout_1)
            m_dyn.fs.Inlet_Valve.inlet.temperature[t].fix(Tin_1)
            m_dyn.fs.FB.fluid_temperature[t].fix(Tfluid_1)
            m_dyn.fs.Inlet_Valve.valve_opening[t].fix(openin)
            m_dyn.fs.Outlet_Valve.valve_opening[t].fix(openout)
        elif t<=900: # adsorption step, open inlet and outlet valves to maximum
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"CO2"].fix(yco2_1)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"H2O"].fix(yh2o_1)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t, "N2"].fix(yn2_1)
            m_dyn.fs.Inlet_Valve.inlet.pressure[t].fix(pin_1)
            m_dyn.fs.Outlet_Valve.outlet.pressure[t].fix(pout_1)
            m_dyn.fs.Inlet_Valve.inlet.temperature[t].fix(Tin_1)
            m_dyn.fs.FB.fluid_temperature[t].fix(Tfluid_1)
            m_dyn.fs.Inlet_Valve.valve_opening[t].fix(openin_2)
            m_dyn.fs.Outlet_Valve.valve_opening[t].fix(openout_2)
        elif t<=960: # purge step, add very small amount of steam with inlet valve opening very small
            dt = 60
            openin = openin_3 + (t-900)*(openin_0-openin_3)/dt
            yco2 = yco2_1 + (t-900)*(yco2_0-yco2_1)/dt
            yh2o = yh2o_1 + (t-900)*(yh2o_0-yh2o_1)/dt
            yn2 = yn2_1 + (t-900)*(yn2_0-yn2_1)/dt
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"CO2"].fix(yco2)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"H2O"].fix(yh2o)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t, "N2"].fix(yn2)
            m_dyn.fs.Inlet_Valve.inlet.pressure[t].fix(pin_0)
            m_dyn.fs.Outlet_Valve.outlet.pressure[t].fix(pout_0)
            m_dyn.fs.Inlet_Valve.inlet.temperature[t].fix(Tin_0)
            m_dyn.fs.FB.fluid_temperature[t].fix(Tfluid_0)
            m_dyn.fs.Inlet_Valve.valve_opening[t].fix(openin)
            m_dyn.fs.Outlet_Valve.valve_opening[t].fix(openout_3)
        elif t<2000: # desorption step, open inlet and outlet valves to original openings
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"CO2"].fix(yco2_0)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t,"H2O"].fix(yh2o_0)
            m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t, "N2"].fix(yn2_0)
            m_dyn.fs.Inlet_Valve.inlet.pressure[t].fix(pin_0)
            m_dyn.fs.Outlet_Valve.outlet.pressure[t].fix(pout_0)
            m_dyn.fs.Inlet_Valve.inlet.temperature[t].fix(Tin_0)
            m_dyn.fs.FB.fluid_temperature[t].fix(Tfluid_0)
            m_dyn.fs.Inlet_Valve.valve_opening[t].fix(openin_0)
            m_dyn.fs.Outlet_Valve.valve_opening[t].fix(openout_0)
    print("inlet and outlet valve opening at t=0 are", value(m_dyn.fs.Inlet_Valve.valve_opening[0]), value(m_dyn.fs.Outlet_Valve.valve_opening[0]))

    # solve each time element one by one
    initialize_by_time_element(m_dyn.fs, m_dyn.fs.time, solver=solver)
    #solver.solve(m_dyn,tee=True)

    # solve cyclic operation by setting the time derivative terms the same for the first and last time elements
    fb = m_dyn.fs.FB
    fb.solid_energy_accumulation[0, :].unfix()
    fb.adsorbate_accumulation[0, :, :].unfix()
    fb.gas_phase.material_accumulation[0, :, :, :].unfix()
    fb.gas_phase.energy_accumulation[0, :, :].unfix()
    fb.wall_temperature_dt[0,:].unfix()
    fb.f_rlx = Param(initialize=0.15, mutable=True, doc="relaxation factor")
    @fb.Constraint(
        fb.length_domain,
        doc="Constraint for solid energy accummulation")
    def solid_energy_accummulation_eq(b,x):
        time = b.flowsheet().time
        t0 = time.first()
        t1 = time.last()
        return b.solid_energy_accumulation[t0,x] == b.solid_energy_accumulation[t1,x]*b.f_rlx

    @fb.Constraint(
        fb.length_domain,
        fb.adsorbed_components,
        doc="Constraint for solid energy accummulation")
    def adsorbate_accumulation_eq(b,x,i):
        time = b.flowsheet().time
        t0 = time.first()
        t1 = time.last()
        return b.adsorbate_accumulation[t0,x,i] == b.adsorbate_accumulation[t1,x,i]*b.f_rlx

    @fb.Constraint(
        fb.length_domain,
        m_dyn.fs.gas_properties.component_list,
        doc="Constraint for gas material accummulation")
    def gas_material_accumulation_eq(b,x,i):
        time = b.flowsheet().time
        t0 = time.first()
        t1 = time.last()
        return b.gas_phase.material_accumulation[t0,x,"Vap",i] == b.gas_phase.material_accumulation[t1,x,"Vap",i]*b.f_rlx

    @fb.Constraint(
        fb.length_domain,
        doc="Constraint for gas energy accummulation")
    def gas_energy_accumulation_eq(b,x):
        time = b.flowsheet().time
        t0 = time.first()
        t1 = time.last()
        return b.gas_phase.energy_accumulation[t0,x,"Vap"] == b.gas_phase.energy_accumulation[t1,x,"Vap"]*b.f_rlx

    @fb.Constraint(
        fb.length_domain,
        doc="Constraint for wall temperature time derivative")
    def wall_temperature_dt_eq(b,x):
        time = b.flowsheet().time
        t0 = time.first()
        t1 = time.last()
        return b.wall_temperature_dt[t0,x] == b.wall_temperature_dt[t1,x]*b.f_rlx

    # adjust relaxation factor to solve cyclic problem
    fb.f_rlx = 0.2
    solver.solve(m_dyn,tee=True)
    fb.f_rlx = 1.0
    solver.solve(m_dyn,tee=True)

    #write_dynamic_results_to_csv(m_dyn, "netl_sorbent_full_cycle_results.csv")
    
    # calculate perfromance data
    time_ads = []
    time_purge = []
    time_des = []
    for t in m_dyn.fs.config.time:
        if t<=900:
            time_ads.append(t)
            if t==900:
                time_purge.append(t)
        elif t<=960:
            time_purge.append(t)
            if t==960:
                time_des.append(t)
        else:
            time_des.append(t)
    # calculate exit total flow
    flow_ads_total = 0
    flow_ads_co2 = 0
    for i in range(len(time_ads)-1):
        t1 = time_ads[i]
        t2 = time_ads[i+1]
        flow_ads_total += value((m_dyn.fs.Inlet_Valve.inlet.flow_mol[t1]+
                     m_dyn.fs.Inlet_Valve.inlet.flow_mol[t2])/2*(t2-t1))
        flow_ads_co2 += value((m_dyn.fs.Inlet_Valve.inlet.flow_mol[t1]*m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t1,"CO2"]+
                     m_dyn.fs.Inlet_Valve.inlet.flow_mol[t2]*m_dyn.fs.Inlet_Valve.inlet.mole_frac_comp[t2,"CO2"])/2*(t2-t1))
    print(f"flow_ads_total={flow_ads_total}, flow_ads_co2={flow_ads_co2}")
    flow_purge_total = 0
    flow_purge_co2 = 0
    flow_purge_n2 = 0
    flow_purge_h2o = 0
    for i in range(len(time_purge)-1):
        t1 = time_purge[i]
        t2 = time_purge[i+1]
        flow_purge_total += value((m_dyn.fs.Outlet_Valve.outlet.flow_mol[t1]+
                     m_dyn.fs.Outlet_Valve.outlet.flow_mol[t2])/2*(t2-t1))
        flow_purge_co2 += value((m_dyn.fs.Outlet_Valve.outlet.flow_mol[t1]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t1,"CO2"]+
                     m_dyn.fs.Outlet_Valve.outlet.flow_mol[t2]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t2,"CO2"])/2*(t2-t1))
        flow_purge_n2 += value((m_dyn.fs.Outlet_Valve.outlet.flow_mol[t1]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t1,"N2"]+
                     m_dyn.fs.Outlet_Valve.outlet.flow_mol[t2]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t2,"N2"])/2*(t2-t1))
        flow_purge_h2o += value((m_dyn.fs.Outlet_Valve.outlet.flow_mol[t1]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t1,"H2O"]+
                     m_dyn.fs.Outlet_Valve.outlet.flow_mol[t2]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t2,"H2O"])/2*(t2-t1))
    print(f"flow_purge_total={flow_purge_total}, flow_purge_co2={flow_purge_co2}, flow_purge_n2={flow_purge_n2}, flow_purge_h2o={flow_purge_h2o}")  

    flow_des_total = 0
    flow_des_co2 = 0
    flow_des_n2 = 0
    flow_des_h2o = 0
    for i in range(len(time_des)-1):
        t1 = time_des[i]
        t2 = time_des[i+1]
        flow_des_total += value((m_dyn.fs.Outlet_Valve.outlet.flow_mol[t1]+
                     m_dyn.fs.Outlet_Valve.outlet.flow_mol[t2])/2*(t2-t1))
        flow_des_co2 += value((m_dyn.fs.Outlet_Valve.outlet.flow_mol[t1]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t1,"CO2"]+
                     m_dyn.fs.Outlet_Valve.outlet.flow_mol[t2]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t2,"CO2"])/2*(t2-t1))
        flow_des_n2 += value((m_dyn.fs.Outlet_Valve.outlet.flow_mol[t1]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t1,"N2"]+
                     m_dyn.fs.Outlet_Valve.outlet.flow_mol[t2]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t2,"N2"])/2*(t2-t1))
        flow_des_h2o += value((m_dyn.fs.Outlet_Valve.outlet.flow_mol[t1]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t1,"H2O"]+
                     m_dyn.fs.Outlet_Valve.outlet.flow_mol[t2]*m_dyn.fs.Outlet_Valve.outlet.mole_frac_comp[t2,"H2O"])/2*(t2-t1))
    print("flow_des_total, flow_des_co2=, flow_des_n2, flow_des_h2o",flow_des_total, flow_des_co2, flow_des_n2, flow_des_h2o)  


    #------------------------------------------------------------------------------
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

    # This method builds and runs a subcritical coal-fired power plant
    # dynamic simulation. The simulation consists of 5%/min ramping down from
    # full load to 50% load, holding for 30 minutes and then ramping up
    # to 100% load and holding for 20 minutes.
    # uncomment the code (line 1821) to run this simulation,
    # note that this simulation takes around ~60 minutes to complete
    m_dyn = main_dynamic()

    # This method builds and runs a steady state subcritical coal-fired power
    # plant, the simulation consists of a typical base load case.
    #m_ss = main_steady_state_steps()




