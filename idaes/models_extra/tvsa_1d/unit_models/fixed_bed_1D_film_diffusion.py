#################################################################################
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
#################################################################################
"""
IDAES 1D Fixed Bed model.
Gas phase based on IDAES ControlVolume1DBlock unit model.
Gas phase property package based on IDAES generic property model and set as ideal gas mixture.
Solid phase variables declared locally without using solid property package.
Isotherm variables are also declared locally.
Currently only Zeolite-13X, NETL polymer sorbent, and Lewatit sorbent isotherms are implemented.
NETL sorbent isotherm contains an enhancement factor due to H2O in humid air.
The Lewatit sorbent has three CO2/H2O co-adsorption isoterms including Stampi-Bombelli, WADST, and Mechanistic
Enthalpy transfer due to mass transfer between gas and solid phases is considered
Column wall and water jacket for heating and cooling is considered and the wall temperatures are solved
The bed could be a packed bed with sorbent beads or a monolith bed
"""

# Import Pyomo libraries
from pyomo.environ import (
    Set,
    Var,
    Param,
    Reals,
    value,
    TransformationFactory,
    Constraint,
    check_optimal_termination,
    exp,
    log,
    sqrt,
    units as pyunits,
)
from pyomo.common.config import ConfigValue, In, Bool
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.dae import ContinuousSet, DerivativeVar

# Import IDAES cores
from idaes.core import (
    ControlVolume1DBlock,
    UnitModelBlockData,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    FlowDirection,
    DistributedVars,
)
from idaes.core.util.config import (
    is_physical_parameter_block,
)
from idaes.core.util.exceptions import (
    ConfigurationError,
    BurntToast,
)

from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.constants import Constants as constants
import idaes.logger as idaeslog
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver

__author__ = "Chinedu Okoli, Anca Ostace, Jinliang Ma"

# Set up logger
_log = idaeslog.getLogger(__name__)

# Assumptions:
# Discretized in axial (bed length or height) direction only (1-D model)
# Uniform in radius direction
# No diffusion or heat conduction in axial direction
# Adsorption and desorption rate based on linear driving force model


@declare_process_block_class("FixedBed1D")
class FixedBed1DData(UnitModelBlockData):
    """
    1D Fixed Bed Adsorption/Desorption Model Class
    """

    # Create template for unit level config arguments
    CONFIG = UnitModelBlockData.CONFIG()

    # Unit level config arguments
    CONFIG.declare(
        "finite_elements",
        ConfigValue(
            default=10,
            domain=int,
            description="Number of finite elements length domain",
            doc="""Number of finite elements to use when discretizing length
domain (default=10)""",
        ),
    )
    CONFIG.declare(
        "length_domain_set",
        ConfigValue(
            default=[0.0, 1.0],
            domain=list,
            description="Number of finite elements length domain",
            doc="""length_domain_set - (optional) list of point to use to
initialize a new ContinuousSet if length_domain is not
provided (default = [0.0, 1.0])""",
        ),
    )
    CONFIG.declare(
        "transformation_method",
        ConfigValue(
            default="dae.finite_difference",
            description="Method to use for DAE transformation",
            doc="""Method to use to transform domain. Must be a method recognized
by the Pyomo TransformationFactory,
**default** - "dae.finite_difference".
**Valid values:** {
**"dae.finite_difference"** - Use a finite difference transformation method,
**"dae.collocation"** - use a collocation transformation method}""",
        ),
    )
    CONFIG.declare(
        "transformation_scheme",
        ConfigValue(
            default=None,
            domain=In([None, "BACKWARD", "FORWARD", "LAGRANGE-RADAU"]),
            description="Scheme to use for DAE transformation",
            doc="""Scheme to use when transforming domain. See Pyomo
documentation for supported schemes,
**default** - None.
**Valid values:** {
**None** - defaults to "BACKWARD" for finite difference transformation method,
and to "LAGRANGE-RADAU" for collocation transformation method,
**"BACKWARD"** - Use a finite difference transformation method,
**"FORWARD""** - use a finite difference transformation method,
**"LAGRANGE-RADAU""** - use a collocation transformation method}""",
        ),
    )
    CONFIG.declare(
        "collocation_points",
        ConfigValue(
            default=3,
            domain=int,
            description="Number of collocation points per finite element",
            doc="""Number of collocation points to use per finite element when
discretizing length domain (default=3)""",
        ),
    )
    CONFIG.declare(
        "flow_type",
        ConfigValue(
            default="forward_flow",
            domain=In(["forward_flow", "reverse_flow"]),
            description="Flow configuration of Fixed Bed",
            doc="""Flow configuration of Fixed Bed
**default** - "forward_flow".
**Valid values:** {
**"forward_flow"** - gas flows from 0 to 1,
**"reverse_flow"** -  gas flows from 1 to 0.}""",
        ),
    )
    CONFIG.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.componentTotal,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
**default** - MaterialBalanceType.componentTotal.
**Valid values:** {
**MaterialBalanceType.none** - exclude material balances,
**MaterialBalanceType.componentPhase** - use phase component balances,
**MaterialBalanceType.componentTotal** - use total component balances,
**MaterialBalanceType.elementTotal** - use total element balances,
**MaterialBalanceType.total** - use total material balance.}""",
        ),
    )
    CONFIG.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.enthalpyTotal,
            domain=In(EnergyBalanceType),
            description="Energy balance construction flag",
            doc="""Indicates what type of energy balance should be constructed,
**default** - EnergyBalanceType.enthalpyTotal.
**Valid values:** {
**EnergyBalanceType.none** - exclude energy balances,
**EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
**EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
**EnergyBalanceType.energyTotal** - single energy balance for material,
**EnergyBalanceType.energyPhase** - energy balances for each phase.}""",
        ),
    )
    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
**default** - MomentumBalanceType.pressureTotal.
**Valid values:** {
**MomentumBalanceType.none** - exclude momentum balances,
**MomentumBalanceType.pressureTotal** - single pressure balance for material,
**MomentumBalanceType.pressurePhase** - pressure balances for each phase,
**MomentumBalanceType.momentumTotal** - single momentum balance for material,
**MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
        ),
    )
    CONFIG.declare(
        "has_pressure_change",
        ConfigValue(
            default=True,
            domain=Bool,
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be
constructed,
**default** - False.
**Valid values:** {
**True** - include pressure change terms,
**False** - exclude pressure change terms.}""",
        ),
    )
    CONFIG.declare(
        "has_joule_heating",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Joule heating flag",
            doc="""Indicates whether terms for Joule heating should be
constructed,
**default** - False.
**Valid values:** {
**True** - include Joule heating terms,
**False** - exclude Joule heating terms.}""",
        ),
    )
    CONFIG.declare(
        "has_microwave_heating",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Microwave heating flag",
            doc="""Indicates whether terms for microwave heating should be
constructed,
**default** - False.
**Valid values:** {
**True** - include microwave heating terms,
**False** - exclude microwave heating terms.}""",
        ),
    )
    CONFIG.declare(
        "pressure_drop_type",
        ConfigValue(
            default="ergun_correlation",
            domain=In(["ergun_correlation", "simple_correlation"]),
            description="Construction flag for type of pressure drop for particle bed",
            doc="""Indicates what type of pressure drop correlation should be used,
**default** - "ergun_correlation".
**Valid values:** {
**"ergun_correlation"** - Use the Ergun equation,
**"simple_correlation"** - Use a simplified pressure drop correlation. Should not
be used with fixed-bed reactors, as it is an approximation for pressure drop in 
moving-bed reactors.}""",
        ),
    )

    CONFIG.declare(
        "adsorbent",
        ConfigValue(
            default="Zeolite-13X",
            domain=In(["Zeolite-13X", "netl_sorbent", "Lewatit"]),
            description="Adsorbent flag",
            doc="""Construction flag to add adsorbent-related parameters and
        isotherms. Currently supports Zeolite 13X (Extended Sips Isotherm), 
        netl_sorbent, and Lewatit VP OC 1065.
        Default: Zeolite-13X.
        Valid values: "Zeolite-13X", "netl_sorbent", "Lewatit".""",
        ),
    )

    CONFIG.declare(
        "adsorbent_shape",
        ConfigValue(
            default="particle",
            domain=In(["particle", "monolith"]),
            description="Adsorbent shape",
            doc="""Construction flag to add adsorbent shape.
        Default: particle.
        Valid values: "particle", "monolith".""",
        ),
    )

    CONFIG.declare(
        "coadsorption_isotherm",
        ConfigValue(
            default="None",
            domain=In(["None", "Stampi-Bombelli", "Mechanistic", "WADST"]),
            description="isoterm form for CO2 and H2O co-adsorption",
            doc="""Construction flag to specify the isotherm formula for co-adsorption.
        Default: None, indicating water has no effect on CO2 uptake.
        Valid values: "None", "Stampi-Bombelli", "Mechanistic" "WADST".""",
        ),
    )

    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=None,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations
(default = 'use_parent_value')
- 'use_parent_value' - get package from parent (default = None)
- a ParameterBlock object""",
        ),
    )

    CONFIG.declare(
        "property_package_args",
        ConfigValue(
            default={},
            domain=dict,
            description="Arguments for constructing gas property package",
            doc="""A dict of arguments to be passed to the PropertyBlockData
and used when constructing these
(default = 'use_parent_value')
- 'use_parent_value' - get package from parent (default = None)
- a dict (see property package for documentation)""",
        ),
    )

    # =========================================================================
    def build(self):
        """
        Begin building model (pre-DAE transformation).

        Args:
            None

        Returns:
            None
        """
        # Call UnitModel.build to build default attributes
        super().build()

        # add adsorbent parameters
        if self.config.adsorbent == "Zeolite-13X":
            self.adsorbed_components = Set(initialize=["CO2", "N2"])
            self.enth_mol_comp_std = Param(
                self.adsorbed_components,
                initialize={"CO2": -393522.4, "N2": 0.0},
                units=pyunits.J / pyunits.mol,
                doc="standard heat of formation at 298.15 K",
            )
            self.cp_mol_comp_adsorbate = Param(
                self.adsorbed_components,
                initialize={"CO2": 36.61, "N2": 29.12},
                units=pyunits.J / pyunits.mol,
                doc="heat capacity of adsorbate at 298.15 K",
            )
            self._add_parameters_zeolite_13x()
        elif self.config.adsorbent == "netl_sorbent":
            self.adsorbed_components = Set(initialize=["CO2", "H2O"])
            self.enth_mol_comp_std = Param(
                self.adsorbed_components,
                initialize={"CO2": -393522.4, "H2O": -241826.0},
                units=pyunits.J / pyunits.mol,
                doc="standard heat of formation of gas species at 298.15 K",
            )
            self.cp_mol_comp_adsorbate = Param(
                self.adsorbed_components,
                initialize={"CO2": 36.61, "H2O": 4.184 * 18.0},
                units=pyunits.J / pyunits.mol / pyunits.K,
                doc="heat capacity of adsorbate at 298.15 K",
            )
            self._add_parameters_netl_sorbent()
        elif self.config.adsorbent == "Lewatit":
            self.adsorbed_components = Set(initialize=["CO2", "H2O"])
            self.enth_mol_comp_std = Param(
                self.adsorbed_components,
                initialize={"CO2": -393522.4, "H2O": -241826.0},
                units=pyunits.J / pyunits.mol,
                doc="standard heat of formation of gas species at 298.15 K",
            )
            self.cp_mol_comp_adsorbate = Param(
                self.adsorbed_components,
                initialize={"CO2": 36.61, "H2O": 4.184 * 18.0},
                units=pyunits.J / pyunits.mol / pyunits.K,
                doc="heat capacity of adsorbate at 298.15 K",
            )
            self._add_parameters_lewatit()
        else:
            pass  # TODO: add error message

        # Set flow direction for the gas control volume
        # Gas flows from 0 to 1
        if self.config.flow_type == "forward_flow":
            set_direction_gas = FlowDirection.forward
        # Gas flows from 1 to 0
        if self.config.flow_type == "reverse_flow":
            set_direction_gas = FlowDirection.backward

        # Consistency check for flow direction, transformation method and
        # transformation scheme
        if (
            self.config.flow_type == "forward_flow"
            and self.config.transformation_method == "dae.finite_difference"
            and self.config.transformation_scheme is None
        ):
            self.config.transformation_scheme = "BACKWARD"
        elif (
            self.config.flow_type == "reverse_flow"
            and self.config.transformation_method == "dae.finite_difference"
            and self.config.transformation_scheme is None
        ):
            self.config.transformation_scheme = "FORWARD"
        elif (
            self.config.flow_type == "forward_flow"
            and self.config.transformation_method == "dae.collocation"
            and self.config.transformation_scheme is None
        ):
            self.config.transformation_scheme = "LAGRANGE-RADAU"
        elif (
            self.config.flow_type == "reverse_flow"
            and self.config.transformation_method == "dae.collocation"
        ):
            raise ConfigurationError(
                "{} invalid value for "
                "transformation_method argument."
                "Must be "
                "dae.finite_difference "
                "if "
                "flow_type is"
                " "
                "reverse_flow"
                ".".format(self.name)
            )
        elif (
            self.config.flow_type == "forward_flow"
            and self.config.transformation_scheme == "FORWARD"
        ):
            raise ConfigurationError(
                "{} invalid value for "
                "transformation_scheme argument. "
                "Must be "
                "BACKWARD "
                "if flow_type is"
                " "
                "forward_flow"
                ".".format(self.name)
            )
        elif (
            self.config.flow_type == "reverse_flow"
            and self.config.transformation_scheme == "BACKWARD"
        ):
            raise ConfigurationError(
                "{} invalid value for "
                "transformation_scheme argument."
                "Must be "
                "FORWARD "
                "if "
                "flow_type is"
                " "
                "reverse_flow"
                ".".format(self.name)
            )
        elif (
            self.config.transformation_method == "dae.finite_difference"
            and self.config.transformation_scheme != "BACKWARD"
            and self.config.transformation_scheme != "FORWARD"
        ):
            raise ConfigurationError(
                "{} invalid value for "
                "transformation_scheme argument. "
                "Must be "
                "BACKWARD"
                " or "
                "FORWARD"
                " "
                "if transformation_method is"
                " "
                "dae.finite_difference"
                ".".format(self.name)
            )
        elif (
            self.config.transformation_method == "dae.collocation"
            and self.config.transformation_scheme != "LAGRANGE-RADAU"
        ):
            raise ConfigurationError(
                "{} invalid value for "
                "transformation_scheme argument."
                "Must be "
                "LAGRANGE-RADAU"
                " if "
                "transformation_method is"
                " "
                "dae.collocation"
                ".".format(self.name)
            )

        # Declare design parameters and variables
        self.kf = Param(
            self.adsorbed_components,
            initialize=0.4,
            mutable=True,
            units=1 / pyunits.s,
            doc="mass transfer parameter for linear driving force model",
        )

        self.heat_transfer_coeff_gas_wall = Param(
            initialize=35.5,
            mutable=True,
            units=pyunits.W / pyunits.m**2 / pyunits.K,
            doc="Global heat transfer coefficient bed-wall [J/m2/s/K]",
        )

        self.heat_transfer_coeff_fluid_wall = Param(
            initialize=200,
            mutable=True,
            units=pyunits.W / pyunits.m**2 / pyunits.K,
            doc="Global heat transfer coefficient bed-wall [J/m2/s/K]",
        )

        self.dens_wall = Param(
            initialize=7800,
            mutable=True,
            units=pyunits.kg / pyunits.m**3,
            doc="Density of wall material [kg/m3]",
        )

        self.cp_wall = Param(
            initialize=466,
            mutable=True,
            units=pyunits.J / pyunits.kg / pyunits.K,
            doc="Heat capacity of wall material [J/kg/K]",
        )

        # Create a unit model length domain
        self.length_domain = ContinuousSet(
            bounds=(0.0, 1.0),
            initialize=self.config.length_domain_set,
            doc="Normalized length domain",
        )

        self.bed_height = Var(
            initialize=1,
            doc="Bed length",
            units=pyunits.m,
        )

        self.bed_diameter = Var(
            initialize=0.1,
            doc="Reactor diameter",
            units=pyunits.m,
        )

        if self.config.adsorbent_shape == "monolith":
            self.hd_monolith = Var(
                initialize=0.005,
                doc="hydraulic diameter of monolith holes",
                units=pyunits.m,
            )
            self.hd_monolith.fix()
        else:
            self.particle_dia = Var(
                initialize=2e-3,
                doc="Particle diameter [m]",
                units=pyunits.m,
            )
            self.particle_dia.fix()

        self.wall_diameter = Var(
            initialize=1.0,
            doc="Reactor wall outside diameter",
            units=pyunits.m,
        )

        self.wall_temperature = Var(
            self.flowsheet().time,
            self.length_domain,
            initialize=298.15,
            units=pyunits.K,
            doc="Wall temperature used for external heat transfer",
        )

        if self.config.dynamic:
            self.wall_temperature_dt = DerivativeVar(
                self.wall_temperature,
                wrt=self.flowsheet().config.time,
                doc="Temperature time derivative",
                units=pyunits.K / pyunits.s,
            )

        self.fluid_temperature = Var(
            self.flowsheet().time,
            initialize=298.15,
            units=pyunits.K,
            doc="Cooling or heating fluid temperature",
        )

        # =========================================================================
        # Build control volume 1D for gas phase and populate gas control volume

        self.gas_phase = ControlVolume1DBlock(
            transformation_method=self.config.transformation_method,
            transformation_scheme=self.config.transformation_scheme,
            finite_elements=self.config.finite_elements,
            collocation_points=self.config.collocation_points,
            dynamic=self.config.dynamic,
            has_holdup=True,
            area_definition=DistributedVars.variant,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
        )

        self.gas_phase.add_geometry(
            length_domain=self.length_domain,
            length_domain_set=self.config.length_domain_set,
            length_var=self.bed_height,
            flow_direction=set_direction_gas,
        )

        self.gas_phase.add_state_blocks(
            information_flow=set_direction_gas, has_phase_equilibrium=False
        )

        self.gas_phase.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_phase_equilibrium=False,
            has_mass_transfer=True,
            has_rate_reactions=False,
        )

        self.gas_phase.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_heat_transfer=True,
            has_heat_of_reaction=False,
            has_enthalpy_transfer=True,
        )

        self.gas_phase.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )

        # =========================================================================
        # There is no solid phase control volume in this revised code
        # solid phase variables are declared as model variables
        # including temperature, loadings of individual species
        # solid parameters are also declared based on the sorbent type

        # =========================================================================
        # Add Ports for gas side
        self.add_inlet_port(name="gas_inlet", block=self.gas_phase)
        self.add_outlet_port(name="gas_outlet", block=self.gas_phase)

        # =========================================================================
        # Add performance equation method
        self._make_performance()
        self._apply_transformation()

    def _add_parameters_zeolite_13x(self):
        """
        Method to add adsorbent-related parameters to run fixed bed model.
        This method is to add parameters for Zeolite 13X.

        Reference: Hefti, M.; Marx, D.; Joss, L.; Mazzotti, M. Adsorption
        Equilibrium of Binary Mixtures of Carbon Dioxide and Nitrogen on
        Zeolites ZSM-5 and 13X. Microporous Mesoporous Materials, 215, 2014.

        """
        # adsorbent parameters
        self.voidage = Param(
            initialize=0.4,
            units=pyunits.dimensionless,
            doc="Bed voidage - external or interparticle porosity [-]",
        )
        self.particle_voidage = Param(
            initialize=0.54,  # original 0.5
            units=pyunits.dimensionless,
            doc="Particle voidage - internal or intraparticle porosity [-]",
        )
        self.cp_mass_param = Param(
            initialize=920,
            units=pyunits.J / pyunits.kg / pyunits.K,
            doc="Heat capacity of adsorbent [J/kg/K]",
        )
        self.dens_mass_particle_param = Param(
            initialize=2360,
            units=pyunits.kg / pyunits.m**3,
            doc="Density of adsorbent material without pore [kg/m3]",
        )
        # isotherm parameters
        self.dh_ads = Param(
            self.adsorbed_components,
            initialize={"CO2": -37000, "N2": -18510},
            units=pyunits.J / pyunits.mol,
            doc="Heat of adsorption [J/mol]",
        )
        self.temperature_ref = Param(
            initialize=298.15, units=pyunits.K, doc="Reference temperature [K]"
        )
        self.n_ref = Param(
            self.adsorbed_components,
            initialize={"CO2": 7.268, "N2": 4.051},
            units=pyunits.mol / pyunits.kg,
            doc="Isotherm parameter [mol/kg]",
        )
        self.X = Param(
            self.adsorbed_components,
            initialize={"CO2": -0.61684, "N2": 0.0},
            units=pyunits.dimensionless,
            doc="Isotherm parameter [-]",
        )
        self.b0 = Param(
            self.adsorbed_components,
            initialize={"CO2": 1.129e-4, "N2": 5.8470e-5},
            units=pyunits.bar**-1,
            doc="Isotherm parameter [bar-1]",
        )
        self.Qb = Param(
            self.adsorbed_components,
            initialize={"CO2": 28.389, "N2": 18.4740},
            units=pyunits.kJ / pyunits.mol,
            doc="Isotherm parameter [kJ/mol]",
        )
        self.c_ref = Param(
            self.adsorbed_components,
            initialize={"CO2": 0.42456, "N2": 0.98624},
            units=pyunits.dimensionless,
            doc="Isotherm parameter [-]",
        )
        self.alpha = Param(
            self.adsorbed_components,
            initialize={"CO2": 0.72378, "N2": 0.0},
            units=pyunits.dimensionless,
            doc="Isotherm parameter [-]",
        )

    def _add_parameters_netl_sorbent(self):
        """
        Method to add adsorbent-related parameters to run fixed bed model.
        This method is to add parameters for NETL polymer sorbent with data fitted to Toth model
        by Ryan Hughes. Enhancement factor for CO2 isotherm is applied to account for
        CO2/H2O co-adsorption. Water isotherm is not affected by CO2 loading

        """
        # adsorbent parameters
        self.voidage = Param(
            initialize=0.4,
            units=pyunits.dimensionless,
            doc="Bed voidage - external or interparticle porosity [-]",
        )
        self.particle_voidage = Param(
            initialize=0.5,
            units=pyunits.dimensionless,
            doc="Particle voidage - internal or intraparticle porosity [-]",
        )
        self.cp_mass_param = Param(
            initialize=920,
            units=pyunits.J / pyunits.kg / pyunits.K,
            doc="Heat capacity of adsorbent [J/kg/K]",
        )
        self.dens_mass_particle_param = Param(
            initialize=1000,
            units=pyunits.kg / pyunits.m**3,
            doc="Density of adsorbent material without pore [kg/m3]",
        )
        # isotherm parameters
        self.dh_ads = Param(
            self.adsorbed_components,
            initialize={"CO2": -37000, "H2O": -44004},
            units=pyunits.J / pyunits.mol,
            doc="Heat of adsorption [J/mol]",
        )
        self.temperature_ref = Param(
            initialize=298.15, units=pyunits.K, doc="Reference temperature [K]"
        )
        self.q0_chem = Param(
            self.adsorbed_components,
            initialize={"CO2": 1.114637, "H2O": 0},
            units=pyunits.mol / pyunits.kg,
            doc="Isotherm parameter based on chemisorption [mol/kg]",
        )
        self.q0_phys = Param(
            self.adsorbed_components,
            initialize={"CO2": 1.866981, "H2O": 0},
            units=pyunits.mol / pyunits.kg,
            doc="Isotherm parameter based on physisorption [mol/kg]",
        )
        self.b0_chem = Param(
            self.adsorbed_components,
            initialize={"CO2": exp(9.499742), "H2O": 0},
            units=pyunits.bar**-1,
            doc="Isotherm parameter based on chemisorption [bar-1]",
        )
        self.b0_phys = Param(
            self.adsorbed_components,
            initialize={"CO2": exp(4.665084), "H2O": 0},
            units=pyunits.bar**-1,
            doc="Isotherm parameter based on physisorption [bar-1]",
        )
        self.t0_chem = Param(
            self.adsorbed_components,
            initialize={"CO2": 1.114991, "H2O": 0},
            doc="Isotherm parameter based on chemisorption [-]",
        )
        self.t0_phys = Param(
            self.adsorbed_components,
            initialize={"CO2": 0.496593, "H2O": 0},
            doc="Isotherm parameter based on physisorption [-]",
        )
        self.alpha_chem = Param(
            self.adsorbed_components,
            initialize={"CO2": 0.0, "H2O": 0},
            doc="Isotherm parameter based on chemisorption [-]",
        )
        self.alpha_phys = Param(
            self.adsorbed_components,
            initialize={"CO2": 0.0, "H2O": 0},
            doc="Isotherm parameter based on physisorption [-]",
        )
        self.X_chem = Param(
            self.adsorbed_components,
            initialize={"CO2": -3.234665, "H2O": 0},
            doc="Isotherm parameter based on chemisorption [-]",
        )
        self.X_phys = Param(
            self.adsorbed_components,
            initialize={"CO2": 0.0, "H2O": 0},
            doc="Isotherm parameter based on physisorption [-]",
        )
        self.E_chem = Param(
            self.adsorbed_components,
            initialize={"CO2": 68.73612, "H2O": 0},
            units=pyunits.kJ / pyunits.mol,
            doc="Isotherm parameter based on chemisorption[kJ/mol]",
        )
        self.E_phys = Param(
            self.adsorbed_components,
            initialize={"CO2": 0.0, "H2O": 0},
            units=pyunits.kJ / pyunits.mol,
            doc="Isotherm parameter based on physisorption[kJ/mol]",
        )

    def _add_parameters_lewatit(self):
        """
        Method to add adsorbent-related parameters to run fixed bed model.
        This method is to add parameters for isotherm of Lewatit sorbent.
        Toth model is used for CO2 isotherm with terms corrected for co-adsorption.
        GAB model for H2O adsorption. Three options to consider the effect of co-adsorption
        including Stampi-Bombelli, mechanistic, and WADST models.
        see Young et al, The impact of binary water-CO2 isotherm models on
        the optimal performance of sorbent-based direct air capture processes,
        Energy and Environmental Science, 2021

        """
        # adsorbent parameters
        self.voidage = Param(
            initialize=0.4,
            units=pyunits.dimensionless,
            doc="Bed voidage - external or interparticle porosity [-]",
        )
        self.particle_voidage = Param(
            initialize=0.238,
            units=pyunits.dimensionless,
            doc="Particle voidage - internal or intraparticle porosity [-]",
        )
        self.cp_mass_param = Param(
            initialize=1580,
            units=pyunits.J / pyunits.kg / pyunits.K,
            doc="Heat capacity of adsorbent [J/kg/K]",
        )
        self.dens_mass_particle_param = Param(
            initialize=1155,
            units=pyunits.kg / pyunits.m**3,
            doc="Density of adsorbent material without pore [kg/m3]",
        )
        # corresponding to particle bulk density of 880 kg/m3
        # isotherm parameters
        self.dh_ads = Param(
            self.adsorbed_components,
            initialize={"CO2": -117798, "H2O": -44004},
            units=pyunits.J / pyunits.mol,
            doc="Heat of adsorption [J/mol]",
        )
        # H2O heat of adsorption based on heat of condensation (layer 2- layer 9)
        self.temperature_ref = Param(
            initialize=298.15, units=pyunits.K, doc="Reference temperature [K]"
        )
        self.q0_inf = Param(
            initialize=4.86,
            units=pyunits.mol / pyunits.kg,
            doc="Isotherm parameter based on single species isotherm [mol/kg]",
        )
        self.b0 = Param(
            initialize=2.85e-21,
            units=pyunits.Pa**-1,
            doc="Isotherm parameter based on single species isotherm [Pa-1]",
        )
        self.tau0 = Param(
            initialize=0.209,
            doc="Isotherm parameter based on single species isotherm [-]",
        )
        self.alpha = Param(
            initialize=0.523,
            doc="Isotherm parameter based on single species isotherm [-]",
        )
        self.hoa = Param(
            initialize=-117798,
            units=pyunits.J / pyunits.mol,
            doc="Isotherm parameter isosteric heat of adsorption based on single species isotherm [J/mol]",
        )
        # parameters for Guggenheim-Anderson-deBoer (GAB) model
        self.GAB_qm = Param(
            initialize=3.63,
            units=pyunits.mol / pyunits.kg,
            doc="GAB model monolayer loading [mol/kg]",
        )
        self.GAB_C = Param(
            initialize=47110.0,
            units=pyunits.J / pyunits.mol,
            doc="GAB model parameter C for E1=C-exp(DT) [J/mol]",
        )
        self.GAB_D = Param(
            initialize=0.023744,
            units=1 / pyunits.K,
            doc="GAB model parameter D for E1=C-exp(DT) [1/K]",
        )
        self.GAB_F = Param(
            initialize=57706.0,
            units=pyunits.J / pyunits.mol,
            doc="GAB model parameter F for E2_9=F+GT [J/mol]",
        )
        self.GAB_G = Param(
            initialize=-47.814,
            units=pyunits.J / pyunits.mol / pyunits.K,
            doc="GAB model parameter G for E2_9=F+GT [J/mol/K]",
        )
        self.GAB_A = Param(
            initialize=57220.0,
            units=pyunits.J / pyunits.mol,
            doc="GAB model parameter A for E10=A+BT [J/mol]",
        )
        self.GAB_B = Param(
            initialize=-44.38,
            units=pyunits.J / pyunits.mol / pyunits.K,
            doc="GAB model parameter B for E10=A+BT [J/mol/K]",
        )
        if self.config.coadsorption_isotherm == "WADST":
            self.WADST_A = Param(
                initialize=1.532,
                units=pyunits.mol / pyunits.kg,
                doc="WADST model parameter A [mol/kg]",
            )
            self.WADST_b0_wet = Param(
                initialize=1.23e-18,
                units=1 / pyunits.Pa,
                doc="WADST model parameter b0_wet [1/Pa]",
            )
            self.WADST_q0_inf_wet = Param(
                initialize=9.035,
                units=pyunits.mol / pyunits.kg,
                doc="WADST model parameter q0_inf_wet [mol/kg]",
            )
            self.WADST_tau0_wet = Param(
                initialize=0.053, doc="WADST model parameter tau0_wet [-]"
            )
            self.WADST_alpha_wet = Param(
                initialize=0.053, doc="WADST model parameter alpha_wet [-]"
            )
            self.WADST_hoa_wet = Param(
                initialize=-203687.0,
                units=pyunits.J / pyunits.mol,
                doc="WADST model parameter heat of adsorption for wet case [J/mol]",
            )
        elif self.config.coadsorption_isotherm == "Mechanistic":
            self.MECH_fblock_max = Param(
                initialize=0.433,
                doc="Mechanistic model parameter maximum block fraction [-]",
            )
            self.MECH_k = Param(
                initialize=0.795,
                units=pyunits.kg / pyunits.mol,
                doc="Mechanistic model parameter k [kg/mol]",
            )
            self.MECH_phi_dry = Param(
                initialize=1, doc="Mechanistic model parameter phi in dry case [-]"
            )
            self.MECH_A = Param(
                initialize=1.535,
                units=pyunits.mol / pyunits.kg,
                doc="Mechanistic model parameter A [mol/kg]",
            )
            self.MECH_hoa_wet = Param(
                initialize=-130155.0,
                units=pyunits.J / pyunits.mol,
                doc="Mechanistic model parameter heat of adsorption for wet case [J/mol]",
            )
            self.MECH_n = Param(
                initialize=1.425, doc="Mechanistic model parameter n [-]"
            )
        elif self.config.coadsorption_isotherm == "Stampi-Bombelli":
            self.SB_gamma = Param(
                initialize=-0.137,
                units=pyunits.kg / pyunits.mol,
                doc="Stampi-Bomblli model parameter gamma [kg/mol]",
            )
            self.SB_beta = Param(
                initialize=5.612,
                units=pyunits.kg / pyunits.mol,
                doc="Stampi-Bomblli model parameter beta [kg/mol]",
            )

    def _isotherm_zeolite_13x(self, i, pressure, temperature):
        """
        Method to add isotherm for components.
        Isotherm equation: Extended Sips

        Keyword Arguments:
            i : component
            pressure : partial pressure of components
            temperature : temperature
        """
        T = temperature
        n_inf = {}
        b = {}
        c = {}
        p = {}
        for j in self.adsorbed_components:
            p[j] = pyunits.convert(pressure[j], to_units=pyunits.bar)
            n_inf[j] = self.n_ref[j] * exp(self.X[j] * (T / self.temperature_ref - 1))
            b[j] = self.b0[j] * exp(
                pyunits.convert(self.Qb[j], to_units=pyunits.J / pyunits.mol)
                / constants.gas_constant
                / T
            )
            c[j] = self.c_ref[j] + self.alpha[j] * (T / self.temperature_ref - 1)
        loading = (
            n_inf[i]
            * (b[i] * p[i]) ** c[i]
            / (1 + sum((b[k] * p[k]) ** c[k] for k in self.adsorbed_components))
        )
        return loading

    def _isotherm_netl_sorbent(self, i, pressure, temperature):
        """
        Method to add isotherm for components.
        Isotherm equation: Extended Toth model with physical and chemical terms combined

        Keyword Arguments:
            i : component
            pressure : partial pressure of components
            temperature : temperature
        """
        T = temperature
        p = pyunits.convert(pressure[i], to_units=pyunits.bar)
        if i == "CO2":
            ns_chem = self.q0_chem[i] * exp(
                self.X_chem[i] * (1 - self.temperature_ref / T)
            )
            ns_phys = self.q0_phys[i] * exp(
                self.X_phys[i] * (1 - self.temperature_ref / T)
            )
            b_chem = self.b0_chem[i] * exp(
                pyunits.convert(self.E_chem[i], to_units=pyunits.J / pyunits.mol)
                / constants.gas_constant
                / self.temperature_ref
                * (self.temperature_ref / T - 1)
            )
            b_phys = self.b0_phys[i] * exp(
                pyunits.convert(self.E_phys[i], to_units=pyunits.J / pyunits.mol)
                / constants.gas_constant
                / self.temperature_ref
                * (self.temperature_ref / T - 1)
            )
            t_chem = self.t0_chem[i] + self.alpha_chem[i] * (
                1 - self.temperature_ref / T
            )
            t_phys = self.t0_phys[i] + self.alpha_phys[i] * (
                1 - self.temperature_ref / T
            )
            loading_dry = ns_chem * b_chem * p / (1 + (b_chem * p) ** t_chem) ** (
                1 / t_chem
            ) + ns_phys * b_phys * p / (1 + (b_phys * p) ** t_phys) ** (1 / t_phys)
            log_p = log(p)
            f_en = -0.0048 * log_p**3 - 0.0364 * log_p**2 - 0.0963 * log_p + 1.0226
            loading = loading_dry * f_en
            return loading
        elif i == "H2O":  # GAB model
            c_g0 = 6.86
            dh_c = -5088.0
            k_0 = 2.27
            dh_k = -3443.0
            c_m0 = 0.0208
            cm_beta = 1797
            p_vap = 133 * exp(20.386 - 5132 / T)
            rh = pressure[i] / p_vap
            # rh_limit = min(0.95, rh)
            rh_limit = 0.5 * (rh + 0.95 - sqrt((rh - 0.95) * (rh - 0.95) + 1e-10))
            # rh_limit2 = max(0, rh_limit)
            rh_limit2 = 0.5 * (rh_limit + sqrt(rh_limit * rh_limit + 1e-10))
            c_g = c_g0 * exp(dh_c / constants.gas_constant / T)
            k_ads = k_0 * exp(dh_k / constants.gas_constant / T)
            c_m = c_m0 * exp(cm_beta / T)
            k_ads_rh = k_ads * rh_limit2
            loading = c_m * c_g * k_ads_rh / (1 - k_ads_rh) / (1 + (c_g - 1) * k_ads_rh)
            return loading
        else:
            return 0.0

    # =========================================================================
    def _apply_transformation(self):
        """
        Method to apply DAE transformation to the Control Volume length domain.
        Transformation applied will be based on the Control Volume
        configuration arguments.
        """
        if self.config.finite_elements is None:
            raise ConfigurationError(
                "{} was not provided a value for the finite_elements"
                " configuration argument. Please provide a valid value.".format(
                    self.name
                )
            )

        if self.config.transformation_method == "dae.finite_difference":
            self.discretizer = TransformationFactory(self.config.transformation_method)
            self.discretizer.apply_to(
                self,
                wrt=self.length_domain,
                nfe=self.config.finite_elements,
                scheme=self.config.transformation_scheme,
            )
        elif self.config.transformation_method == "dae.collocation":
            self.discretizer = TransformationFactory(self.config.transformation_method)
            self.discretizer.apply_to(
                self,
                wrt=self.length_domain,
                nfe=self.config.finite_elements,
                ncp=self.config.collocation_points,
                scheme=self.config.transformation_scheme,
            )

    def _make_performance(self):
        """
        Constraints for unit model.

        Args:
            None

        Returns:
            None
        """
        # Joule heating rate W/m^3 if Joule heating is considered
        if self.config.has_joule_heating:
            self.joule_heating_rate = Var(
                self.flowsheet().time,
                self.length_domain,
                initialize=0,
                doc="Volumetric Joule heating rate in W/m3",
                units=pyunits.W / pyunits.m**3,
            )

        # Microwave heating rate W/m^3 if microwave heating is considered
        if self.config.has_microwave_heating:
            self.mass_frac_active = Param(
                initialize=0.1,
                mutable=True,
                doc="Mass fraction of active part of sorbent",
            )
            # lumped heat transfer coefficient
            # product of heat transfer area per kg of total solid material
            # and the htc of W/m^2/K
            self.htc_between_solid_phases = Param(
                initialize=1,
                mutable=True,
                units=pyunits.W / pyunits.kg / pyunits.K,
                doc="Conductive heat transfer coefficient between active and inactive sorbent phases",
            )
            self.microwave_heating_rate_active = Var(
                self.flowsheet().time,
                self.length_domain,
                initialize=0,
                doc="Volumetric microwave heating rate for active part in W/m3",
                units=pyunits.W / pyunits.m**3,
            )
            self.microwave_heating_rate_inactive = Var(
                self.flowsheet().time,
                self.length_domain,
                initialize=0,
                doc="Volumetric microwave heating rate for inactive part in W/m3",
                units=pyunits.W / pyunits.m**3,
            )
            self.solid_temperature_active = Var(
                self.flowsheet().time,
                self.length_domain,
                domain=Reals,
                initialize=300,
                doc="Temperature of active part of solid sorbent",
                units=pyunits.K,
            )
            self.solid_energy_holdup_active = Var(
                self.flowsheet().time,
                self.length_domain,
                initialize=1,
                doc="energy holdup for active part of solid including adsorbates ",
                units=pyunits.J / pyunits.m,
            )
            self.heat_transfer_rate_active_to_inactive = Var(
                self.flowsheet().time,
                self.length_domain,
                initialize=1,
                doc="Heat transfer rate between to solid phases per unit bed length",
                units=pyunits.W / pyunits.m,
            )

        # Bed cross section area
        self.bed_area = Var(
            domain=Reals,
            initialize=1,
            doc="Reactor cross-sectional area",
            units=pyunits.m**2,
        )

        # Gas phase specific variables
        self.velocity_superficial_gas = Var(
            self.flowsheet().time,
            self.length_domain,
            domain=Reals,
            initialize=0.05,
            doc="Gas superficial velocity",
            units=pyunits.m / pyunits.s,
        )

        # Dimensionless numbers, mass and heat transfer coefficients
        self.Sc_number = Var(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            domain=Reals,
            initialize=1.0,
            doc="Schmidt number",
            units=pyunits.dimensionless,
        )

        self.Re_number = Var(
            self.flowsheet().time,
            self.length_domain,
            domain=Reals,
            initialize=100.0,
            doc="Reynolds number",
            units=pyunits.dimensionless,
        )

        self.Nu_number = Var(
            self.flowsheet().time,
            self.length_domain,
            domain=Reals,
            initialize=10.0,
            doc="Nusselt number",
            units=pyunits.dimensionless,
        )

        self.Sh_number = Var(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            domain=Reals,
            initialize=10.0,
            doc="Sherwood number",
            units=pyunits.dimensionless,
        )
        # Note: Pr number is a property of gas phase

        self.RH = Var(
            self.flowsheet().time,
            self.length_domain,
            initialize=0.1,
            doc="Relative humidity",
            units=pyunits.dimensionless,
        )

        self.gas_solid_htc = Var(
            self.flowsheet().time,
            self.length_domain,
            domain=Reals,
            initialize=1.0,
            doc="Gas-solid heat transfer coefficient",
            units=pyunits.m / pyunits.s / pyunits.K / pyunits.m**2,
        )

        self.kc_film = Var(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            domain=Reals,
            initialize=1.0,
            doc="film diffusion mass transfer coefficient",
            units=pyunits.m / pyunits.s,
        )

        self.mole_frac_comp_surface = Var(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            domain=Reals,
            initialize=0.1,
            doc="mole fraction of adsorbed species at sorbent external surface",
            units=pyunits.dimensionless,
        )

        self.heat_solid_to_gas = Var(
            self.flowsheet().time,
            self.length_domain,
            domain=Reals,
            initialize=1.0,
            doc="heat transfer rate per length from solid to gas",
            units=pyunits.W / pyunits.m,
        )

        self.solid_temperature = Var(
            self.flowsheet().time,
            self.length_domain,
            domain=Reals,
            initialize=300,
            doc="Solid phase temperature",
            units=pyunits.K,
        )

        self.adsorbate_loading = Var(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            domain=Reals,
            initialize=1.0,
            doc="Loading of adsorbed species",
            units=pyunits.mol / pyunits.kg,
        )

        self.adsorbate_loading_equil = Var(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            domain=Reals,
            initialize=1.0,
            doc="Loading of adsorbed species if in equilibrium",
            units=pyunits.mol / pyunits.kg,
        )

        self.adsorbate_holdup = Var(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            domain=Reals,
            initialize=1.0,
            doc="Adsorbate holdup per unit length",
            units=pyunits.mol / pyunits.m,
        )

        self.solid_energy_holdup = Var(
            self.flowsheet().time,
            self.length_domain,
            initialize=1,
            doc="Solid phase energy holdup",
            units=pyunits.J / pyunits.m,
        )

        if self.config.dynamic:
            self.adsorbate_accumulation = DerivativeVar(
                self.adsorbate_holdup,
                wrt=self.flowsheet().config.time,
                doc="Adsorbate accumulation per unit length",
                units=pyunits.mol / pyunits.m / pyunits.s,
            )

            self.solid_energy_accumulation = DerivativeVar(
                self.solid_energy_holdup,
                initialize=0,
                wrt=self.flowsheet().config.time,
                doc="Solid energy accumulation",
                units=pyunits.W / pyunits.m,
            )

            if self.config.has_microwave_heating:
                self.solid_energy_accumulation_active = DerivativeVar(
                    self.solid_energy_holdup_active,
                    initialize=0,
                    wrt=self.flowsheet().config.time,
                    doc="Active part solid energy accumulation",
                    units=pyunits.W / pyunits.m,
                )

        # =========================================================================
        # Add performance equations
        # ---------------------------------------------------------------------
        # Geometry constraints

        # Bed area
        @self.Constraint(doc="Bed area")
        def bed_area_eqn(b):
            return b.bed_area == (constants.pi * (0.5 * b.bed_diameter) ** 2)

        @self.Expression(doc="Wet surface area per unit reactor length")
        def wet_surface_area_per_length(b):
            if self.config.adsorbent_shape == "monolith":
                return 4 * b.bed_area * b.voidage / b.hd_monolith
            else:
                return 6 * b.bed_area * (1 - b.voidage) / b.particle_dia

        # Area of gas side, and solid side
        @self.Constraint(self.flowsheet().time, self.length_domain, doc="Gas side area")
        def gas_phase_area_constraint(b, t, x):
            return b.gas_phase.area[t, x] == b.bed_area * (
                b.voidage + (1.0 - b.voidage) * b.particle_voidage
            )

        @self.Expression(doc="Solid phase area")
        def solid_phase_area(b):
            return b.bed_area * (1.0 - b.voidage) * (1.0 - b.particle_voidage)

        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            doc="relative humidity constraint",
        )
        def relative_humidity_eqn(b, t, x):
            # Use gas phase temperature instead of solid temperature seems help the convergence
            # Use solid tempreature may cause condensation at the begining of desorption
            T = b.gas_phase.properties[t, x].temperature
            X1 = T / (273.15 * pyunits.K)
            # water saturation pressure based on ALAMO fitted model, which seems better than
            # other models in terms of convergence
            p_vap = (
                -159601176.32580688595772 * X1
                + 25060265.349311158061028 * log(X1)
                + 63357093.373115316033363 * exp(X1)
                + 35809220.168829716742039 * X1**2
                - 36429260.874471694231033 * X1**3
                - 12000607.575006244704127
            ) * pyunits.Pa
            # use mole fraction at external surface
            return (
                b.RH[t, x] * p_vap * 1e-3
                == 1e-3
                * b.mole_frac_comp_surface[t, x, "H2O"]
                * b.gas_phase.properties[t, x].pressure
            )

        # ---------------------------------------------------------------------
        # Hydrodynamic constraints

        # Gas superficial velocity
        @self.Constraint(
            self.flowsheet().time, self.length_domain, doc="Gas superficial velocity"
        )
        def velocity_superficial_gas_eqn(b, t, x):
            return (
                b.velocity_superficial_gas[t, x]
                * b.bed_area
                * b.gas_phase.properties[t, x].dens_mol
                == b.gas_phase.properties[t, x].flow_mol
            )

        @self.Expression(
            self.flowsheet().time, self.length_domain, doc="Gas phase velocity"
        )
        def velocity_gas_phase(b, t, x):
            return b.velocity_superficial_gas[t, x] / b.voidage

        # Gas side pressure drop calculation
        if self.config.has_pressure_change:
            if self.config.adsorbent_shape == "monolith":
                # since the Re for monolith is around 500, use laminar flow pipe correlation f=16/Re (Bird et al)
                @self.Constraint(
                    self.flowsheet().time,
                    self.length_domain,
                    doc="Gas side pressure drop calculation - pipe pressure drop",
                )
                def gas_phase_config_pressure_drop(b, t, x):
                    return (
                        b.gas_phase.deltaP[t, x] * b.Re_number[t, x] * b.hd_monolith
                        == -32
                        * b.gas_phase.properties[t, x].dens_mass
                        * b.velocity_gas_phase[t, x] ** 2
                    )

            else:
                if self.config.pressure_drop_type == "simple_correlation":
                    # Simplified pressure drop
                    @self.Constraint(
                        self.flowsheet().time,
                        self.length_domain,
                        doc="Gas side pressure drop calculation - simplified pressure drop",
                    )
                    def gas_phase_config_pressure_drop(b, t, x):
                        #  0.2/s is a unitted constant in the correlation
                        return b.gas_phase.deltaP[t, x] == -(
                            0.2 / pyunits.s
                        ) * b.velocity_superficial_gas[t, x] * (
                            b.dens_mass_particle_param,
                            -b.gas_phase.properties[t, x].dens_mass,
                        )

                elif self.config.pressure_drop_type == "ergun_correlation":
                    # Ergun equation
                    @self.Constraint(
                        self.flowsheet().time,
                        self.length_domain,
                        doc="Gas side pressure drop calculation -" "Ergun equation",
                    )
                    def gas_phase_config_pressure_drop(b, t, x):
                        return -b.gas_phase.deltaP[t, x] == (
                            1 - b.voidage
                        ) / b.voidage**3 * b.velocity_superficial_gas[
                            t, x
                        ] / b.particle_dia * (
                            150
                            * (1 - b.voidage)
                            * b.gas_phase.properties[t, x].visc_d_phase["Vap"]
                            / b.particle_dia
                            + 1.75
                            * b.gas_phase.properties[t, x].dens_mass
                            * b.velocity_superficial_gas[t, x]
                        )

                else:
                    raise BurntToast(
                        "{} encountered unrecognized argument for "
                        "the pressure drop correlation. Please contact the IDAES"
                        " developers with this bug.".format(self.name)
                    )

        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            doc=""""Equilibrium loading based on gas phase composition""",
        )
        def isotherm_eqn(b, t, x, j):
            pres = {}
            # use external surface mole fractions
            for i in b.adsorbed_components:
                pres[i] = (
                    b.gas_phase.properties[t, x].pressure
                    * b.mole_frac_comp_surface[t, x, i]
                )
            if self.config.has_microwave_heating:
                T = b.solid_temperature_active[t, x]
            else:
                T = b.solid_temperature[t, x]
            if self.config.adsorbent == "Zeolite-13X":
                return b.adsorbate_loading_equil[t, x, j] == self._isotherm_zeolite_13x(
                    j, pres, T
                )
            elif self.config.adsorbent == "netl_sorbent":
                return b.adsorbate_loading_equil[
                    t, x, j
                ] == self._isotherm_netl_sorbent(j, pres, T)
            elif (
                self.config.adsorbent == "Lewatit"
            ):  # Those constraints seem to help convergence
                q_h2o = b.adsorbate_loading_equil[t, x, "H2O"]
                if j == "CO2":
                    if self.config.coadsorption_isotherm == "None":
                        b_ = self.b0 * exp(-self.hoa / constants.gas_constant / T)
                        b_p = b_ * pres[j]
                        tau = self.tau0 + self.alpha * (1 - self.temperature_ref / T)
                        return b.adsorbate_loading_equil[
                            t, x, j
                        ] == self.q0_inf * b_p / (1 + b_p**tau) ** (1 / tau)
                    else:  # consider co-adsorption effect, need water loading
                        if self.config.coadsorption_isotherm == "Mechanistic":
                            fblock = self.MECH_fblock_max * (
                                1 - exp(-((self.MECH_k * q_h2o) ** self.MECH_n))
                            )
                            phi_avialable = 1.0 - fblock
                            exp_term = exp(-self.MECH_A / q_h2o)
                            phi = (
                                self.MECH_phi_dry
                                + (phi_avialable - self.MECH_phi_dry) * exp_term
                            )
                            hoa_ave = (
                                1 - exp_term
                            ) * self.hoa + exp_term * self.MECH_hoa_wet
                            b_ = self.b0 * exp(-hoa_ave / constants.gas_constant / T)
                            b_p = b_ * pres[j]
                            tau = self.tau0 + self.alpha * (
                                1 - self.temperature_ref / T
                            )
                            return b.adsorbate_loading_equil[
                                t, x, j
                            ] * self.MECH_phi_dry == self.q0_inf * b_p * phi / (
                                1 + b_p**tau
                            ) ** (
                                1 / tau
                            )
                        elif self.config.coadsorption_isotherm == "WADST":
                            b_ = self.b0 * exp(-self.hoa / constants.gas_constant / T)
                            b_p = b_ * pres[j]
                            tau = self.tau0 + self.alpha * (
                                1 - self.temperature_ref / T
                            )
                            q_dry = self.q0_inf * b_p / (1 + b_p**tau) ** (1 / tau)
                            b_wet = self.WADST_b0_wet * exp(
                                -self.WADST_hoa_wet / constants.gas_constant / T
                            )
                            b_p_wet = b_wet * pres[j]
                            tau_wet = self.WADST_tau0_wet + self.WADST_alpha_wet * (
                                1 - self.temperature_ref / T
                            )
                            q_wet = (
                                self.WADST_q0_inf_wet
                                * b_p_wet
                                / (1 + b_p_wet**tau_wet) ** (1 / tau_wet)
                            )
                            return (
                                b.adsorbate_loading_equil[t, x, j]
                                == (1 - exp(-self.WADST_A / q_h2o)) * q_dry
                                + exp(-self.WADST_A / q_h2o) * q_wet
                            )
                        elif self.config.coadsorption_isotherm == "Stampi-Bombelli":
                            b_ = (
                                self.b0
                                * exp(-self.hoa / constants.gas_constant / T)
                                * (1 + self.SB_beta * q_h2o)
                            )
                            b_p = b_ * pres[j]
                            tau = self.tau0 + self.alpha * (
                                1 - self.temperature_ref / T
                            )
                            return b.adsorbate_loading_equil[
                                t, x, j
                            ] == self.q0_inf * b_p / (1 + b_p**tau) ** (1 / tau) / (
                                1 - self.SB_gamma * q_h2o
                            )
                        else:  # invalid configuration
                            raise BurntToast(
                                "{} encountered unrecognized argument for "
                                "CO2-H2O co-adsorption isotherm type. Please contact the IDAES"
                                " developers with this bug.".format(self.name)
                            )
                elif j == "H2O":
                    E1 = self.GAB_C - exp(self.GAB_D * T) * pyunits.J / pyunits.mol
                    E2_9 = self.GAB_F + self.GAB_G * T
                    E10 = self.GAB_A + self.GAB_B * T
                    c = exp((E1 - E10) / constants.gas_constant / T)
                    k = exp((E2_9 - E10) / constants.gas_constant / T)
                    rh = b.RH[t, x]
                    # rh_limit = min(0.95, rh) Note that without limiting this, the steam sweep will cause rh>1
                    # and the solver will diverge
                    rh_limit = 0.5 * (
                        rh + 0.95 - sqrt((rh - 0.95) * (rh - 0.95) + 1e-10)
                    )
                    # rh_limits = max(1, rh_limit)
                    rh_limit2 = 0.5 * (rh_limit + sqrt(rh_limit * rh_limit + 1e-10))
                    kx = k * rh_limit2
                    return (
                        b.adsorbate_loading_equil[t, x, j]
                        * (1 - kx)
                        * (1 + (c - 1) * kx)
                        == self.GAB_qm * c * kx
                    )
            else:
                Constraint.Skip

        # Mass transfer term due to adsorption
        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            self.config.property_package.component_list,
            doc=""""Adsoption of the gas phase components onto 
                the solid phase modeled using the Linear Driving Force model""",
        )
        def mass_transfer_eqn(b, t, x, j):
            if j in b.adsorbed_components:
                return b.gas_phase.mass_transfer_term[t, x, "Vap", j] == -(
                    b.kf[j]
                    * (
                        b.adsorbate_loading_equil[t, x, j]
                        - b.adsorbate_loading[t, x, j]
                    )
                    * b.solid_phase_area
                    * b.dens_mass_particle_param
                )
            else:
                return b.gas_phase.mass_transfer_term[t, x, "Vap", j] == 0.0

        # Mass transfer term due to film diffusion
        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            doc=""""Mass transfer rate due to film diffusion""",
        )
        def mass_transfer_film_diffusion_eqn(b, t, x, j):
            return (b.gas_phase.mass_transfer_term[t, x, "Vap", j] == 
                b.kc_film[t, x, j]
                * (
                    b.mole_frac_comp_surface[t, x, j]
                    - b.gas_phase.properties[t, x].mole_frac_comp[j]
                )
                * b.wet_surface_area_per_length
                * b.gas_phase.properties[t, x].pressure
                / b.gas_phase.properties[t, x].temperature
                / constants.gas_constant
            )

        # Enthalpy transfer term due to adsorption, use enthalpy in gas phase
        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            doc=""""Enthalpy flow to the gas phase due to adsorpton/desorption""",
        )
        def enthalpy_transfer_eqn(b, t, x):
            return b.gas_phase.enthalpy_transfer[t, x] == (
                sum(
                    b.gas_phase.mass_transfer_term[t, x, "Vap", j]
                    * b.gas_phase.properties[t, x].enth_mol_phase_comp["Vap", j]
                    for j in b.adsorbed_components
                )
            )

        # heat transfer from solid phase to gas phase
        # Dimensionless numbers, mass and heat transfer coefficients
        # Particle Reynolds number, Nusselt number, etc.
        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            doc="Reynolds number",
        )
        def reynolds_number_eqn(b, t, x):
            if self.config.adsorbent_shape == "particle":
                # Re is calculated based on superficial velocity
                return (
                    b.Re_number[t, x] * b.gas_phase.properties[t, x].visc_d_phase["Vap"]
                    == b.velocity_superficial_gas[t, x]
                    * b.particle_dia
                    * b.gas_phase.properties[t, x].dens_mass
                )
            else:
                # Re is calculated based on velocity in the monolith channel
                return (
                    b.Re_number[t, x] * b.gas_phase.properties[t, x].visc_d_phase["Vap"]
                    == b.velocity_gas_phase[t, x]
                    * b.hd_monolith
                    * b.gas_phase.properties[t, x].dens_mass
                )

        # Particle Nusselt number
        @self.Constraint(
            self.flowsheet().time, self.length_domain, doc="Particle Nusselt number"
        )
        def nusselt_number_particle(b, t, x):
            if self.config.adsorbent_shape == "particle":
                return (
                    b.Nu_number[t, x]
                    == 2.0
                    + 1.1
                    * b.Re_number[t, x] ** 0.3
                    * b.gas_phase.properties[t, x].prandtl_number_phase["Vap"] ** 0.3333
                )
            else:
                # for fully developed laminar flow, use constant Nu (Incropera & DeWitt)
                # currently doubled to help convergence
                return (
                    b.Nu_number[t, x]
                    == 3.66 * 2  # Literature value is 3.66
                    # 0.023 * b.Re_number[t, x]** 0.8
                    # * b.gas_phase.properties[t,x].prandtl_number_phase["Vap"] ** 0.3
                )

        # Gas phase Schmidt number, it is actually a property
        @self.Constraint(
            self.flowsheet().time, self.length_domain, self.adsorbed_components, doc="Gas phase schmidt number"
        )
        def schmidt_number_gas(b, t, x, i):
            if i=="CO2":
                diffusivity = 1.65e-5*pyunits.m**2/pyunits.s
            elif i=="H2O":
                diffusivity = 2.6e-5*pyunits.m**2/pyunits.s
            else:
                diffusivity = 1.5e-5*pyunits.m**2/pyunits.s
            return (
                b.Sc_number[t, x, i]
                == b.gas_phase.properties[t, x].visc_d_phase["Vap"]
                /b.gas_phase.properties[t, x].dens_mass_phase["Vap"]
                /diffusivity
            )

        # Particle Sherwood number
        @self.Constraint(
            self.flowsheet().time, self.length_domain, self.adsorbed_components, doc="Particle Sherwood number"
        )
        def sherwood_number_particle(b, t, x, i):
            if self.config.adsorbent_shape == "particle":
                return (
                    b.Sh_number[t, x, i]
                    == 2.0
                    + 0.552
                    * b.Re_number[t, x] ** 0.5
                    * b.Sc_number[t, x, i] ** 0.3333
                )
            else:
                # for fully developed laminar flow, use constant Nu (Incropera & DeWitt)
                # currently doubled to help convergence
                return (
                    b.Sh_number[t, x, i]
                    == 3.66 * 2  # Literature value is 3.66
                )

        # Gas-solid heat transfer coefficient
        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            doc="Gas-solid heat transfer coefficient",
        )
        def gas_solid_htc_eqn(b, t, x):
            if self.config.adsorbent_shape == "particle":
                return (
                    b.gas_solid_htc[t, x] * b.particle_dia
                    == b.Nu_number[t, x]
                    * b.gas_phase.properties[t, x].therm_cond_phase["Vap"]
                )
            else:
                return (
                    b.gas_solid_htc[t, x] * b.hd_monolith
                    == b.Nu_number[t, x]
                    * b.gas_phase.properties[t, x].therm_cond_phase["Vap"]
                )

        # film mass transfer coefficient
        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            doc="Film mass transfer coefficient",
        )
        def kc_film_eqn(b, t, x, i):
            if i=="CO2":
                diffusivity = 1.65e-5*pyunits.m**2/pyunits.s
            elif i=="H2O":
                diffusivity = 2.6e-5*pyunits.m**2/pyunits.s
            else:
                diffusivity = 1.5e-5*pyunits.m**2/pyunits.s
            if self.config.adsorbent_shape == "particle":
                return (
                    b.kc_film[t, x, i] * b.particle_dia
                    == b.Sh_number[t, x, i]
                    * diffusivity
                )
            else:
                return (
                    b.kc_film[t, x] * b.hd_monolith
                    == b.Sh_number[t, x, i]
                    * diffusivity
                )

        # heat transfer rate from solid phase to gas phase
        # Note: Current code consider the number of particles in a unit bed volume based on volume
        # of porous particles while the original code based on the volume of true solid materials
        # Therefore, the heat transfer area is larger than that in the original code
        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            doc="Solid to gas heat transfer",
        )
        def solid_to_gas_heat_transfer(b, t, x):
            return (
                b.heat_solid_to_gas[t, x]
                == b.gas_solid_htc[t, x]
                * (b.solid_temperature[t, x] - b.gas_phase.properties[t, x].temperature)
                * b.wet_surface_area_per_length
            )

        # gas phase total heat duty
        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            doc="Solid to gas heat transfer",
        )
        def gas_phase_heat_transfer(b, t, x):
            return b.gas_phase.heat[t, x] == b.heat_solid_to_gas[
                t, x
            ] + constants.pi * b.bed_diameter * b.heat_transfer_coeff_gas_wall * (
                b.wall_temperature[t, x] - b.gas_phase.properties[t, x].temperature
            )

        # Solid phase component balance
        # material holdup constraint
        @self.Constraint(
            self.flowsheet().config.time,
            self.length_domain,
            self.adsorbed_components,
            doc="Solid phase adsorbate holdup constraints",
        )
        def adsorbate_holdup_eqn(b, t, x, j):
            return b.adsorbate_holdup[t, x, j] == (
                b.solid_phase_area
                * b.dens_mass_particle_param
                * b.adsorbate_loading[t, x, j]
            )

        # Add component balances of adsorbate
        @self.Constraint(
            self.flowsheet().time,
            self.length_domain,
            self.adsorbed_components,
            doc="Material balances of adsorbates",
        )
        def solid_material_balances(b, t, x, j):
            if self.config.dynamic:
                return b.adsorbate_accumulation[t, x, j] == (
                    -b.gas_phase.mass_transfer_term[t, x, "Vap", j]
                )
            else:
                return 0 == -b.gas_phase.mass_transfer_term[t, x, "Vap", j]

        # Solid phase energy balance
        if self.config.has_microwave_heating:

            @self.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Active part solid phase energy holdup constraints",
            )
            def solid_energy_holdup_active_eqn(b, t, x):
                return b.solid_energy_holdup_active[t, x] == (
                    b.solid_phase_area
                    * b.dens_mass_particle_param
                    * b.cp_mass_param
                    * b.mass_frac_active
                    * (b.solid_temperature_active[t, x] - b.temperature_ref)
                    + sum(
                        b.adsorbate_holdup[t, x, i]
                        * (
                            b.enth_mol_comp_std[i]
                            + b.dh_ads[i]
                            + b.cp_mol_comp_adsorbate[i]
                            * (b.solid_temperature_active[t, x] - b.temperature_ref)
                        )
                        for i in b.adsorbed_components
                    )
                )

            @self.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Solid phase energy holdup constraints",
            )
            def solid_energy_holdup_eqn(b, t, x):
                return b.solid_energy_holdup[t, x] == (
                    b.solid_phase_area
                    * b.dens_mass_particle_param
                    * b.cp_mass_param
                    * (1 - b.mass_frac_active)
                    * (b.solid_temperature[t, x] - b.temperature_ref)
                )

            @self.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Heat transfer rate between active and inactive solid phases",
            )
            def heat_transfer_rate_active_to_inactive_eqn(b, t, x):
                return (
                    b.heat_transfer_rate_active_to_inactive[t, x]
                    == b.htc_between_solid_phases
                    * (b.solid_temperature_active[t, x] - b.solid_temperature[t, x])
                    * b.solid_phase_area
                    * b.dens_mass_particle_param
                )

            @self.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Active part of solid phase energy balances",
            )
            def solid_energy_balances_active(b, t, x):
                if self.config.dynamic:
                    return (
                        b.solid_energy_accumulation_active[t, x]
                        == b.microwave_heating_rate_active[t, x] * b.bed_area
                        - b.gas_phase.enthalpy_transfer[t, x]
                        - b.heat_transfer_rate_active_to_inactive[t, x]
                    )
                else:
                    return (
                        0
                        == b.microwave_heating_rate_active[t, x] * b.bed_area
                        - b.gas_phase.enthalpy_transfer[t, x]
                        - b.heat_transfer_rate_active_to_inactive[t, x]
                    )

            @self.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Inactive part of solid phase energy balances",
            )
            def solid_energy_balances(b, t, x):
                if self.config.has_joule_heating:
                    jheat = b.joule_heating_rate[t, x]
                else:
                    jheat = 0 * pyunits.W / pyunits.m**3
                if self.config.dynamic:
                    return (
                        b.solid_energy_accumulation[t, x]
                        == b.microwave_heating_rate_inactive[t, x] * b.bed_area
                        - b.heat_solid_to_gas[t, x]
                        + jheat * b.bed_area
                        + b.heat_transfer_rate_active_to_inactive[t, x]
                    )
                else:
                    return (
                        0
                        == b.microwave_heating_rate_inactive[t, x] * b.bed_area
                        - b.heat_solid_to_gas[t, x]
                        + jheat * b.bed_area
                        + b.heat_transfer_rate_active_to_inactive[t, x]
                    )

        else:

            @self.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Solid phase energy holdup constraints",
            )
            def solid_energy_holdup_eqn(b, t, x):
                return b.solid_energy_holdup[t, x] == (
                    b.solid_phase_area
                    * b.dens_mass_particle_param
                    * b.cp_mass_param
                    * (b.solid_temperature[t, x] - b.temperature_ref)
                    + sum(
                        b.adsorbate_holdup[t, x, i]
                        * (
                            b.enth_mol_comp_std[i]
                            + b.dh_ads[i]
                            + b.cp_mol_comp_adsorbate[i]
                            * (b.solid_temperature[t, x] - b.temperature_ref)
                        )
                        for i in b.adsorbed_components
                    )
                )

            @self.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Solid phase energy balances",
            )
            def solid_energy_balances(b, t, x):
                if self.config.has_joule_heating:
                    jheat = b.joule_heating_rate[t, x]
                else:
                    jheat = 0 * pyunits.W / pyunits.m**3
                if self.config.dynamic:
                    return (
                        b.solid_energy_accumulation[t, x]
                        == -b.heat_solid_to_gas[t, x]
                        - b.gas_phase.enthalpy_transfer[t, x]
                        + jheat * b.bed_area
                    )
                else:
                    return (
                        0
                        == -b.heat_solid_to_gas[t, x]
                        - b.gas_phase.enthalpy_transfer[t, x]
                        + jheat * b.bed_area
                    )

        @self.Constraint(
            self.flowsheet().config.time,
            self.length_domain,
            doc="Wall energy balances",
        )
        def wall_energy_balances(b, t, x):
            if self.config.dynamic:
                return b.wall_temperature_dt[t, x] * b.cp_wall / 4 * (
                    b.wall_diameter**2 - b.bed_diameter**2
                ) * b.dens_wall == b.bed_diameter * b.heat_transfer_coeff_gas_wall * (
                    b.gas_phase.properties[t, x].temperature - b.wall_temperature[t, x]
                ) + b.wall_diameter * b.heat_transfer_coeff_fluid_wall * (
                    b.fluid_temperature[t] - b.wall_temperature[t, x]
                )
            else:
                return 0 == b.bed_diameter * b.heat_transfer_coeff_gas_wall * (
                    b.gas_phase.properties[t, x].temperature - b.wall_temperature[t, x]
                ) + b.wall_diameter * b.heat_transfer_coeff_fluid_wall * (
                    b.fluid_temperature[t] - b.wall_temperature[t, x]
                )

        @self.Expression(
            self.flowsheet().config.time,
            self.length_domain,
            doc="Heat transfer rate from fluid to wall per bed length",
        )
        def heat_fluid_to_wall(b, t, x):
            return (
                constants.pi
                * b.wall_diameter
                * b.heat_transfer_coeff_fluid_wall
                * (b.fluid_temperature[t] - b.wall_temperature[t, x])
            )

    # =========================================================================
    # Model initialization routine

    def initialize_build(
        blk,
        gas_phase_state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        Initialization routine for 1DFixedBed unit.

        Keyword Arguments:
            gas_phase_state_args : a dict of arguments to be passed to the
                        property package(s) to provide an initial state for
                        initialization (see documentation of the specific
                        property package) (default = None).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None, use
                     default solver options)
            solver : str indicating which solver to use during
                     initialization (default = None, use default solver)

        Returns:
            None
        """

        # Set up logger for initialization and solve
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

        # Create solver
        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # Keep all unit model geometry constraints, derivative_var constraints,
        # and property block constraints active. Additionally, in control
        # volumes - keep conservation linking constraints and
        # holdup calculation (for dynamic flowsheets) constraints active

        # ---------------------------------------------------------------------
        # Initialize thermophysical property constraints
        init_log.info("Initialize Thermophysical Properties")
        # Initialize gas_phase block
        flags = blk.gas_phase.initialize(
            state_args=gas_phase_state_args,
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
        )

        init_log.info_high("Initialization Step 1 Complete.")

        # ---------------------------------------------------------------------
        # Initialize hydrodynamics (gas velocity)
        for t in blk.flowsheet().time:
            for x in blk.length_domain:
                calculate_variable_from_constraint(
                    blk.velocity_superficial_gas[t, x],
                    blk.velocity_superficial_gas_eqn[t, x],
                )
        blk.velocity_superficial_gas_eqn.activate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            results = opt.solve(blk, tee=slc.tee, symbolic_solver_labels=True)
        if check_optimal_termination(results):
            init_log.info_high(
                "Initialization Step 2 {}.".format(idaeslog.condition(results))
            )
        else:
            _log.warning("{} Initialization Step 2 Failed.".format(blk.name))
        blk.gas_phase.release_state(flags)

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        # scale some variables
        if hasattr(self, "bed_height"):
            sf = 1 / value(self.bed_height)
            iscale.set_scaling_factor(self.bed_height, sf)

        if hasattr(self, "bed_diameter"):
            sf = 1 / value(self.bed_diameter)
            iscale.set_scaling_factor(self.bed_diameter, sf)

        if hasattr(self, "hd_monolith"):
            sf = 1 / value(self.hd_monolith)
            iscale.set_scaling_factor(self.hd_monolith, sf)

        if hasattr(self, "particle_dia"):
            sf = 1 / value(self.particle_dia)
            iscale.set_scaling_factor(self.particle_dia, sf)

        if hasattr(self, "wall_diameter"):
            sf = 1 / value(self.wall_diameter)
            iscale.set_scaling_factor(self.wall_diameter, sf)

        if hasattr(self, "bed_area"):
            sf = 1 / value(constants.pi * (0.5 * self.bed_diameter) ** 2)
            iscale.set_scaling_factor(self.bed_area, sf)

        if hasattr(self.gas_phase, "area"):
            sf = iscale.get_scaling_factor(self.bed_area)
            iscale.set_scaling_factor(self.gas_phase.area, 2 * sf)

        if hasattr(self, "wall_temperature"):
            iscale.set_scaling_factor(self.wall_temperature, 1e-2)

        if hasattr(self, "wall_temperature_dt"):
            iscale.set_scaling_factor(self.wall_temperature_dt, 1)

        if hasattr(self, "fluid_temperature"):
            iscale.set_scaling_factor(self.fluid_temperature, 1e-2)

        if hasattr(self, "solid_temperature"):
            iscale.set_scaling_factor(self.solid_temperature, 1e-2)

        if hasattr(self, "velocity_superficial_gas"):
            iscale.set_scaling_factor(self.velocity_superficial_gas, 10)

        # Re number assuming velocity of 1 m/s and density of 1 kg/m^3 and viscosity of 1e-5
        if hasattr(self, "Re_number"):
            if self.config.adsorbent_shape == "particle":
                sf = 1 / value(self.particle_dia)
            else:
                sf = 1 / value(self.hd_monolith)
            for (t, x), v in self.Re_number.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 1e-5 * sf)

        # Nu number > 2, close to 10
        if hasattr(self, "Nu_number"):
            for (t, x), v in self.Nu_number.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 0.1)

        # thermal conductivity around 0.025 W/m/K and Nu around 2
        if hasattr(self, "gas_solid_htc"):
            if self.config.adsorbent_shape == "particle":
                sf = value(self.particle_dia)
            else:
                sf = value(self.hd_monolith)
            for (t, x), v in self.gas_solid_htc.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 4 * sf)

        if hasattr(self, "heat_solid_to_gas"):
            for (t, x), v in self.heat_solid_to_gas.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 0.01)

        if hasattr(self, "adsorbate_loading"):
            for (t, x, i), v in self.adsorbate_loading.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 1)

        if hasattr(self, "adsorbate_loading_equil"):
            for (t, x, i), v in self.adsorbate_loading_equil.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 1)

        if hasattr(self, "adsorbate_holdup"):
            for (t, x, i), v in self.adsorbate_holdup.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 1)

        if hasattr(self.gas_phase, "deltaP"):
            for (t, x), v in self.gas_phase.deltaP.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 1e-2)

        if hasattr(self.gas_phase, "enthalpy_transfer"):
            for (t, x), v in self.gas_phase.enthalpy_transfer.items():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 1e-3)

        for (t, x, p, i), v in self.gas_phase.mass_transfer_term.items():
            iscale.set_scaling_factor(v, 1e5)

        for (t, x), v in self.gas_phase.heat.items():
            iscale.set_scaling_factor(v, 1e-1)

        for (t, x), v in self.solid_energy_holdup.items():
            iscale.set_scaling_factor(v, 1e-4)

        # Scale some constraints
        if hasattr(self, "bed_area_eqn"):
            for c in self.bed_area_eqn.values():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.bed_area), overwrite=False
                )

        # need to have a better scaling
        if hasattr(self, "velocity_superficial_gas_eqn"):
            for (t, x), c in self.velocity_superficial_gas_eqn.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.gas_phase.properties[t, x].flow_mol),
                    overwrite=False,
                )

        if hasattr(self, "gas_phase_config_pressure_drop"):
            for (t, x), c in self.gas_phase_config_pressure_drop.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.gas_phase.deltaP[t, x]),
                    overwrite=False,
                )

        if hasattr(self, "reynolds_number_eqn"):
            for (t, x), c in self.reynolds_number_eqn.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.Re_number[t, x]),
                    overwrite=False,
                )

        if hasattr(self, "nusselt_number_particle"):
            for (t, x), c in self.nusselt_number_particle.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.Nu_number[t, x]),
                    overwrite=False,
                )

        if hasattr(self, "mass_transfer_eqn"):
            for (t, x, i), c in self.mass_transfer_eqn.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(
                        self.gas_phase.mass_transfer_term[t, x, "Vap", i]
                    ),
                    overwrite=False,
                )

        if hasattr(self, "enthalpy_transfer_eqn"):
            for (t, x), c in self.enthalpy_transfer_eqn.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.gas_phase.enthalpy_transfer[t, x]),
                    overwrite=False,
                )

        if hasattr(self, "gas_solid_htc_eqn"):
            for (t, x), c in self.gas_solid_htc_eqn.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.gas_solid_htc[t, x]) * 100,
                    overwrite=False,
                )

        if hasattr(self, "gas_phase_heat_transfer"):
            for (t, x), c in self.gas_phase_heat_transfer.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.gas_phase.heat[t, x]),
                    overwrite=False,
                )

        if hasattr(self, "adsorbate_holdup_eqn"):
            for (t, x, j), c in self.adsorbate_holdup_eqn.items():
                sf = iscale.get_scaling_factor(self.adsorbate_holdup[t, x, j])
                iscale.constraint_scaling_transform(c, sf, overwrite=False)

        if hasattr(self, "solid_energy_holdup_eqn"):
            for (t, x), c in self.solid_energy_holdup_eqn.items():
                sf = iscale.get_scaling_factor(self.solid_energy_holdup[t, x])
                iscale.constraint_scaling_transform(c, sf, overwrite=False)

        if hasattr(self, "solid_material_balances"):
            for (t, x, i), c in self.solid_material_balances.items():
                sf = iscale.get_scaling_factor(
                    self.gas_phase.mass_transfer_term[t, x, "Vap", i]
                )
                iscale.constraint_scaling_transform(c, sf, overwrite=False)

        if hasattr(self, "solid_energy_balance"):
            for (t, x), c in self.solid_energy_balance.items():
                sf = iscale.get_scaling_factor(self.heat_solid_to_gas[t, x])
                iscale.constraint_scaling_transform(c, sf, overwrite=False)

        if hasattr(self, "wall_energy_balance"):
            for (t, x), c in self.wall_energy_balance.items():
                sf = value(1 / self.wall_diameter / self.heat_transfer_coeff_fluid_wall)
                iscale.constraint_scaling_transform(c, sf, overwrite=False)

        if hasattr(self, "relative_humidity_eqn"):
            for (t, x), c in self.relative_humidity_eqn.items():
                sf = 1
                iscale.constraint_scaling_transform(c, sf, overwrite=False)

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {"Gas Inlet": self.gas_inlet, "Gas Outlet": self.gas_outlet},
            time_point=time_point,
        )

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        var_dict["Bed Height"] = self.bed_height
        var_dict["Bed Area"] = self.bed_area
        var_dict["Gas Inlet Velocity"] = self.velocity_superficial_gas[time_point, 0]
        var_dict["Gas Outlet Velocity"] = self.velocity_superficial_gas[time_point, 1]
        return {"vars": var_dict}

    def set_initial_condition(self):
        if self.config.dynamic is True:
            self.solid_energy_accumulation[:, :].value = 0
            self.adsorbate_accumulation[:, :, :].value = 0
            self.solid_energy_accumulation[0, :].fix(0)
            self.adsorbate_accumulation[0, :, :].fix(0)
            if self.config.has_microwave_heating:
                self.solid_energy_accumulation_active[:, :].value = 0
                self.solid_energy_accumulation_active[0, :].fix(0)
            self.gas_phase.material_accumulation[:, :, :, :].value = 0
            self.gas_phase.energy_accumulation[:, :, :].value = 0
            self.gas_phase.material_accumulation[0, :, :, :].fix(0)
            self.gas_phase.energy_accumulation[0, :, :].fix(0)
            self.wall_temperature_dt[:, :].value = 0
            self.wall_temperature_dt[0, :].fix(0)
