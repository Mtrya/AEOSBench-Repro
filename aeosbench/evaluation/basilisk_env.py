"""Minimal Basilisk environment wrapper for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import os
from typing import Iterable

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from Basilisk.architecture import messaging
from Basilisk.architecture.messaging import VehicleConfigMsg, VehicleConfigMsgPayload
from Basilisk.fswAlgorithms.locationPointing import locationPointing
from Basilisk.fswAlgorithms.mrpFeedback import mrpFeedback
from Basilisk.fswAlgorithms.rwMotorTorque import rwMotorTorque
from Basilisk.simulation.eclipse import Eclipse
from Basilisk.simulation.groundLocation import GroundLocation
from Basilisk.simulation.groundMapping import GroundMapping
from Basilisk.simulation.ReactionWheelPower import ReactionWheelPower
from Basilisk.simulation.reactionWheelStateEffector import ReactionWheelStateEffector
from Basilisk.simulation.simpleBattery import SimpleBattery
from Basilisk.simulation.simpleNav import SimpleNav
from Basilisk.simulation.simplePowerSink import SimplePowerSink
from Basilisk.simulation.simpleSolarPanel import SimpleSolarPanel
from Basilisk.simulation.spacecraft import HubEffector, Spacecraft
from Basilisk.utilities import macros, orbitalMotion, unitTestSupport
from Basilisk.utilities.simIncludeGravBody import gravBodyFactory
from Basilisk.utilities.simIncludeRW import rwFactory
from Basilisk.utilities.SimulationBaseClass import SimBaseClass

from aeosbench.constants import INTERVAL, MU_EARTH, RADIUS_EARTH, TIMESTAMP
from aeosbench.data import (
    Battery,
    Constellation,
    MRPControl,
    Orbit,
    ReactionWheel,
    Satellite,
    Sensor,
    SensorType,
    SolarPanel,
    TaskSet,
)
from aeosbench.data.geodetics import lla_to_pcpf

IDENTITY_MATRIX_3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
UNIT_VECTOR_Z = [0.0, 0.0, 1.0]


def _datetime_to_basilisk(date_object: datetime) -> str:
    return date_object.strftime("%Y %b %d %H:%M:%S.%f (UTC)")


@dataclass
class Timer:
    time: int = 0

    def step(self) -> None:
        self.time += 1


class BasiliskSatellite:
    def __init__(
        self,
        simulator: SimBaseClass,
        process: object,
        gravity_factory: gravBodyFactory,
        spice_object: object,
        satellite: Satellite,
    ) -> None:
        self._id = satellite.id_
        self._orbit_id = satellite.orbit_id
        self._task_name = f"task-{self._id}"
        process.addTask(simulator.CreateNewTask(self._task_name, macros.sec2nano(INTERVAL)))

        self._spacecraft = Spacecraft()
        self._spacecraft.ModelTag = f"spacecraft-{self._id}"
        hub: HubEffector = self._spacecraft.hub
        hub.r_CN_NInit, hub.v_CN_NInit = satellite.rv
        hub.mHub = satellite.mass
        hub.r_BcB_B = np.reshape(satellite.center_of_mass, (-1, 1))
        hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(satellite.inertia)
        hub.sigma_BNInit = np.reshape(satellite.mrp_attitude_bn, (-1, 1))
        simulator.AddModelToTask(self._task_name, self._spacecraft)

        self._eclipse = Eclipse()
        self._eclipse.ModelTag = f"eclipse-{self._id}"
        simulator.AddModelToTask(self._task_name, self._eclipse)

        self._solar_panel = SimpleSolarPanel()
        self._solar_panel.ModelTag = f"solar_panel-{self._id}"
        self._solar_panel.setPanelParameters(
            satellite.solar_panel.direction,
            satellite.solar_panel.area,
            satellite.solar_panel.efficiency,
        )
        simulator.AddModelToTask(self._task_name, self._solar_panel)

        self._power_sink = SimplePowerSink()
        self._power_sink.ModelTag = f"power_sink-{self._id}"
        self._power_sink.powerStatus = satellite.sensor.enabled
        self._power_sink.nodePowerOut = -satellite.sensor.power
        simulator.AddModelToTask(self._task_name, self._power_sink)

        self._battery = SimpleBattery()
        self._battery.ModelTag = f"battery-{self._id}"
        self._battery.storageCapacity = satellite.battery.capacity
        self._battery.storedCharge_Init = satellite.battery.percentage * satellite.battery.capacity
        simulator.AddModelToTask(self._task_name, self._battery)

        self._simple_navigation = SimpleNav()
        self._simple_navigation.ModelTag = f"simple_navigation-{self._id}"
        simulator.AddModelToTask(self._task_name, self._simple_navigation)

        self._pointing_location = GroundLocation()
        self._pointing_location.ModelTag = f"pointing_location-{self._id}"
        self._pointing_location.planetRadius = RADIUS_EARTH
        self._pointing_location.minimumElevation = 0
        simulator.AddModelToTask(self._task_name, self._pointing_location)

        self._pointing_guide = locationPointing()
        self._pointing_guide.ModelTag = f"pointing_guide-{self._id}"
        self._pointing_guide.pHat_B = UNIT_VECTOR_Z
        simulator.AddModelToTask(self._task_name, self._pointing_guide)

        self._ground_mapping = GroundMapping()
        self._ground_mapping.ModelTag = f"ground_mapping-{self._id}"
        self._ground_mapping.minimumElevation = 0
        self._ground_mapping.maximumRange = 1e9
        self._ground_mapping.cameraPos_B = [0, 0, 0]
        self._ground_mapping.nHat_B = UNIT_VECTOR_Z
        self._ground_mapping.halfFieldOfView = math.radians(satellite.sensor.half_field_of_view)
        simulator.AddModelToTask(self._task_name, self._ground_mapping)

        self._rw_factory = rwFactory()
        for index, wheel in enumerate(satellite.reaction_wheels):
            self._rw_factory.create(
                wheel.rw_type,
                wheel.rw_direction,
                maxMomentum=wheel.max_momentum,
                Omega=wheel.rw_speed_init,
                RWModel=messaging.BalancedWheels,
                label=self.reaction_wheel_id(index),
            )

        self._mrp_control = mrpFeedback()
        self._mrp_control.ModelTag = f"mrpFeedback-{self._id}"
        self._mrp_control.K = satellite.mrp_control.k
        self._mrp_control.Ki = satellite.mrp_control.ki
        self._mrp_control.P = satellite.mrp_control.p
        self._mrp_control.integralLimit = satellite.mrp_control.integral_limit
        config_payload = VehicleConfigMsgPayload()
        config_payload.ISCPntB_B = satellite.inertia
        config_msg = VehicleConfigMsg()
        config_msg.write(config_payload)
        self._mrp_control.vehConfigInMsg.subscribeTo(config_msg)
        self._config_msg = config_msg
        simulator.AddModelToTask(self._task_name, self._mrp_control)

        self._rw_motor_torque = rwMotorTorque()
        self._rw_motor_torque.ModelTag = f"rw_motor_torque-{self._id}"
        self._rw_motor_torque.controlAxes_B = IDENTITY_MATRIX_3
        simulator.AddModelToTask(self._task_name, self._rw_motor_torque)

        self._rw_state_effector = ReactionWheelStateEffector()
        self._rw_state_effector.ModelTag = f"rw_state_effector-{self._id}"
        simulator.AddModelToTask(self._task_name, self._rw_state_effector)

        self._rw_power_list: list[ReactionWheelPower] = []
        for index, wheel in enumerate(satellite.reaction_wheels):
            rw_power = ReactionWheelPower()
            rw_power.ModelTag = f"rw_power-{self._id}-{index}"
            rw_power.basePowerNeed = wheel.power
            rw_power.mechToElecEfficiency = wheel.efficiency
            self._rw_power_list.append(rw_power)
            simulator.AddModelToTask(self._task_name, rw_power)

        self._sensor_type = satellite.sensor.type_
        self._reaction_wheels = satellite.reaction_wheels

        earth_state = spice_object.planetStateOutMsgs[0]
        sun_state = spice_object.planetStateOutMsgs[1]
        gravity_factory.addBodiesTo(self._spacecraft)
        self._eclipse.addSpacecraftToModel(self._spacecraft.scStateOutMsg)
        self._eclipse.addPlanetToModel(earth_state)
        self._eclipse.sunInMsg.subscribeTo(sun_state)
        self._solar_panel.stateInMsg.subscribeTo(self._spacecraft.scStateOutMsg)
        self._solar_panel.sunEclipseInMsg.subscribeTo(self._eclipse.eclipseOutMsgs[0])
        self._solar_panel.sunInMsg.subscribeTo(sun_state)
        self._battery.addPowerNodeToModel(self._solar_panel.nodePowerOutMsg)
        self._battery.addPowerNodeToModel(self._power_sink.nodePowerOutMsg)
        for rw_power in self._rw_power_list:
            self._battery.addPowerNodeToModel(rw_power.nodePowerOutMsg)
        self._simple_navigation.scStateInMsg.subscribeTo(self._spacecraft.scStateOutMsg)
        self._pointing_location.planetInMsg.subscribeTo(earth_state)
        self._pointing_location.addSpacecraftToModel(self._spacecraft.scStateOutMsg)
        self._pointing_guide.scAttInMsg.subscribeTo(self._simple_navigation.attOutMsg)
        self._pointing_guide.scTransInMsg.subscribeTo(self._simple_navigation.transOutMsg)
        self._pointing_guide.locationInMsg.subscribeTo(self._pointing_location.currentGroundStateOutMsg)
        self._ground_mapping.scStateInMsg.subscribeTo(self._spacecraft.scStateOutMsg)
        self._ground_mapping.planetInMsg.subscribeTo(earth_state)
        self._rw_factory.addToSpacecraft(self._spacecraft.ModelTag, self._rw_state_effector, self._spacecraft)
        self._rw_params_message = self._rw_factory.getConfigMessage()
        self._mrp_control.guidInMsg.subscribeTo(self._pointing_guide.attGuidOutMsg)
        self._mrp_control.rwParamsInMsg.subscribeTo(self._rw_params_message)
        self._mrp_control.rwSpeedsInMsg.subscribeTo(self._rw_state_effector.rwSpeedOutMsg)
        self._rw_motor_torque.vehControlInMsg.subscribeTo(self._mrp_control.cmdTorqueOutMsg)
        self._rw_motor_torque.rwParamsInMsg.subscribeTo(self._rw_params_message)
        self._rw_state_effector.rwMotorCmdInMsg.subscribeTo(self._rw_motor_torque.rwMotorTorqueOutMsg)
        for rw_power, rw_out_message in zip(self._rw_power_list, self._rw_state_effector.rwOutMsgs):
            rw_power.rwStateInMsg.subscribeTo(rw_out_message)

    def add_task_locations(self, taskset: TaskSet) -> None:
        for task in taskset:
            location = lla_to_pcpf(
                math.radians(task.coordinate[0]),
                math.radians(task.coordinate[1]),
                0.0,
            )
            self._ground_mapping.addPointToModel(np.array(location))

    def reaction_wheel_id(self, index: int) -> str:
        return f"{index}RW{self._id}"

    def toggle(self) -> None:
        self._power_sink.powerStatus = 1 - self._power_sink.powerStatus

    def guide_attitude(self, target_location: tuple[float, float] | None) -> None:
        if target_location is None:
            self._pointing_location.specifyLocationPCPF([[0.0], [0.0], [0.0]])
            return
        self._pointing_location.specifyLocation(
            math.radians(target_location[0]),
            math.radians(target_location[1]),
            0,
        )

    def to_satellite(self) -> Satellite:
        hub = self._spacecraft.hub
        inertia = tuple(np.reshape(hub.IHubPntBc_B, -1).tolist())
        mass = float(hub.mHub)
        center_of_mass = tuple(np.squeeze(hub.r_BcB_B).tolist())
        orbital_elements = self.orbital_elements
        orbit = Orbit(
            id_=self._orbit_id,
            eccentricity=float(orbital_elements.e),
            semi_major_axis=float(orbital_elements.a),
            inclination=float(orbital_elements.i / macros.D2R),
            right_ascension_of_the_ascending_node=float(orbital_elements.Omega / macros.D2R),
            argument_of_perigee=float(orbital_elements.omega / macros.D2R),
        )
        solar_panel = SolarPanel(
            direction=tuple(np.squeeze(self._solar_panel.nHat_B).tolist()),
            area=float(self._solar_panel.panelArea),
            efficiency=float(self._solar_panel.panelEfficiency),
        )
        sensor = Sensor(
            type_=self._sensor_type,
            enabled=bool(self._power_sink.powerStatus),
            half_field_of_view=float(np.rad2deg(self._ground_mapping.halfFieldOfView)),
            power=float(-self._power_sink.nodePowerOut),
        )
        battery = Battery(
            capacity=float(self._battery.storageCapacity),
            percentage=float(self._battery.batPowerOutMsg.read().storageLevel / self._battery.storageCapacity),
        )
        reaction_wheels = tuple(
            ReactionWheel(
                rw_type=wheel.rw_type,
                rw_direction=wheel.rw_direction,
                max_momentum=wheel.max_momentum,
                rw_speed_init=float(self._rw_factory.rwList[self.reaction_wheel_id(index)].Omega / macros.rpm2radsec),
                power=wheel.power,
                efficiency=wheel.efficiency,
            )
            for index, wheel in enumerate(self._reaction_wheels)
        )
        mrp_control = MRPControl(
            k=float(self._mrp_control.K),
            ki=float(self._mrp_control.Ki),
            p=float(self._mrp_control.P),
            integral_limit=float(self._mrp_control.integralLimit),
        )
        mrp_attitude_bn = tuple(np.array(self.mrp_attitude_bn).squeeze().tolist())
        return Satellite(
            id_=self._id,
            inertia=inertia,  # type: ignore[arg-type]
            mass=mass,
            center_of_mass=center_of_mass,  # type: ignore[arg-type]
            orbit_id=self._orbit_id,
            orbit=orbit,
            solar_panel=solar_panel,
            sensor=sensor,
            battery=battery,
            reaction_wheels=reaction_wheels,  # type: ignore[arg-type]
            mrp_control=mrp_control,
            true_anomaly=float(self.orbital_elements.f / macros.D2R),
            mrp_attitude_bn=mrp_attitude_bn,  # type: ignore[arg-type]
        )

    @property
    def orbital_elements(self) -> object:
        spacecraft_state = self.spacecraft_state
        return orbitalMotion.rv2elem(
            MU_EARTH,
            np.array(spacecraft_state.r_CN_N),
            np.array(spacecraft_state.v_CN_N),
        )

    @property
    def spacecraft_state(self) -> object:
        return self._spacecraft.scStateOutMsg.read()

    @property
    def mrp_attitude_bn(self) -> Iterable[float]:
        return self._spacecraft.scStateOutMsg.read().sigma_BN

    @property
    def power_sink(self) -> SimplePowerSink:
        return self._power_sink

    @property
    def ground_mapping(self) -> GroundMapping:
        return self._ground_mapping

    @property
    def sensor_type(self) -> SensorType:
        return self._sensor_type


class BasiliskEnvironment:
    def __init__(self, *, constellation: Constellation, taskset: TaskSet, standard_time_init: str = TIMESTAMP) -> None:
        self.timer = Timer(0)
        self._simulator = SimBaseClass()
        self._taskset = taskset
        process = self._simulator.CreateNewProcess(__file__)
        process.addTask(self._simulator.CreateNewTask("task_environment", macros.sec2nano(INTERVAL)))
        gravity_factory = gravBodyFactory()
        earth = gravity_factory.createEarth()
        earth.isCentralBody = True
        gravity_factory.createSun()
        date_object = datetime.strptime(standard_time_init, "%Y%m%d%H%M%S")
        spice_object = gravity_factory.createSpiceInterface(time=_datetime_to_basilisk(date_object))
        spice_object.zeroBase = "Earth"
        self._simulator.AddModelToTask("task_environment", spice_object)
        self._spice_object = spice_object

        self._satellites = [
            BasiliskSatellite(self._simulator, process, gravity_factory, spice_object, satellite)
            for satellite in constellation.values()
        ]
        for satellite in self._satellites:
            satellite.add_task_locations(taskset)

        self._simulator.InitializeSimulation()
        self._simulator.ConfigureStopTime(0)
        self._simulator.ExecuteSimulation()

    @property
    def num_satellites(self) -> int:
        return len(self._satellites)

    def get_constellation(self) -> Constellation:
        satellites = [satellite.to_satellite() for satellite in self._satellites]
        return Constellation({satellite.id_: satellite for satellite in satellites})

    def apply_assignment(self, assignment: list[int], ongoing_tasks: TaskSet) -> None:
        current_constellation = self.get_constellation().sort()
        for satellite, task_index, live_satellite in zip(self._satellites, assignment, current_constellation):
            target = None
            if task_index != -1 and task_index < len(ongoing_tasks):
                target = ongoing_tasks[task_index].coordinate
            toggle = (task_index == -1 and live_satellite.sensor.enabled) or (
                task_index != -1 and not live_satellite.sensor.enabled
            )
            if toggle:
                satellite.toggle()
            satellite.guide_attitude(target)

    def step(self) -> None:
        self.timer.step()
        self._simulator.ConfigureStopTime(macros.sec2nano(self.timer.time * INTERVAL))
        self._simulator.ExecuteSimulation()

    def is_visible(self, taskset: TaskSet) -> torch.Tensor:
        visibility = torch.zeros((self.num_satellites, len(taskset)), dtype=torch.bool)
        for satellite_index, satellite in enumerate(self._satellites):
            for task_index, task in enumerate(taskset):
                access = satellite.ground_mapping.accessOutMsgs[task_index].read().hasAccess
                if access and satellite.power_sink.powerStatus and task.sensor_type == satellite.sensor_type:
                    visibility[satellite_index, task_index] = True
        return visibility
