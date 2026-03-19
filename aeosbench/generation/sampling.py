"""Sampling and serialization helpers for generated datasets."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random

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
    Task,
    TaskSet,
)


def _unit_vector_from_angles(rng: random.Random) -> tuple[float, float, float]:
    azimuth = rng.uniform(-180.0, 180.0)
    elevation = rng.uniform(-90.0, 90.0)
    azimuth_radians = math.radians(azimuth)
    elevation_radians = math.radians(elevation)
    return (
        math.cos(elevation_radians) * math.cos(azimuth_radians),
        math.cos(elevation_radians) * math.sin(azimuth_radians),
        math.sin(elevation_radians),
    )


def sample_orbit(rng: random.Random, orbit_id: int) -> Orbit:
    return Orbit(
        id_=orbit_id,
        eccentricity=rng.uniform(0.0, 0.005),
        semi_major_axis=rng.uniform(6.8e6, 8.0e6),
        inclination=rng.uniform(0.0, 180.0),
        right_ascension_of_the_ascending_node=rng.uniform(0.0, 360.0),
        argument_of_perigee=rng.uniform(0.0, 360.0),
    )


def sample_solar_panel(rng: random.Random) -> SolarPanel:
    return SolarPanel(
        direction=_unit_vector_from_angles(rng),
        area=rng.uniform(5.0, 10.0),
        efficiency=rng.uniform(0.1, 0.5),
    )


def sample_sensor(rng: random.Random, *, screening: bool = False) -> Sensor:
    if screening:
        # Asset screening uses one fixed, generous visible sensor so acceptance
        # reflects the platform rather than random sensor draws.
        half_fov = 5.0
        power = 5.0
    else:
        half_fov = rng.uniform(0.5, 1.5)
        power = rng.uniform(2.0, 8.0)
    return Sensor(
        type_=SensorType.VISIBLE,
        enabled=False,
        half_field_of_view=half_fov,
        power=power,
    )


def sample_battery(rng: random.Random, *, screening: bool = False) -> Battery:
    if screening:
        return Battery(capacity=3.0e6, percentage=1.0)
    return Battery(
        capacity=rng.uniform(8_000.0, 30_000.0),
        percentage=rng.uniform(0.0, 1.0),
    )


def sample_reaction_wheels(rng: random.Random) -> tuple[ReactionWheel, ReactionWheel, ReactionWheel]:
    wheel_types = [
        ("Honeywell_HR12", [12.0, 25.0, 50.0]),
        ("Honeywell_HR14", [75.0, 25.0, 50.0]),
        ("Honeywell_HR16", [100.0, 75.0, 50.0]),
    ]
    wheel_type, candidate_momentum = rng.choice(wheel_types)
    max_momentum = float(rng.choice(candidate_momentum))
    directions = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return tuple(
        ReactionWheel(
            rw_type=wheel_type,
            rw_direction=direction,
            max_momentum=max_momentum,
            rw_speed_init=rng.uniform(400.0, 750.0),
            power=rng.uniform(5.0, 7.0),
            efficiency=rng.uniform(0.5, 0.6),
        )
        for direction in directions
    )


def sample_mrp_control(
    rng: random.Random,
    inertia: tuple[float, float, float, float, float, float, float, float, float],
    reaction_wheels: tuple[ReactionWheel, ReactionWheel, ReactionWheel],
) -> MRPControl:
    max_momentum = max(wheel.max_momentum for wheel in reaction_wheels)
    k = rng.uniform(2.0, 5.0) * max(inertia) / max_momentum
    ki = rng.uniform(0.0, 0.01)
    p = rng.uniform(2.0, 4.0) * k
    integral_limit = rng.uniform(0.0, 5.0) * ki
    return MRPControl(k=k, ki=ki, p=p, integral_limit=integral_limit)


def sample_screening_satellite(rng: random.Random, *, orbit_id: int = 0) -> Satellite:
    inertia_diag = [rng.uniform(50.0, 200.0) for _ in range(3)]
    inertia = (
        inertia_diag[0],
        0.0,
        0.0,
        0.0,
        inertia_diag[1],
        0.0,
        0.0,
        0.0,
        inertia_diag[2],
    )
    reaction_wheels = sample_reaction_wheels(rng)
    return Satellite(
        id_=0,
        inertia=inertia,
        mass=rng.uniform(50.0, 200.0),
        center_of_mass=(0.0, 0.0, 0.0),
        orbit_id=orbit_id,
        orbit=Orbit(
            id_=orbit_id,
            eccentricity=0.0001,
            semi_major_axis=7.2e6,
            inclination=0.0,
            right_ascension_of_the_ascending_node=0.0,
            argument_of_perigee=0.0,
        ),
        solar_panel=SolarPanel(
            direction=(0.0, 1.0, 0.0),
            area=5.0,
            efficiency=0.5,
        ),
        sensor=sample_sensor(rng, screening=True),
        battery=sample_battery(rng, screening=True),
        reaction_wheels=reaction_wheels,
        mrp_control=sample_mrp_control(rng, inertia, reaction_wheels),
        true_anomaly=0.0,
        mrp_attitude_bn=(0.0, 0.0, 0.0),
    )


def instantiate_satellite_from_asset(
    rng: random.Random,
    *,
    satellite_id: int,
    orbit_id: int,
    asset: Satellite,
) -> Satellite:
    orbit = sample_orbit(rng, orbit_id)
    return Satellite(
        id_=satellite_id,
        inertia=asset.inertia,
        mass=asset.mass,
        center_of_mass=asset.center_of_mass,
        orbit_id=orbit.id_,
        orbit=orbit,
        solar_panel=sample_solar_panel(rng),
        sensor=sample_sensor(rng),
        battery=sample_battery(rng),
        reaction_wheels=asset.reaction_wheels,
        mrp_control=asset.mrp_control,
        true_anomaly=rng.uniform(0.0, 360.0),
        mrp_attitude_bn=(0.0, 0.0, 0.0),
    )


def sample_task(rng: random.Random, *, task_id: int, config_max_time_step: int, min_duration: int, max_duration: int) -> Task:
    duration = rng.randint(min_duration, max_duration)
    latest_release = max(0, config_max_time_step - duration * 3)
    release_time = rng.randint(0, latest_release)
    due_time = rng.randint(release_time + duration * 3, config_max_time_step)
    return Task(
        id_=task_id,
        release_time=release_time,
        due_time=due_time,
        duration=duration,
        coordinate=(rng.uniform(-90.0, 90.0), rng.uniform(-180.0, 180.0)),
        sensor_type=SensorType.VISIBLE,
    )


def sample_screening_taskset(rng: random.Random, *, size: int, horizon: int) -> TaskSet:
    return TaskSet(
        Task(
            id_=task_id,
            release_time=0,
            due_time=horizon,
            duration=10,
            coordinate=(rng.uniform(-10.0, 10.0), rng.uniform(-180.0, 180.0)),
            sensor_type=SensorType.VISIBLE,
        )
        for task_id in range(size)
    )


def sample_taskset(
    rng: random.Random,
    *,
    num_tasks: int,
    min_duration: int,
    max_duration: int,
    max_time_step: int,
) -> TaskSet:
    return TaskSet(
        sample_task(
            rng,
            task_id=task_id,
            config_max_time_step=max_time_step,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        for task_id in range(num_tasks)
    )


def orbit_to_payload(orbit: Orbit) -> dict[str, object]:
    return {
        "id": orbit.id_,
        "eccentricity": orbit.eccentricity,
        "semi_major_axis": orbit.semi_major_axis,
        "inclination": orbit.inclination,
        "right_ascension_of_the_ascending_node": orbit.right_ascension_of_the_ascending_node,
        "argument_of_perigee": orbit.argument_of_perigee,
    }


def satellite_to_payload(satellite: Satellite) -> dict[str, object]:
    return {
        "id": satellite.id_,
        "inertia": list(satellite.inertia),
        "mass": satellite.mass,
        "center_of_mass": list(satellite.center_of_mass),
        "orbit": satellite.orbit_id,
        "solar_panel": {
            "direction": list(satellite.solar_panel.direction),
            "area": satellite.solar_panel.area,
            "efficiency": satellite.solar_panel.efficiency,
        },
        "sensor": {
            "type": int(satellite.sensor.type_),
            "enabled": satellite.sensor.enabled,
            "half_field_of_view": satellite.sensor.half_field_of_view,
            "power": satellite.sensor.power,
        },
        "battery": {
            "capacity": satellite.battery.capacity,
            "percentage": satellite.battery.percentage,
        },
        "reaction_wheels": [
            {
                "rw_type": wheel.rw_type,
                "rw_direction": list(wheel.rw_direction),
                "max_momentum": wheel.max_momentum,
                "rw_speed_init": wheel.rw_speed_init,
                "power": wheel.power,
                "efficiency": wheel.efficiency,
            }
            for wheel in satellite.reaction_wheels
        ],
        "mrp_control": {
            "k": satellite.mrp_control.k,
            "ki": satellite.mrp_control.ki,
            "p": satellite.mrp_control.p,
            "integral_limit": satellite.mrp_control.integral_limit,
        },
        "true_anomaly": satellite.true_anomaly,
        "mrp_attitude_bn": list(satellite.mrp_attitude_bn),
    }


def task_to_payload(task: Task) -> dict[str, object]:
    return {
        "id": task.id_,
        "release_time": task.release_time,
        "due_time": task.due_time,
        "duration": task.duration,
        "coordinate": [task.coordinate[0], task.coordinate[1]],
        "sensor_type": int(task.sensor_type),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def write_orbit(path: Path, orbit: Orbit) -> None:
    write_json(path, orbit_to_payload(orbit))


def write_constellation(path: Path, constellation: Constellation) -> None:
    satellites = constellation.sort()
    orbit_by_id = {satellite.orbit_id: satellite.orbit for satellite in satellites}
    write_json(
        path,
        {
            "orbits": [
                orbit_to_payload(orbit)
                for orbit in sorted(orbit_by_id.values(), key=lambda item: item.id_)
            ],
            "satellites": [satellite_to_payload(satellite) for satellite in satellites],
        },
    )


def write_taskset(path: Path, taskset: TaskSet) -> None:
    write_json(path, [task_to_payload(task) for task in taskset])
