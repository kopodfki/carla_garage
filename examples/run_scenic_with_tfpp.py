#!/usr/bin/env python3
import os
import time
import threading
from pathlib import Path

import numpy as np
import carla
import scenic
from scenic.simulators.carla.simulator import CarlaSimulator

import external_control

DEBUG = False
TIMESTEP = 0.05
MAX_STEPS = 800

SCENIC_FILE = Path("examples/scenario1.scenic")


def dbg(*a):
    if DEBUG:
        print(*a)


def find_vehicle_by_role(world: carla.World, role_name: str):
    for v in world.get_actors().filter("vehicle.*"):
        if v.attributes.get("role_name") == role_name:
            return v
    return None


class SensorInterface:
    def __init__(self, expected_ids):
        self.expected = set(expected_ids)
        self.lock = threading.Condition()
        self.buffer = {}

    def push(self, sensor_id, frame, value):
        with self.lock:
            d = self.buffer.setdefault(frame, {})
            d[sensor_id] = value
            self.lock.notify_all()

    def get_complete(self, timeout=2.0):
        deadline = time.time() + timeout
        with self.lock:
            while True:
                for frame in sorted(self.buffer.keys()):
                    d = self.buffer[frame]
                    if self.expected.issubset(d.keys()):
                        self.buffer.pop(frame, None)
                        return frame, d
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None, None
                self.lock.wait(timeout=remaining)


def spawn_sensors(world: carla.World, ego: carla.Vehicle, sensor_specs, iface: SensorInterface):
    bp_lib = world.get_blueprint_library()
    spawned = []

    def make_transform(s):
        loc = carla.Location(x=float(s.get("x", 0)), y=float(s.get("y", 0)), z=float(s.get("z", 0)))
        rot = carla.Rotation(roll=float(s.get("roll", 0)), pitch=float(s.get("pitch", 0)), yaw=float(s.get("yaw", 0)))
        return carla.Transform(loc, rot)

    def on_rgb(sensor_id, image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        arr = arr[:, :, ::-1]
        iface.push(sensor_id, image.frame, arr)

    def on_lidar(sensor_id, lidar: carla.LidarMeasurement):
        pts = np.frombuffer(lidar.raw_data, dtype=np.float32).reshape((-1, 4))
        iface.push(sensor_id, lidar.frame, pts)

    def on_gnss(sensor_id, gnss: carla.GnssMeasurement):
        iface.push(sensor_id, gnss.frame, {"lat": gnss.latitude, "lon": gnss.longitude, "z": gnss.altitude})

    def on_imu(sensor_id, imu: carla.IMUMeasurement):
        iface.push(
            sensor_id,
            imu.frame,
            {
                "accelerometer": (imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z),
                "gyroscope": (imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z),
                "compass": imu.compass,
            },
        )

    for s in sensor_specs:
        stype = s["type"]
        sid = s["id"]
        tf = make_transform(s)

        if stype == "sensor.camera.rgb":
            bp = bp_lib.find(stype)
            bp.set_attribute("image_size_x", str(int(s.get("width", 800))))
            bp.set_attribute("image_size_y", str(int(s.get("height", 600))))
            bp.set_attribute("fov", str(float(s.get("fov", 90))))
            sensor = world.spawn_actor(bp, tf, attach_to=ego)
            sensor.listen(lambda data, _sid=sid: on_rgb(_sid, data))
            spawned.append(sensor)

        elif stype == "sensor.lidar.ray_cast":
            bp = bp_lib.find(stype)
            sensor = world.spawn_actor(bp, tf, attach_to=ego)
            sensor.listen(lambda data, _sid=sid: on_lidar(_sid, data))
            spawned.append(sensor)

        elif stype == "sensor.other.gnss":
            bp = bp_lib.find(stype)
            sensor = world.spawn_actor(bp, tf, attach_to=ego)
            sensor.listen(lambda data, _sid=sid: on_gnss(_sid, data))
            spawned.append(sensor)

        elif stype == "sensor.other.imu":
            bp = bp_lib.find(stype)
            sensor = world.spawn_actor(bp, tf, attach_to=ego)
            sensor.listen(lambda data, _sid=sid: on_imu(_sid, data))
            spawned.append(sensor)

        else:
            raise RuntimeError(f"Unsupported sensor type: {stype}")

    return spawned


def build_straight_global_plan(world: carla.World, ego: carla.Vehicle, meters_ahead=80.0):
    try:
        from agents.navigation.local_planner import RoadOption
        cmd = RoadOption.LANEFOLLOW
    except Exception:
        cmd = 0

    m = world.get_map()
    start = ego.get_transform()
    wp0 = m.get_waypoint(start.location)
    wp1 = wp0.next(meters_ahead)[0]
    end = wp1.transform

    world_plan = [(start, cmd), (end, cmd)]

    g0 = m.transform_to_geolocation(start.location)
    g1 = m.transform_to_geolocation(end.location)
    gps_plan = [
        ({"lat": g0.latitude, "lon": g0.longitude, "z": g0.altitude}, cmd),
        ({"lat": g1.latitude, "lon": g1.longitude, "z": g1.altitude}, cmd),
    ]
    return gps_plan, world_plan


def run_scenic(scene, sim_kwargs, done_evt, exc_holder):
    try:
        import inspect
        from scenic.simulators.carla.simulator import CarlaSimulator

        sim_kwargs = dict(sim_kwargs)
        sim_kwargs["timestep"] = TIMESTEP

        sig = inspect.signature(CarlaSimulator.__init__)
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        filtered = {k: v for k, v in sim_kwargs.items() if k in allowed}

        sim = CarlaSimulator(**filtered)

        sim.createSimulation(scene, maxSteps=MAX_STEPS, name="tfpp")

    except Exception as e:
        exc_holder["exc"] = e
    finally:
        done_evt.set()


def main():
    if not SCENIC_FILE.exists():
        raise FileNotFoundError(f"Cannot find {SCENIC_FILE}. Run from repo root?")

    scenario = scenic.scenarioFromFile(str(SCENIC_FILE), params={"use2DMap": True})
    scene, _ = scenario.generate()

    address = getattr(scenario, "params", {}).get("address", "127.0.0.1")
    port = int(getattr(scenario, "params", {}).get("port", 2000))
    carla_map = getattr(scenario, "params", {}).get("carla_map", None)
    map_path = getattr(scenario, "params", {}).get("map", None)

    if carla_map is None or map_path is None:
        raise RuntimeError("Your Scenic file must define params 'carla_map' and 'map' (xodr path).")

    sim_kwargs = dict(
        address=address,
        port=port,
        carla_map=carla_map,
        map_path=map_path,
        render=True,
    )

    done_evt = threading.Event()
    exc_holder = {}
    t = threading.Thread(target=run_scenic, args=(scene, sim_kwargs, done_evt, exc_holder), daemon=True)
    t.start()

    client = carla.Client(address, port)
    client.set_timeout(10.0)
    world = client.get_world()

    ego = None
    for _ in range(200):
        ego = find_vehicle_by_role(world, "ego")
        if ego is not None:
            break
        if done_evt.is_set():
            break
        time.sleep(0.05)

    if ego is None:
        if "exc" in exc_holder:
            raise RuntimeError("Scenic thread crashed before spawning ego") from exc_holder["exc"]
        raise RuntimeError("Could not find ego vehicle with role_name='ego'")

    dbg("Found ego:", ego.id)

    from team_code.sensor_agent import SensorAgent

    import inspect
    sig = inspect.signature(SensorAgent)
    if "carla_port" in sig.parameters:
        agent = SensorAgent(carla_port=port)
    else:
        agent = SensorAgent(port)

    if hasattr(agent, "setup"):
        try:
            agent.setup(str(Path(".")))
        except TypeError:
            agent.setup()

    gps_plan, world_plan = build_straight_global_plan(world, ego)
    if hasattr(agent, "set_global_plan"):
        try:
            agent.set_global_plan(gps_plan, world_plan)
        except TypeError:
            try:
                agent.set_global_plan(world_plan)
            except Exception:
                pass

    sensor_specs = agent.sensors()
    sensor_ids = [s["id"] for s in sensor_specs if s["type"] != "sensor.speed"]
    iface = SensorInterface(sensor_ids)
    sensors = spawn_sensors(world, ego, [s for s in sensor_specs if s["type"] != "sensor.speed"], iface)

    try:
        external_control.put((0.0, 0.0, 0.0))

        while not done_evt.is_set():
            frame, data = iface.get_complete(timeout=2.0)
            if frame is None:
                continue

            input_data = {sid: (frame, payload) for sid, payload in data.items()}

            v = ego.get_velocity()
            speed = float((v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5)
            input_data["speed"] = (frame, {"speed": speed})

            ts = world.get_snapshot().timestamp.elapsed_seconds

            control = agent.run_step(input_data, ts)
            external_control.put((float(control.throttle), float(control.steer), float(control.brake)))

            if DEBUG:
                adv = None
                for w in world.get_actors().filter("walker.pedestrian.*"):
                    adv = w
                    break
                if adv:
                    d = ego.get_location().distance(adv.get_location())
                    dbg(f"frame={frame} speed={speed:.2f}m/s dist_to_ped={d:.1f}m")

        if "exc" in exc_holder:
            raise exc_holder["exc"]

    finally:
        for s in sensors:
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
