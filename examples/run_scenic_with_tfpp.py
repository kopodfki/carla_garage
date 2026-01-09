import os, sys, time
import carla
import numpy as np
import scenic

from scenic.simulators.carla.simulator import CarlaSimulator
from scenic.core.simulators import SimulationCreationError
from scenic.core.dynamics.utils import RejectSimulationException

SCENIC_FILE  = "examples/scenario1.scenic"
AGENT_CONFIG = "pretrained_models/all_towns"
HOST         = "192.168.16.1"
PORT         = 2000
TOWN         = "Town05"
TIMESTEP     = 0.05
MAX_STEPS    = 10_000_000
MAX_ATTEMPTS = 10
CLEAN_WORLD  = True
DEBUG        = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
for p in (REPO_ROOT,
          os.path.join(REPO_ROOT, "team_code"),
          os.path.join(REPO_ROOT, "scenario_runner"),
          os.path.join(REPO_ROOT, "leaderboard")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sensor_agent import SensorAgent


def _is_listening(sensor) -> bool:
    """CARLA sensors differ by version: is_listening can be a method or property."""
    try:
        v = getattr(sensor, "is_listening", False)
        return v() if callable(v) else bool(v)
    except Exception:
        return False


def destroy_all_sensors(world: carla.World):
    """Hard kill all sensors in the world (best effort)."""
    try:
        sensors = list(world.get_actors().filter("sensor.*"))
    except Exception:
        sensors = []
    for s in sensors:
        try:
            if hasattr(s, "is_alive") and not s.is_alive:
                continue
            if isinstance(s, carla.Sensor) and _is_listening(s):
                try: s.stop()
                except Exception: pass
            s.destroy()
        except Exception:
            pass


def cleanup_dynamic_actors(world: carla.World):
    """Destroy leftover vehicles/walkers/sensors from previous runs."""
    kill = []
    for a in world.get_actors():
        tid = a.type_id
        if tid.startswith("vehicle.") or tid.startswith("walker.") or tid.startswith("sensor."):
            kill.append(a)

    for a in kill:
        try:
            if hasattr(a, "is_alive") and not a.is_alive:
                continue
            if isinstance(a, carla.Sensor) and _is_listening(a):
                try: a.stop()
                except Exception: pass
            a.destroy()
        except Exception:
            pass


def find_ego(simulation, world):
    """Prefer Scenic mapping; fallback to world scan by role_name."""
    for obj in getattr(simulation, "objects", []):
        if getattr(obj, "rolename", None) == "ego" or getattr(obj, "name", None) == "ego":
            act = getattr(obj, "carlaActor", None)
            if act is not None:
                return act
    for v in world.get_actors().filter("vehicle.*"):
        if v.attributes.get("role_name") == "ego":
            return v
    return None


def build_gps_global_plan_dicts(world: carla.World, ego: carla.Vehicle, meters_ahead=250.0, step=2.0):
    """
    TFPP nav_planner expects: [(pos_dict, cmd), ...]
    where pos_dict has keys 'lat','lon','z'
    """
    try:
        from agents.navigation.local_planner import RoadOption
        cmd = RoadOption.LANEFOLLOW
    except Exception:
        cmd = 0

    m = world.get_map()
    wp = m.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    n = max(2, int(meters_ahead / step))

    plan = []
    for _ in range(n):
        loc = wp.transform.location
        geo = m.transform_to_geolocation(loc)
        pos = {"lat": float(geo.latitude), "lon": float(geo.longitude), "z": float(geo.altitude)}
        plan.append((pos, cmd))

        nxt = wp.next(step)
        if not nxt:
            break
        wp = nxt[0]
    return plan


class SensorBuffer:
    """
    Spawn sensors from agent.sensors() and provide input_data as:
      input_data[sensor_id] = (timestamp, data)
    IMPORTANT: TFPP expects speed as dict: {'speed': <kmh>}
    """
    def __init__(self, world: carla.World, ego: carla.Vehicle):
        self.world = world
        self.ego = ego
        self.sensors = []
        self.expected = []
        self.speedometer_ids = set()
        self.latest = {}

    def spawn_from_agent(self, agent):
        if not hasattr(agent, "sensors"):
            return

        specs = agent.sensors()
        lib = self.world.get_blueprint_library()

        for s in specs:
            stype = s.get("type")
            sid = s.get("id")
            if not stype or not sid:
                continue

            self.expected.append(sid)

            if stype == "sensor.speedometer":
                self.speedometer_ids.add(sid)
                continue

            bp = lib.find(stype)

            if "width" in s and bp.has_attribute("image_size_x"):
                bp.set_attribute("image_size_x", str(s["width"]))
            if "height" in s and bp.has_attribute("image_size_y"):
                bp.set_attribute("image_size_y", str(s["height"]))
            if "fov" in s and bp.has_attribute("fov"):
                bp.set_attribute("fov", str(s["fov"]))
            if bp.has_attribute("sensor_tick"):
                bp.set_attribute("sensor_tick", str(TIMESTEP))

            tf = carla.Transform(
                carla.Location(float(s.get("x", 0)), float(s.get("y", 0)), float(s.get("z", 0))),
                carla.Rotation(float(s.get("roll", 0)), float(s.get("pitch", 0)), float(s.get("yaw", 0))),
            )

            actor = self.world.spawn_actor(bp, tf, attach_to=self.ego)
            actor.listen(self._cb_maker(sid))
            self.sensors.append(actor)

    def _cb_maker(self, sid):
        def cb(meas):
            frame = getattr(meas, "frame", None)
            self.latest[sid] = (frame, self._convert(meas))
        return cb

    def _convert(self, meas):
        if isinstance(meas, carla.Image):
            arr = np.frombuffer(meas.raw_data, dtype=np.uint8).reshape((meas.height, meas.width, 4))
            return arr[:, :, :3][:, :, ::-1]

        if isinstance(meas, (carla.LidarMeasurement, carla.SemanticLidarMeasurement)):
            return np.frombuffer(meas.raw_data, dtype=np.float32).reshape((-1, 4))

        if isinstance(meas, carla.GnssMeasurement):
            return np.array([meas.latitude, meas.longitude, meas.altitude], dtype=np.float32)

        if isinstance(meas, carla.IMUMeasurement):
            return np.array([
                meas.accelerometer.x, meas.accelerometer.y, meas.accelerometer.z,
                meas.gyroscope.x, meas.gyroscope.y, meas.gyroscope.z,
                meas.compass
            ], dtype=np.float32)

        return meas

    def input_data(self, frame, timestamp):
        v = self.ego.get_velocity()
        speed_mps = float((v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5)
        speed_kmh = speed_mps * 3.6
        for sid in self.speedometer_ids:
            self.latest[sid] = (frame, {"speed": speed_kmh})

        deadline = time.time() + 0.3
        while time.time() < deadline:
            ok = True
            for sid in self.expected:
                if sid in self.speedometer_ids:
                    continue
                fr, _ = self.latest.get(sid, (None, None))
                if fr != frame:
                    ok = False
                    break
            if ok:
                break
            time.sleep(0.001)

        out = {}
        for sid in self.expected:
            fr, data = self.latest.get(sid, (None, None))
            if fr == frame:
                out[sid] = (timestamp, data)
        return out

    def destroy(self):
        for s in self.sensors:
            try:
                if hasattr(s, "is_alive") and not s.is_alive:
                    continue
                if isinstance(s, carla.Sensor) and _is_listening(s):
                    try: s.stop()
                    except Exception: pass
                s.destroy()
            except Exception:
                pass
        self.sensors.clear()
        self.expected.clear()
        self.speedometer_ids.clear()
        self.latest.clear()


def main():
    try:
        agent = SensorAgent(AGENT_CONFIG, PORT)
    except TypeError:
        agent = SensorAgent(AGENT_CONFIG, HOST, PORT)

    try:
        agent.setup(AGENT_CONFIG, time.strftime("%Y%m%d-%H%M%S"), None)
    except TypeError:
        agent.setup(AGENT_CONFIG)

    scenario = scenic.scenarioFromFile(SCENIC_FILE, mode2D=True, params={"use2DMap": True})

    sim = CarlaSimulator(
        carla_map=TOWN,
        map_path=None,
        address=HOST,
        port=PORT,
        timeout=10.0,
        render=True,
        traffic_manager_port=None,
        timestep=TIMESTEP,
    )

    if CLEAN_WORLD:
        cleanup_dynamic_actors(sim.world)

    simulation = None
    sensors = None

    try:
        last_err = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            scene, _ = scenario.generate()
            try:
                simulation = sim.createSimulation(scene, timestep=TIMESTEP, maxSteps=MAX_STEPS, name=f"attempt{attempt}")
                simulation.setup()
                print(f"Simulation ready (attempt {attempt})")
                break
            except (SimulationCreationError, RejectSimulationException, RuntimeError) as e:
                last_err = e
                try:
                    if simulation is not None:
                        simulation.destroy()
                except Exception:
                    pass
                simulation = None

        if simulation is None:
            raise RuntimeError(f"Failed to create simulation after {MAX_ATTEMPTS} attempts. Last error: {last_err}")

        world = simulation.world
        ego = find_ego(simulation, world)
        if ego is None:
            raise RuntimeError("Could not find ego after simulation.setup().")

        ego.set_autopilot(False)
        ego.set_simulate_physics(True)

        gps_global_plan = build_gps_global_plan_dicts(world, ego)
        agent._global_plan = gps_global_plan

        sensors = SensorBuffer(world, ego)
        sensors.spawn_from_agent(agent)
        for _ in range(5):
            simulation.step()

        while True:
            simulation.step()
            snap = world.get_snapshot()
            ts = snap.timestamp
            frame = snap.frame

            input_data = sensors.input_data(frame, ts)
            control = agent.run_step(input_data, ts)
            ego.apply_control(control)

            if DEBUG and frame % 20 == 0:
                v = ego.get_velocity()
                sp = float((v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5)
                loc = ego.get_transform().location
                print(f"frame={frame} speed={sp:.2f} m/s pos=({loc.x:.1f},{loc.y:.1f})")

    finally:
        try:
            if sensors is not None:
                sensors.destroy()
        except Exception:
            pass

        try:
            destroy_all_sensors(sim.world)
        except Exception:
            pass

        try:
            if simulation is not None:
                simulation.destroy()
        except Exception:
            pass

        try:
            sim.destroy()
        except Exception:
            pass

        try:
            if hasattr(agent, "destroy"):
                try:
                    agent.destroy({})
                except TypeError:
                    agent.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
