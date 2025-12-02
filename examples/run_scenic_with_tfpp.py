import scenic
from scenic.simulators.carla.simulator import CarlaSimulator
import carla
import os
import time
from scenic.core.simulators import SimulationCreationError
from scenic.core.dynamics.utils import RejectSimulationException

from team_code.sensor_agent import SensorAgent  # from carla_garage

# Paths from your working setup
AGENT_CONFIG = "pretrained_models/all_towns"   # folder with config.json + .pth
HOST = "192.168.16.1"
PORT = 2000
TIMESTEP = 0.05

def main():
    # 1) Load Scenic scenario from file
    scenario = scenic.scenarioFromFile("examples/scenario1.scenic", mode2D=True)

    # 2) Create CARLA simulator (same IP/port as leaderboard)
    sim = CarlaSimulator(
        carla_map="Town05",
        map_path=None,
        address=HOST,
        port=PORT,
        timeout=10.0,
        render=True,
        traffic_manager_port=None,
	timestep=TIMESTEP,
    )

    # 3) Initialize scenario in the simulator (spawns actors)
    # scene, _ = scenario.generate()  # sample one scene
    # simulation = sim.createSimulation(scene)
    # simulation = sim.createSimulation(scene, timestep=sim.timestep)
    # simulation = sim.createSimulation(
    #   scene,
    # 	timestep=sim.timestep,
    # 	maxSteps=None,          # or some int, e.g. 1000
    # 	name="scenario1"        # just an identifier for this run
    # )

    # 3) Generate a scene and create a simulation, retrying on spawn failures
    max_attempts = 10
    simulation = None

    for attempt in range(1, max_attempts + 1):
        scene, _ = scenario.generate()
        try:
            simulation = sim.createSimulation(
                scene,
                timestep=sim.timestep,
                maxSteps=None,
                name=f"scenario1_attempt{attempt}"
            )
            print(f"Created simulation on attempt {attempt}")
            break
        except (SimulationCreationError, RejectSimulationException) as e:
            print(f"[Attempt {attempt}] Rejecting: {e}")
            continue

    if simulation is None:
        raise RuntimeError(f"Could not create a valid simulation after {max_attempts} attempts.")

    # Scenic stores ego(s) as simulation.egoObjects
    ego_obj = simulation.egoObjects[0]
    ego_actor: carla.Actor = ego_obj.carlaActor  # CARLA vehicle

    world = sim.world
    traffic_manager = sim.trafficManager  # if you want for NPCs

    # 4) Create TransFuser++ agent
    class Args:
        agent_config = AGENT_CONFIG
    # SensorAgent.setup expects args.agent_config, date_string, traffic_manager
    agent = SensorAgent(ego_actor.id)
    agent.setup(AGENT_CONFIG, time.strftime("%Y%m%d-%H%M%S"), traffic_manager)

    # 5) Main control loop
    try:
        while True:
            world.tick()
            control = agent.run_step()
            ego_actor.apply_control(control)
    finally:
        results = {}  # or however you want to log
        agent.destroy(results)

if __name__ == "__main__":
    main()
