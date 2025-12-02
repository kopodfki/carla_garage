param map = "Town05"
param address = "192.168.16.1"
param port = 2000

model scenic.simulators.carla.model

# Simple ego + one NPC with built-in Scenic behaviors
behavior EgoBehavior(speed=10):
    do FollowLaneBehavior(speed)

behavior NPCBehavior(speed=8):
    do FollowLaneBehavior(speed)

lane = Uniform(*network.lanes)

ego = new Car on lane.centerline, with behavior EgoBehavior(10)
npc = new Car ahead of ego by 15, with behavior NPCBehavior(8)

terminate when ego.speed < 0.1
