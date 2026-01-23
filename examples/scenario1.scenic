'''The ego vehicle is driving on a straight road. The adversarial pedestrian is hidden behind a parked delivery truck on the right front. The pedestrian sprints abruptly into the road directly in front of the ego vehicle. Due to the sudden appearance and short reaction time, the ego vehicle collides with the pedestrian.'''
param address = "127.0.0.1"
param port = 2000
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
# param use2DMap = True
param map = localPath(f'/home/tda/Desktop/phd/code/ChatScene/safebench/scenario/scenario_data/scenic_data/maps/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(
	ego, 
	globalParameters.OPT_ADV_SPEED, 
	globalParameters.OPT_ADV_DISTANCE
    )
    while True:
        take SetWalkingSpeedAction(0)

behavior EgoExternalControl():
    while True:
        wait

param OPT_ADV_SPEED = Range(3, 6)
param OPT_ADV_DISTANCE = Range(5, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
# END BEHAVIOR
# BEGIN GEOMETRY
lane = Uniform(*network.lanes)
EgoTrajectory = lane.centerline
EgoSpawnPt =  OrientedPoint on lane.centerline

ego =  Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL,
    # with behavior AutopilotBehavior(),
    with behavior EgoExternalControl(),
    with name "ego",
    with rolename "ego"

# END GEOMETRY
# BEGIN SPAWN
param OPT_GEO_BLOCKER_X_DISTANCE = Range(2, 6)
param OPT_GEO_BLOCKER_Y_DISTANCE = Range(10, 40)
param OPT_GEO_X_DISTANCE = Range(-1, 1)
param OPT_GEO_Y_DISTANCE = Range(1, 5)

IntSpawnPt =  OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_BLOCKER_Y_DISTANCE
Blocker =  Car right of IntSpawnPt by globalParameters.OPT_GEO_BLOCKER_X_DISTANCE,
    facing IntSpawnPt.heading,
    with blueprint EGO_MODEL,
    with regionContainedIn None,
    with name "blocker"

SHIFT = Vector(globalParameters.OPT_GEO_X_DISTANCE, globalParameters.OPT_GEO_Y_DISTANCE)
AdvAgent =  Pedestrian at Blocker offset along IntSpawnPt.heading by SHIFT,
    facing IntSpawnPt.heading + 90 deg,
    with regionContainedIn None,
    with behavior AdvBehavior(),
    with name "adv"
# END SPAWN

# BEGIN REQUIREMENTS
# require distance from ego to AdvAgent > 5
# require distance from ego to Blocker > 5
# require AdvAgent in network.walkableRegion
# require ego in network.drivableRegion
# require eventually AdvAgent in network.drivableRegion
# require ego.canSee(AdvAgent) and not ego.canSee(
#     AdvAgent, occludingObjects=tuple([Blocker])
# )
# require eventually distance from ego to AdvAgent < 1
# require eventually distance from ego to Blocker > 3

terminate after 30 seconds
# END REQUIREMENTS
