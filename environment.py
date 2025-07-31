from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from synthnova_config import PhysicsSimulatorConfig, RobotConfig, MujocoConfig, CuboidConfig
from pathlib import Path
from physics_simulator.utils.data_types import JointTrajectory
import numpy as np
import math
import random
import time


def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps)

def printEnv(string):
    envName = "Mujoco"
    print("[" + envName +"] " + str(string))

class IoaiNavEnv:

    def distBetween(self, startVector, endVector):
        term1 = math.pow(endVector[0] - startVector[0],2)
        term2 = math.pow(endVector[1] - startVector[1],2)

        return math.sqrt(term1 + term2)

    def __init__(self, headless=False, seed=random.randint(10000000, 99999999)):
        self.headless = headless
        self.seed = seed

        self.simulator = None
        self.robot = None
        self.interface = None

        self.stepOffset = 0
        self.actionSteps = 0

        ### Goals
        random.seed(self.seed)
        while (True):
            self.startPoint = [random.uniform(-2,4), random.uniform(-3,3)]
            self.endPoint = [random.uniform(-2,4), random.uniform(-3,3)]
            if (self.distBetween(self.startPoint, self.endPoint) >= 1.5):
                # ensure the start and end point are at least 1.5 unit apart
                break

        ### PPO Variables
        self.reward = 0 # float 0-1
        self.done = False

        ### Sim setup
        self._setup_simulator(self.headless)
        self._setup_interface()
    
        self.initDone = False
        self._init_pose()

       

        # action fifo queue
        self.simulator.add_physics_callback("follow_path_callback", self.follow_path_callback)
        self.moving = False
        self.fifoPath = [[self.startPoint[0],self.startPoint[1],0]] # path offset [x,y,yaw], yaw in radians


        # self.moveForward(10)
        # self.simulator.play()
        printEnv("sim is loading...")
        while(not self.initDone):
            # print("running step")
            self.simulator.step()
            # self.follow_path_callback()
        self.stepOffset = self.simulator.get_step_count()

        # self.step(4)
        # self.step(1)
        # self.step(2)
        # self.step(3)
        # self.step(4)
        #self.step(4)
        # if (len(self.fifoPath) == 0):
        #     self.fifoPath.append(self.computeRobotPositionRelative())
        # self.step(1)

        # self.moveForwardsAlt(1)
 
        # while(True):
        #     self.simulator.step(1) 


    def moveForwardsAlt(self, step):
        current_joint_positions = self.interface.chassis.get_joint_positions()

        # Define target joint positions
        target_joint_positions = [0.5, 0, 0]

        # Interpolate joint positions
        positions = interpolate_joint_positions(
            current_joint_positions, target_joint_positions, 5
        )
        # Create a joint trajectory
        joint_trajectory = JointTrajectory(positions=positions)

        # Follow the trajectory
        self.interface.chassis.follow_trajectory(joint_trajectory)
    
    def reset(self):

        # move robot back to start position
        self.done = False

        # time.sleep(3)

        printEnv("moving robot back to start...")
        if (len(self.fifoPath) != 0):
            # clear fifo queue
            self.fifoPath = []
        self.fifoPath.append(self.computeRobotPositionRelative())
        self.fifoPath.append([self.startPoint[0],self.startPoint[1],0]) # append start position to queue

        self.interface.chassis.set_joint_positions([0,0,0],True)
        stepCount = self.simulator.get_step_count()
        attempts = 0
        # while(not self.check_movement_complete([self.startPoint[0], self.startPoint[1],0], 0.1)):
        #     self.simulator.step()
        #     
        #     if (500 <= (self.simulator.get_step_count()-stepCount)):
        #         # sim has stalfled for some reason
        #         self.simulator.reset()

        while(not self.check_movement_complete([self.startPoint[0], self.startPoint[1],0], 0.1)):
                printEnv("Robot is at: " + str(self.computeRobotPositionRelative()))
                printEnv("Chassis relative is at: " + str(self.interface.chassis.get_joint_positions()))
                printEnv("Goal is: " + str(self.startPoint))
                # loop until the robot reaches the start position
                self.simulator.step()
                if (500 <= (self.simulator.get_step_count()-stepCount) and attempts < 10):
                    # if it wasn't moved for whatever reason after 300 steps
                    # DO IT AGAIN
                    print("trying again...")
                    self.interface.chassis.set_joint_positions([0,0,0],True)
                    stepCount = self.simulator.get_step_count()
                    attempts += 1
                elif(10 <= attempts):
                    # worst case reset the simulation entirely
                    self.simulator.reset() 
                    # TODO: this will result in a crash, rely on hypervisor
        self.stepOffset = self.simulator.get_step_count()

        # for i in range(0,2):
        #     # repeat this step several times to ensure robot reaches start point
        #     
        #     self.interface.chassis.set_joint_positions([0,0,0],True)
        #     self.fifoPath.append([self.startPoint[0],self.startPoint[1],0])
        #     while(len(self.fifoPath) != 0):
        #         # loop until the robot reaches the start position
        #         self.simulator.step()
        #     self.stepOffset = self.simulator.get_step_count()
        
        self.actionSteps = 0
        self.done = False
        # time.sleep(1)
        printEnv("simulation ready")
        
        return self.observation() # send observation
        # self.moveForward(4)
        # self.moveBackwards(2)
        # self.moveLeft(3)
        # self.moveRight(6)
        # self.shiftYaw(math.pi/2)
        # self.moveForward(1)

    def step(self, number):
        
        # movement setup
        if (len(self.fifoPath) == 0):
            self.fifoPath.append(self.computeRobotPositionRelative())
        
        globalStepDistance = 0.2 #1.75 ## must be greater than tolerance(0.1)

        # switch case
        match number:
            case 0:
                self.moveForward(globalStepDistance)
                printEnv("forwards")
            case 1:
                self.moveBackwards(globalStepDistance)
                printEnv("backwards")
            case 2:
                self.moveLeft(globalStepDistance)
                printEnv("left")
            case 3:
                self.moveRight(globalStepDistance)
                printEnv("right")
            case 4:
                self.shiftYaw(globalStepDistance)
                printEnv("yaw shift positive")
            case 5:
                self.shiftYaw(-globalStepDistance)
                printEnv("yaw shift negative")
        self.moving = True

        # step the simulation until the robot stops moving
        startTime = self.simulator.get_step_count()-self.stepOffset
        while (self.moving):
            if (500 <= ((self.simulator.get_step_count()-self.stepOffset)-startTime)): # movement took more than 200 steps
                # something is probably wrong
                # ie robot has hit a wall
                # robot has likely hit the wall, punish with reward 0, return done state
                if (self.actionSteps == 0):
                    # something went wrong with the reset...
                    self.reset() # try again
                else:
                    return self.observation(), 0, True, [] # return state information  
            self.simulator.step()

        self.actionSteps+=1
        printEnv("Sim time(steps): " + str(self.simulator.get_step_count()-self.stepOffset))
        
        # sim ran out of time
        if (60 <= self.actionSteps):
            self.done = True
        # goal was reached
        if (self.goalReached()):
            self.done = True
        # return |     state     |       reward calc       |   done?  |
        return self.observation(), self.rewardCalculation(), self.done, [] # return state information

        
        
    def _setup_simulator(self, headless):
        """
        Initialize the physics simulator with basic configuration.
        
        Args:
            headless: Whether to run in headless mode
        """
        # Create simulator config
        # Create simulator config
        config = PhysicsSimulatorConfig(
            mujoco_config=MujocoConfig(headless=headless,
                                       timestep=0.1) # run the simulation at 0.1s per step
        )
        self.simulator = PhysicsSimulator(config)
        
        # Add default scene
        self.simulator.add_default_scene()

        # Add robot
        robot_config = RobotConfig(
            prim_path="/World/Galbot",
            name="galbot_one_charlie",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("robot")
            .joinpath("galbot_one_charlie_description")
            .joinpath("galbot_one_charlie.xml"),
            position=[self.startPoint[0], self.startPoint[1], 0],
            orientation=[0, 0, 0, 1]
        )
        self.simulator.add_robot(robot_config)

        # Initialize the scene
        self._init_scene()
        
        # Initialize the simulator
        self.simulator.initialize()
        
        # Get robot instance for joint name discovery
        self.robot = self.simulator.get_robot("/World/Galbot")

    def _init_scene(self):
        """
        Initialize the scene with tables, closet, and cubes.
        """
        # Add four walls
        wall_color = [0.2,0.2,0.2] # r,g,b
        cube_configs = [
            CuboidConfig(
                prim_path=Path(self.simulator.root_prim_path).joinpath("cube_1"),
                position=[5.5, 0, 2], # x,y,z
                orientation=[0, 0, 0, 1], # x,y,z,w
                scale=[1, 8, 1.8], # x,y,z
                color=wall_color  # grey cube
            ),
            CuboidConfig(
                prim_path=Path(self.simulator.root_prim_path).joinpath("cube_1"),
                position=[1, -4.5, 2],
                orientation=[0, 0, 0, 1],
                scale=[8, 1, 1.8],
                color=wall_color  
            ),
            CuboidConfig(
                prim_path=Path(self.simulator.root_prim_path).joinpath("cube_1"),
                position=[-3.5, 0, 2], 
                orientation=[0, 0, 0, 1],
                scale=[1, 8, 1.8],
                color=wall_color  
            ),
            CuboidConfig(
                prim_path=Path(self.simulator.root_prim_path).joinpath("cube_1"),
                position=[1, 4.5, 2],
                orientation=[0, 0, 0, 1],
                scale=[8, 1, 1.8],
                color=wall_color 
            )
        ]
        for cube in cube_configs:
            self.simulator.add_object(cube)

        # display start and end points
        goals_config = [
            CuboidConfig(
                prim_path=Path(self.simulator.root_prim_path).joinpath("cube_1"),
                position=[self.startPoint[0], self.startPoint[1], 0.1], # x,y,z
                orientation=[0, 0, 0, 1],
                scale=[1, 1, 0.001],
                color=[0.0, 1.0, 0.0]  # Green cube
            ),
            CuboidConfig(
                prim_path=Path(self.simulator.root_prim_path).joinpath("cube_1"),
                position=[self.endPoint[0], self.endPoint[1], 0.1],
                orientation=[0, 0, 0, 1],
                scale=[1, 1, 0.001],
                color=[1.0, 0.0, 0.0]  # Red cube
            )
        ]
        #### Commented out so cubes don't spawn, at high sim speeds these cubes mess with robot physics
        # for cube in goals_config:
        #     self.simulator.add_object(cube)


    def _setup_interface(self):
        galbot_interface_config = GalbotInterfaceConfig()

        galbot_interface_config.robot.prim_path = "/World/Galbot"

        robot_name = self.robot.name

        # Enable modules
        galbot_interface_config.modules_manager.enabled_modules.append("right_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("left_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("leg")
        galbot_interface_config.modules_manager.enabled_modules.append("head")
        galbot_interface_config.modules_manager.enabled_modules.append("chassis")

        # for each module you want to use, define each joint
        galbot_interface_config.right_arm.joint_names = [
            f"{robot_name}/right_arm_joint1",
            f"{robot_name}/right_arm_joint2",
            f"{robot_name}/right_arm_joint3",
            f"{robot_name}/right_arm_joint4",
            f"{robot_name}/right_arm_joint5",
            f"{robot_name}/right_arm_joint6",
            f"{robot_name}/right_arm_joint7",
        ]

        galbot_interface_config.left_arm.joint_names = [
            f"{robot_name}/left_arm_joint1",
            f"{robot_name}/left_arm_joint2",
            f"{robot_name}/left_arm_joint3",
            f"{robot_name}/left_arm_joint4",
            f"{robot_name}/left_arm_joint5",
            f"{robot_name}/left_arm_joint6",
            f"{robot_name}/left_arm_joint7",
        ]

        galbot_interface_config.leg.joint_names = [
            f"{robot_name}/leg_joint1",
            f"{robot_name}/leg_joint2",
            f"{robot_name}/leg_joint3",
            f"{robot_name}/leg_joint4",
        ]
        
        galbot_interface_config.head.joint_names = [
            f"{robot_name}/head_joint1",
            f"{robot_name}/head_joint2"
        ]

        galbot_interface_config.chassis.joint_names = [
            f"{robot_name}/mobile_forward_joint",
            f"{robot_name}/mobile_side_joint",
            f"{robot_name}/mobile_yaw_joint",
        ]

        galbot_interface = GalbotInterface(
            galbot_interface_config=galbot_interface_config,
            simulator=self.simulator
        )
        galbot_interface.initialize()

        self.interface = galbot_interface

    # generic joint position to target position check
    def _is_joint_positions_reached(self, module, target_positions):
        current_positions = module.get_joint_positions()
        return np.allclose(current_positions, target_positions, atol=0.1)
    
    def _init_pose(self):
        # Init head pose
        self.head = [0.0, 0.0]
        self._move_joints_to_target(self.interface.head, self.head)

        # Init leg pose
        self.leg = [0.43, 1.48, 1.07, 0.0]
        self._move_joints_to_target(self.interface.leg, self.leg)

        # Init left arm pose
        self.left_arm = [
            0.058147381991147995,
            1.4785659313201904,
            -0.0999724417924881,
            -2.097979784011841,
            1.3999720811843872,
            -0.009971064515411854,
            1.0999830961227417,
        ]
        self._move_joints_to_target(self.interface.left_arm, self.left_arm)

        # Init right arm pose
        self.right_arm = [
            -0.058147381991147995,
            -1.4785659313201904,
            0.0999724417924881,
            2.097979784011841,
            -1.3999720811843872,
            0.009971064515411854,
            -1.0999830961227417,
        ]
        self._move_joints_to_target(self.interface.right_arm, self.right_arm)

        self.simulator.add_physics_callback("is_init_done", self._init_pose_done)
            
    def _init_pose_done(self):
        headState = False
        legState = False
        left_armState = False
        right_armState = False
        # if head has reached target
        if (headState | self._is_joint_positions_reached(self.interface.head, self.head)):
            headState = True
        
        # if leg has reached target
        if (legState | self._is_joint_positions_reached(self.interface.leg, self.leg)):
            legState = True

        # if left arm has reached target
        if (left_armState | self._is_joint_positions_reached(self.interface.left_arm, self.left_arm)):
            left_armState = True
        
        # if right arm has reached target
        if (right_armState | self._is_joint_positions_reached(self.interface.right_arm, self.right_arm)):
            right_armState = True
        
        # if all targets have been reached
        if (headState and legState and left_armState and right_armState):
            self.stepOffset = self.simulator.get_step_count() # set step offset
            printEnv("init done")
            self.initDone = True
            self.simulator.remove_physics_callback("is_init_done")
            

    def computeRobotPositionRelative(self):
        # the chassis coordinates are relative to where the robot starts
        # compute real coordinates from chassis offset
        robotLocation = self.interface.chassis.get_joint_positions()
        robotLocation = [self.startPoint[0]+robotLocation[0],self.startPoint[1]+robotLocation[1],robotLocation[2]]
        return robotLocation


    # 200 steps, 0.1 seconds per step, operation completed in 20 seconds
    def _move_joints_to_target(self, module, target_positions, steps=200): 
        """Move joints from current position to target position smoothly."""
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

    # chassis movement [0,0,0] # x, y, yaw
    def moveGeneric(self, vector):
        # print("moving generic...")

        # convert real position to chassis local coordinates
        real_pos = self.computeRobotPositionRelative()
        start_pos = self.interface.chassis.get_joint_positions()
        relative_vector = [vector[0]-real_pos[0],vector[1]-real_pos[1], real_pos[2]]
        end_pos = [start_pos[0]+relative_vector[0], start_pos[1]+relative_vector[1], vector[2]]
        print("start: " + str(start_pos))
        print("end: " + str(end_pos))
        positions = np.linspace(start_pos, end_pos, 5) # start_pos, end_pos, 
        # print("trajectory: " + str(positions))
        trajectory = JointTrajectory(positions=positions)

        self.interface.chassis.follow_trajectory(trajectory)

    
    ### Moving dynamically based on yaw
    # https://www.desmos.com/calculator/2wknuddhgu

    def moveForward(self, step):
        if step < 0:
            # ensure movement is forwards
            step = step * -1

        # append translation to fifo queue
        current_pos = self.fifoPath[-1] ## real coordinates
        self.fifoPath.append([current_pos[0]+(math.cos(current_pos[2])*step), current_pos[1] + (math.sin(current_pos[2])*step), current_pos[2]])

        ### 
        # self.moveGeneric([step,0,0])

    def moveBackwards(self, step):
        if 0 < step:
            # ensure movement is backwards
            step = step * -1


        # append translation to fifo queue
        current_pos = self.fifoPath[-1] ## real coordinates
        self.fifoPath.append([current_pos[0]+(math.cos(current_pos[2])*step), current_pos[1] + (math.sin(current_pos[2])*step), current_pos[2]])

        ###
        # self.moveGeneric([step,0,0])

    def moveLeft(self, step):
        if step < 0:
            # ensure movement is left
            step = step * -1

        # append translation to fifo queue
        current_pos = self.fifoPath[-1] ## real coordinates
        self.fifoPath.append([current_pos[0]+(math.cos(current_pos[2]+(math.pi/2))*step), current_pos[1] + (math.sin(current_pos[2]+(math.pi/2))*step), current_pos[2]])


        ###
        # self.moveGeneric([0,step,0])

    def moveRight(self, step):
        if step < 0:
            # ensure movement is right
            step = step * -1

        # append translation to fifo queue
        current_pos = self.fifoPath[-1] ## real coordinates
        self.fifoPath.append([current_pos[0]+(math.cos(current_pos[2]-(math.pi/2))*step), current_pos[1] + (math.sin(current_pos[2]-(math.pi/2))*step), current_pos[2]])

        ###
        # self.moveGeneric([0,step,0])

    def shiftYaw(self, step):
        
        # append translation to fifo queue
        current_pos = self.fifoPath[-1]
        self.fifoPath.append([current_pos[0], current_pos[1], current_pos[2]+step])

        ###
        # self.moveGeneric([0,0,step])


    def check_movement_complete(self, target, tolerance):
        current = self.computeRobotPositionRelative()
        # print("robot is at " + str(current) + ", aiming to go " + str(target))

        # check if robot has reached target within a tolerance 
        if np.allclose(current, target, atol=tolerance): 
            return True

    def follow_path_callback(self):
        # print("Local Chassis coordinate: " + str(self.interface.chassis.get_joint_positions()))
        # ensure sim length is below 3000 steps
        # if 3000 <= (self.simulator.get_step_count()-self.stepOffset):
        #     self.done = True # ran out of time
        #     return 
        
        # if there is a movement command in queue
        if (len(self.fifoPath) != 0):
            
            # load command from queue
            target = self.fifoPath[0]
            if (self.check_movement_complete(target, 0.1)): # if target has been reached within 0.1 tolerance

                # print(self.fifoPath)
                # print("pop")
                self.fifoPath.pop(0) # remove element from queue
                # print(self.fifoPath)
                self.moving = False

                if (len(self.fifoPath) != 0): # if another element in queue
                    target = self.fifoPath[0]
                    self.moveGeneric(target) # move to target
                    self.moving = True
                    # self.follow_path_callback() # and then run the loop again
        # else:
            # if not, remove any residual callbacks
            # self.simulator.remove_physics_callback("follow_path_callback")

    def goalReached(self):
        tolerance = 0.1
        robotLocation = self.computeRobotPositionRelative()
        if (self.distBetween([robotLocation[0],robotLocation[1]], self.endPoint) < tolerance):
            self.done = True


    def rewardCalculation(self):
        ### https://www.desmos.com/calculator/0e36419059

        # distance between start and finish 
        dist = self.distBetween(self.startPoint,self.endPoint)

        robotLocation = self.computeRobotPositionRelative()

        # print("robot location: " + str(robotLocation))
        robotToStart = self.distBetween(self.startPoint,[robotLocation[0],robotLocation[1]])
        robotToFinish = self.distBetween(self.endPoint, [robotLocation[0],robotLocation[1]])

        # print("start to end:" + str(dist))

        # print("start to robot: " + str(robotToStart))
        # print("robot to end: " + str(robotToFinish))

        totalDistance = robotToStart+robotToFinish
        # print("total distance: " + str(totalDistance))

        return 1 - (robotToFinish/dist)
    
    def observation(self):
        robotLocation = self.computeRobotPositionRelative()
        
        # Calculate distance to goal
        dist_to_goal = self.distBetween([robotLocation[0], robotLocation[1]], self.endPoint)
        
        # Calculate normalized steps (current steps / max steps)
        normalized_steps = self.actionSteps / 60.0  # 60 is the max action steps

        ########  robot x coord  |  robot y coord  | robot yaw value |    end goal x   | end goal y | dist_to_goal | normalized_steps
        return [ robotLocation[0], robotLocation[1], robotLocation[2], self.endPoint[0], self.endPoint[1], dist_to_goal, normalized_steps]
    
    def info(self):
        ########  sim step position 
        return [self.simulator.get_step_count()]


if __name__ == "__main__":
    env_train = IoaiNavEnv(headless=False, seed=11)
    # env_train.simulator.play()
    env_train.simulator.play()
    # while(True):
    #     env_train.simulator.step(1)
    #     env_train.simulator.forward()
    env_train.simulator.loop()


    

# 
#     observation = env.reset() # loads sim settings
#     print("observation: " + str(observation))
# 

# 
#     env.simulator.loop()
# 
#     env.simulator.add_physics_callback("follow_path_callback", env.follow_path_callback)
#     
#     print("Start pos: " + str(env.startPoint))
#     print("End point: " + str(env.endPoint))
#     print("Reward: " + str(env.rewardCalculation()))
# 
#     
#     
#     env.simulator.close()