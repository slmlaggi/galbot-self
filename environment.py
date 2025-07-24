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
        
        # Initialize tracking variables
        self.previousDistance = self.distBetween(self.startPoint, self.endPoint)

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
        """Reset environment to initial state"""
        # Reset episode variables
        self.done = False
        self.actionSteps = 0
        
        # Generate new random start and end points
        random.seed(self.seed + self.actionSteps)  # Add variation each episode
        while True:
            self.startPoint = [random.uniform(-2, 4), random.uniform(-3, 3)]
            self.endPoint = [random.uniform(-2, 4), random.uniform(-3, 3)]
            if self.distBetween(self.startPoint, self.endPoint) >= 1.5:
                break

        printEnv("Moving robot back to start...")
        
        # Clear movement queue
        if len(self.fifoPath) != 0:
            self.fifoPath = []
        self.fifoPath.append(self.computeRobotPositionRelative())
        self.fifoPath.append([self.startPoint[0], self.startPoint[1], 0])

        # Reset robot position
        self.interface.chassis.set_joint_positions([0, 0, 0], True)
        
        # Wait for robot to reach start position
        stepCount = self.simulator.get_step_count()
        attempts = 0
        max_reset_steps = 300  # Reduced from 500 for faster resets
        
        while not self.check_movement_complete([self.startPoint[0], self.startPoint[1], 0], 0.1):
            self.simulator.step()
            
            if (max_reset_steps <= (self.simulator.get_step_count() - stepCount) and attempts < 5):
                printEnv("Reset taking too long, trying again...")
                self.interface.chassis.set_joint_positions([0, 0, 0], True)
                stepCount = self.simulator.get_step_count()
                attempts += 1
            elif attempts >= 5:
                printEnv("Hard reset required - simulation reset")
                self.simulator.reset()
                break
                
        self.stepOffset = self.simulator.get_step_count()
        
        # Reset tracking variables
        self.previousDistance = self.distBetween([self.startPoint[0], self.startPoint[1]], self.endPoint)
        
        printEnv("Environment reset complete")
        return self.observation()

    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: Integer 0-5 representing movement direction
            
        Returns:
            observation, reward, done, info
        """        
        # Movement setup
        if len(self.fifoPath) == 0:
            self.fifoPath.append(self.computeRobotPositionRelative())
        
        globalStepDistance = 0.2  # Must be greater than tolerance (0.1)

        # Execute action
        action_map = {
            0: ("moveForward", "forwards"),
            1: ("moveBackwards", "backwards"), 
            2: ("moveLeft", "left"),
            3: ("moveRight", "right"),
            4: ("shiftYaw", "yaw shift positive"),
            5: ("shiftYaw", "yaw shift negative")
        }
        
        if action in action_map:
            method_name, action_desc = action_map[action]
            if action == 4:
                getattr(self, method_name)(globalStepDistance)
            elif action == 5:
                getattr(self, method_name)(-globalStepDistance)
            else:
                getattr(self, method_name)(globalStepDistance)
            printEnv(action_desc)
        else:
            printEnv(f"Invalid action: {action}")
            
        self.moving = True

        # Step simulation until robot stops moving
        startTime = self.simulator.get_step_count() - self.stepOffset
        timeout_steps = 300  # Reduced from 500 for faster training
        
        while self.moving:
            if timeout_steps <= ((self.simulator.get_step_count() - self.stepOffset) - startTime):
                # Movement took too long - likely hit wall or stuck
                printEnv("Movement timeout - robot may have hit obstacle")
                if self.actionSteps == 0:
                    # Reset if this is the first action
                    return self.reset(), 0, True, self.info()
                else:
                    # End episode with penalty
                    self.done = True
                    return self.observation(), -1.0, True, self.info()
            self.simulator.step()

        self.actionSteps += 1
        printEnv(f"Action {self.actionSteps} complete. Sim time: {self.simulator.get_step_count() - self.stepOffset}")
        
        # Check episode termination conditions
        if self.actionSteps >= 60:  # Max steps per episode
            self.done = True
            printEnv("Episode ended: Maximum steps reached")
            
        if self.goalReached():
            printEnv("Episode ended: Goal reached!")
            
        # Calculate reward and return step results
        reward = self.rewardCalculation()
        return self.observation(), reward, self.done, self.info()

        
        
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
        """Check if robot has reached the goal within tolerance"""
        tolerance = 0.15  # Slightly increased tolerance for more robust goal detection
        robotLocation = self.computeRobotPositionRelative()
        distance_to_goal = self.distBetween([robotLocation[0], robotLocation[1]], self.endPoint)
        
        if distance_to_goal < tolerance:
            self.done = True
            printEnv(f"Goal reached! Distance: {distance_to_goal:.3f}")
            return True
        return False


    def rewardCalculation(self):
        """
        Calculate reward based on:
        1. Distance to goal (primary reward)
        2. Step efficiency penalty
        3. Goal reached bonus
        4. Wall collision penalty
        
        Returns a reward between -1 and 10 (with 10 being goal reached)
        """
        robotLocation = self.computeRobotPositionRelative()
        
        # Distance from robot to goal
        robotToGoal = self.distBetween(self.endPoint, [robotLocation[0], robotLocation[1]])
        
        # Base distance reward (normalized by initial distance)
        initialDistance = self.distBetween(self.startPoint, self.endPoint)
        distanceReward = max(0, 1 - (robotToGoal / initialDistance))
        
        # Step efficiency penalty (encourage fewer steps)
        stepPenalty = -0.01 * self.actionSteps
        
        # Goal reached bonus
        goalBonus = 0
        if self.goalReached():
            goalBonus = 10.0 - (0.1 * self.actionSteps)  # Bonus decreases with more steps
            
        # Wall collision penalty (if robot hits boundaries)
        wallPenalty = 0
        if (robotLocation[0] < -3.5 or robotLocation[0] > 5.5 or 
            robotLocation[1] < -4.5 or robotLocation[1] > 4.5):
            wallPenalty = -2.0
            self.done = True  # End episode on wall collision
            
        # Previous distance for progress tracking
        if not hasattr(self, 'previousDistance'):
            self.previousDistance = robotToGoal
            
        # Progress reward (positive if getting closer, negative if moving away)
        progressReward = 0.5 * (self.previousDistance - robotToGoal)
        self.previousDistance = robotToGoal
        
        total_reward = distanceReward + stepPenalty + goalBonus + wallPenalty + progressReward
        return np.clip(total_reward, -1.0, 10.0)
    
    def observation(self):
        """
        Return current environment observation
        
        Returns:
            np.array: [robot_x, robot_y, robot_yaw, goal_x, goal_y, distance_to_goal, normalized_steps]
        """
        robotLocation = self.computeRobotPositionRelative()
        distance_to_goal = self.distBetween([robotLocation[0], robotLocation[1]], self.endPoint)
        normalized_steps = self.actionSteps / 60.0  # Normalize by max steps
        
        obs = np.array([
            robotLocation[0], 
            robotLocation[1], 
            robotLocation[2],  # yaw
            self.endPoint[0],
            self.endPoint[1],
            distance_to_goal,
            normalized_steps
        ], dtype=np.float32)
        
        return obs
    
    def info(self):
        """Return additional info about the environment state"""
        robotLocation = self.computeRobotPositionRelative()
        return {
            'sim_steps': self.simulator.get_step_count(),
            'episode_steps': self.actionSteps,
            'robot_position': robotLocation,
            'goal_position': self.endPoint,
            'distance_to_goal': self.distBetween([robotLocation[0], robotLocation[1]], self.endPoint)
        }


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