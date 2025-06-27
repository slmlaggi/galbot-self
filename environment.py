from physics_simulator import PhysicsSimulator
from physics_simulator.galbot_interface import GalbotInterface, GalbotInterfaceConfig
from physics_simulator.utils.data_types import JointTrajectory
from synthnova_config import PhysicsSimulatorConfig, RobotConfig
import numpy as np

from pathlib import Path

def interpolate_joint_positions(start_positions, end_positions, steps):
    return np.linspace(start_positions, end_positions, steps)

class environment():
    def setup_sim(self):
        
        # Create sim config
        my_config = PhysicsSimulatorConfig()

        # Instantiate the simulator
        self.simulator = PhysicsSimulator(my_config)

        # Add default ground plane if you need
        self.simulator.add_default_scene()

        # Add robot
        self.robot_config = RobotConfig(
            prim_path="/World/Galbot",
            name="galbot_one_charlie",
            mjcf_path=Path()
            .joinpath(self.simulator.synthnova_assets_directory)
            .joinpath("synthnova_assets")
            .joinpath("robot")
            .joinpath("galbot_one_charlie_description")
            .joinpath("galbot_one_charlie.xml"),
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1]
        )
        self.robot_path = self.simulator.add_robot(self.robot_config)

        # Initialize the simulator
        self.simulator.initialize()

    def init_scene(self):
        print("scene initialized")
    
    def init_pose(self):
        # Init head pose
        self.head_init_pos = [0.0, 0.0]
        self._move_joints_to_target(self.interface.head, self.head_init_pos)

        # Init leg pose
        self.leg_init_pos = [0.43, 1.48, 1.07, 0.0]
        self._move_joints_to_target(self.interface.leg, self.leg_init_pos)

        # Init left arm pose
        self.left_arm_init_pos = [
            0.058147381991147995,
            1.4785659313201904,
            -0.0999724417924881,
            -2.097979784011841,
            1.3999720811843872,
            -0.009971064515411854,
            1.0999830961227417,
        ]
        self._move_joints_to_target(self.interface.left_arm, self.left_arm_init_pos)

        # Init right arm pose
        self.right_arm_init_pos = [
            -0.058147381991147995,
            -1.4785659313201904,
            0.0999724417924881,
            2.097979784011841,
            -1.3999720811843872,
            0.009971064515411854,
            -1.0999830961227417,
        ]
        self._move_joints_to_target(self.interface.right_arm, self.right_arm_init_pos)
    
    def init_interface(self):
        galbot_interface_config = GalbotInterfaceConfig()

        galbot_interface_config.robot.prim_path = "/World/Galbot"

        robot_name = "galbot_one_charlie"
        # Enable modules
        galbot_interface_config.modules_manager.enabled_modules.append("right_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("left_arm")
        galbot_interface_config.modules_manager.enabled_modules.append("leg")
        galbot_interface_config.modules_manager.enabled_modules.append("head")
        galbot_interface_config.modules_manager.enabled_modules.append("chassis")

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

    def _move_joints_to_target(self, module, target_positions, steps=200):
        current_positions = module.get_joint_positions()
        positions = interpolate_joint_positions(current_positions, target_positions, steps)
        joint_trajectory = JointTrajectory(positions=np.array(positions))
        module.follow_trajectory(joint_trajectory)

    def check_movement_complete(self, target, threshold):
        current_joint_positions = self.interface.chassis.get_joint_positions()
        distance = np.linalg.norm(np.array(current_joint_positions) - np.array(target))
        return distance < threshold

    def moveGeneric(self, target):
        self.moving = True
        current_joint_positions = self.interface.chassis.get_joint_positions()
        target_joint_positions = [target[0], target[1], target[2]]
        positions = interpolate_joint_positions(
            current_joint_positions, target_joint_positions, 200
        )
        # Create a joint trajectory
        joint_trajectory = JointTrajectory(positions=positions)

        # Follow the trajectory
        self.interface.chassis.follow_trajectory(joint_trajectory)

    def follow_path_callback(self):
        if (len(self.fifoPath) != 0):

            target = self.fifoPath[0]
            if (self.check_movement_complete(target, 0.1)):
                self.fifoPath.pop(0)
                self.moving = False

                if (len(self.fifoPath) != 0):
                    target = self.fifoPath[0]
                    self.moveGeneric(target)
                    self.moving = True

    def if_pose_initialized_callback(self):
        left_arm_ready = np.allclose(
            self.interface.left_arm.get_joint_positions(), 
            self.left_arm_init_pos, 
            atol=0.1
        )
        right_arm_ready = np.allclose(
            self.interface.right_arm.get_joint_positions(), 
            self.right_arm_init_pos, 
            atol=0.1
        )
        leg_ready = np.allclose(
            self.interface.leg.get_joint_positions(), 
            self.leg_init_pos, 
            atol=0.1
        )
        head_ready = np.allclose(
            self.interface.head.get_joint_positions(), 
            self.head_init_pos, 
            atol=0.1
        )
        
        # All parts must be in position for initialization to be complete
        if left_arm_ready and right_arm_ready and leg_ready and head_ready:
            self.pose_initialized = True

    # TODO: write Walk in 4 directions and yaw rotation left/right
 
    def main(self):

        self.setup_sim()
        self.init_interface()
        # Start the simulation

        self.pose_initialized = False
        self.init_pose()
        self.init_scene()

        self.simulator.add_physics_callback("init_pose_callback", self.if_pose_initialized_callback)
        while not self.pose_initialized:
            self.simulator.step()
        self.simulator.add_physics_callback("follow_path_callback", self.follow_path_callback)
        self.moving = False
        self.fifoPath = [[0, 0, 0], [1, 6, 1.5], 0]
        self.simulator.remove_physics_callback("init_pose_callback") # save compute
        
        self.simulator.step()

        # # Get current joint positions
        # current_joint_positions = self.galbot_interface.chassis.get_joint_positions() # intially [0, 0, 0] ([x,y,yaw], add z in more dimensions)

        # # Define target joint positions
        # target_joint_positions = [1, 6, 1.5]

        # # Interpolate joint positions
        # positions = interpolate_joint_positions(
        #     current_joint_positions, target_joint_positions, 5000
        # )
        # # Create a joint trajectory
        # joint_trajectory = JointTrajectory(positions=positions)

        # # Follow the trajectory
        # self.galbot_interface.chassis.follow_trajectory(joint_trajectory)

        # Run the display loop
        while True:
            self.simulator.step()

        # Close the simulator
        self.simulator.close()


test = environment()
test.main()
