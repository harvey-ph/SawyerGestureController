#!/usr/bin/env python3

import rospy
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from intera_interface import Limb, Gripper  # Importing the Limb interface from intera SDK
import PyKDL
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import numpy as np
import threading
import socket
import time

class SawyerController:
    def __init__(self):
        rospy.init_node('sawyer_controller')
        self.rate = rospy.Rate(10)
        # Initialize the current direction and gripper order
        self.current_direction_order = "center"
        self.current_gripper_order = "hold"
        self.distance_movement = 0.03
        self.time_movement = 2

        # Initialize threading and socket
        self.is_running = True
        self.host = "0.0.0.0"
        self.port = 2209

        # Initialize the transform listener for getting the end-effector pose
        self.tf_listener = tf.TransformListener()
        rospy.Subscriber('/robot/joint_states', JointState, self.joint_states_callback)

        # Initialize the limb interface for controlling movements
        self.limb = Limb('right')
        self.gripper = Gripper('right_gripper')
        self.joint_states = None

        # Initialize the KDL chain for the arm
        robot_urdf = URDF.from_parameter_server()
        success, kdl_tree = treeFromUrdfModel(robot_urdf)
        if not success:
            rospy.logerr("Failed to construct KDL tree")
            rospy.signal_shutdown("Failed to construct KDL tree")
            return

        self.arm_chain = kdl_tree.getChain("base", "right_hand")

        # KDL Solvers
        self.fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self.arm_chain)
        self.ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self.arm_chain)
        self.ik_p_kdl = PyKDL.ChainIkSolverPos_NR(self.arm_chain, self.fk_p_kdl, self.ik_v_kdl)

        # Joint names
        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']

        # # PID parameters
        # self.kp = 1.0
        # self.ki = 0.0
        # self.kd = 0.0
        # self.tolerance = 0.01  # Allowable deviation in meters for x, y, z
        # self.integral = [0, 0, 0]
        # self.prev_error = [0, 0, 0]

        # Wait for the first joint states message
        while self.joint_states is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        print("Sawyer Controller initialized.")

    def joint_states_callback(self, msg: JointState):
        '''
             Callback function for the joint states subscriber
        '''
        self.joint_states = msg

    def get_end_effector_pose(self) -> Point:
        '''
             Get the end-effector pose in the base frame
             Returns the position of the end-effector with x, y, z coordinates and the orientation as a quaternion
        '''
        try:
            self.tf_listener.waitForTransform('/base', '/right_hand', rospy.Time(), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform('/base', '/right_hand', rospy.Time(0))
            return trans
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Failed to get end-effector pose")
            return None

    # def pid_control(self, current_pose: list, target_pose: list, axis: int) -> list:
    #      '''
    #           PID controller for enhancing the movement accuracy
    #           Returns the corrected target pose
    #      '''
    #      # Calculate the error, integral, derivative and correction for the specific axis
    #      error = target_pose[axis] - current_pose[axis]
    #      self.integral[axis] += error
    #      derivative = error - self.prev_error[axis]
    #      correction = self.kp * error + self.ki * self.integral[axis] + self.kd * derivative

    #      # Apply PID correction to the target pose on the specific axis
    #      target_pose[axis] += correction
    #      self.prev_error[axis] = error

    #      return target_pose

    def move_end_effector(self, direction: str):
        '''
             Move the end-effector in the specified direction
        '''
        # Get the current end-effector pose
        current_pose = self.get_end_effector_pose()
        if not current_pose:
            return

        # Adjust target based on direction
        target_pose = list(current_pose)

        if direction == 'left':
            target_pose[1] += self.distance_movement
        elif direction == 'right':
            target_pose[1] -= self.distance_movement
        elif direction == 'up':
            target_pose[2] += self.distance_movement
        elif direction == 'down':
            target_pose[2] -= self.distance_movement
        elif direction == 'forward':
            target_pose[0] += self.distance_movement
        elif direction == 'backward':
            target_pose[0] -= self.distance_movement
        elif direction == 'neutral':
            self.limb.move_to_neutral(timeout=2)
            return
        else:
            rospy.logerr("Invalid direction")
            return

        # Get the joint angles for the target pose using PyKDL inverse kinematics
        joint_angles = self.inverse_kinematics(target_pose)
        if joint_angles is None:
            rospy.logerr("Unable to find valid joint solution")
            return

        # Move the joints using the limb interface to ensure movement completion
        joint_dict = dict(zip(self.joint_names, joint_angles))
        self.limb.move_to_joint_positions(joint_dict, timeout=self.time_movement)
        time.sleep(1.5)
        # Print new end-effector pose
        new_pose = self.get_end_effector_pose()
        if new_pose is not None and len(new_pose) > 0:
            print(f"New end-effector position: {new_pose}")

        # For PID controller

        # # Check deviation after movement
        # rospy.sleep(1)  # Allow time for the movement to complete
        # current_pose = self.get_end_effector_pose()
        # if not current_pose:
        #      return

        # # Check if correction is needed (tolerance check for each axis separately)
        # deviation = [abs(target_pose[i] - current_pose[i]) for i in range(3)]
        # for i in range(3):
        #      if deviation[i] > self.tolerance:
        #           target_pose = self.pid_control(current_pose, target_pose, i)

        # # Apply correction if necessary
        # joint_angles = self.inverse_kinematics(target_pose)
        # if joint_angles:
        #      joint_dict = dict(zip(self.joint_names, joint_angles))
        #      self.limb.move_to_joint_positions(joint_dict, timeout=1)

    def inverse_kinematics(self, target_pose: list) -> list:
        '''
             Calculate the inverse kinematics solution for the target pose
             Returns the joint angles for the target pose
        '''

        # Get the current joint angles
        current_joints = PyKDL.JntArray(7)
        for i, name in enumerate(self.joint_names):
            current_joints[i] = self.joint_states.position[self.joint_states.name.index(name)]

        # Set the target position KDL
        target_kdl = PyKDL.Vector(target_pose[0], target_pose[1], target_pose[2])

        # Set the target rotation KDL
        current_rot = PyKDL.Rotation.RPY(0, -np.pi, 0)

        # Create the target frame
        target_frame = PyKDL.Frame(current_rot, target_kdl)

        # Calculate the inverse kinematics solution, return if the solution is valid
        result_angles = PyKDL.JntArray(7)
        if self.ik_p_kdl.CartToJnt(current_joints, target_frame, result_angles) >= 0:
            return list(result_angles)
        else:
            return None

    def change_gripper_state(self):
        '''
             Change the gripper state based on the current order
        '''
        if self.current_gripper_order == 'hold':
            self.gripper.close()
        elif self.current_gripper_order == 'release':
            self.gripper.open()

    def run(self):
        '''
             Main loop for controlling the Sawyer robot
        '''

        while not rospy.is_shutdown() and self.is_running:

            # Change the gripper state based on the current order
            self.change_gripper_state()

            direction = self.current_direction_order
            if direction == 'center':
                continue
            self.move_end_effector(direction)
            self.rate.sleep()

    def start_server(self):
        '''
             Start the server for receiving commands from the client
        '''
        # Create the server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        server_socket.settimeout(5)
        print("Server is running, waiting for client")

        # Accept incoming connections and receive data
        while self.is_running:
            try:
                connection, client_address = server_socket.accept()
                try:
                    # Receive data from the client
                    print(f"Connection from {client_address}")
                    data = connection.recv(16)

                    # Decode the data and set the current direction and gripper order
                    if data:
                        data_decode = str(data.decode('utf-8')).split("_")
                        self.current_direction_order = data_decode[0].lower()
                        self.current_gripper_order = data_decode[1].lower()
                        print(f"Received new value: {self.current_direction_order} - {self.current_gripper_order}")
                finally:
                    connection.close()
            except socket.timeout:
                # If no connection or data received within timeout, set default values
                print("No connection received, setting default values.")
                self.current_direction_order = 'center'
                self.current_gripper_order = 'hold'

    def start(self):
        '''
             Start the controller and server threads
        '''
        threading.Thread(target=self.start_server, daemon=True).start()
        self.run()


if __name__ == '__main__':
    try:
        controller = SawyerController()
        controller.start()
    except rospy.ROSInterruptException:
        pass
