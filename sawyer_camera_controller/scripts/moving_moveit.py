#!/usr/bin/env python3

import rospy
import sys
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
import moveit_commander
from geometry_msgs.msg import Pose
import tf
from sensor_msgs.msg import JointState

class SawyerDirectionPlanner:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('sawyer_direction_planner', anonymous=True)

        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group_name = "right_arm"
        self.move_group = MoveGroupCommander(self.group_name)


        # Set the distance to move (in meters)
        self.move_distance = 0.1  # 5 cm

        # For getting current pose:
        self.tf_listener = tf.TransformListener()
        rospy.Subscriber('/robot/joint_states', JointState, self.joint_states_callback)

        # Dictionary to map directions to coordinate changes
        self.direction_map = {
            'left': (-self.move_distance, 0, 0),
            'right': (self.move_distance, 0, 0),
            'forward': (0, self.move_distance, 0),
            'backward': (0, -self.move_distance, 0),
            'up': (0, 0, self.move_distance),
            'down': (0, 0, -self.move_distance)
        }

    # def get_current_pos_pro(self, planner_id):
    #     self.move_group.set_planner_id(planner_id)
    #     self.move_group.set_planning_time(5.0)
    #     print(self.move_group.get_current_pose().pose)

    def joint_states_callback(self, msg):
        self.joint_states = msg

    def get_current_pos(self, planner_id):
        try:
            self.tf_listener.waitForTransform('/base', '/right_hand', rospy.Time(), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform('/base', '/right_hand', rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Failed to get end-effector pose")
            return None


    def plan_and_move(self, direction, planner_id):
        self.move_group.set_planner_id(planner_id)
        self.move_group.set_planning_time(5.0)  # Set planning time to 5 seconds
        # self.move_group.set_start_state_to_current_state()
        # Get current pose
        current_pose, current_orient = self.get_current_pos(planner_id)
        # Calculate new pose based on direction
        dx, dy, dz = self.direction_map.get(direction, (0, 0, 0))
        new_pose = Pose()
        new_pose.position.x = current_pose[0] + dx
        new_pose.position.y = current_pose[1] + dy
        new_pose.position.z = current_pose[2] + dz
        new_pose.orientation.x = current_orient[0]
        new_pose.orientation.y = current_orient[1]
        new_pose.orientation.z = current_orient[2]
        new_pose.orientation.w = current_orient[3]

        # Set the new pose as the target
        self.move_group.set_pose_target(new_pose)

        # Plan and execute
        success, trajectory, planning_time, error_code = self.move_group.plan()
        # print(f'trajectory: {trajectory}')
        if success:
            rospy.loginfo(f"Path planning successful with {planner_id}")
            self.move_group.execute(trajectory, wait=True)
        else:
            rospy.logwarn(f"Path planning failed with {planner_id}")

        self.move_group.clear_pose_targets()

        return success


def main():
    sawyer = SawyerDirectionPlanner()

    # "RRTConnect", "RRTstar", "PRM", "LazyPRM", "LazyPRMstar", "SPARS", "SPARStwo", "PRMstar", "TRRT", "EST", "BKPIECE", "LBKPIECE", "KPIECE", "BiTRRT", "STRIDE", "PDST", "FMT", "BFMT", "STOMP"
    planner = "LazyPRM"
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():

        direction = input("Enter direction (left/right/up/down/forward/backward) or 'q' to quit: ").lower()

        if direction == 'quit':
            break

        if direction not in sawyer.direction_map:
            print("Invalid direction. Please try again.")
            continue

        rospy.loginfo(f"Attempting to plan and move {direction} using {planner}")


        success = sawyer.plan_and_move(direction, planner)
        if success:
            rospy.loginfo(f"Successfully planned and executed with {planner}")
        else:
            rospy.logwarn(f"Failed to plan with {planner}")

        if not success:
            rospy.logerr("Failed to plan with any available planner")
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass