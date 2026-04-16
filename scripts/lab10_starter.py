#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import inf, sqrt, atan2, pi
from time import sleep, time
import queue
import json

import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Twist, Point32, PoseStamped, Pose, Vector3, Quaternion, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion

# AABB format: (x_min, x_max, y_min, y_max)
OBS_TYPE = Tuple[float, float, float, float]
# Position format: {"x": x, "y": y, "theta": theta}
POSITION_TYPE = Dict[str, float]

# don't change this
GOAL_THRESHOLD = 0.1


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


class PIDController:
    """
    Generates control action taking into account instantaneous error (proportional action),
    accumulated error (integral action) and rate of change of error (derivative action).
    """

    def __init__(self, kP, kI, kD, kS, u_min, u_max):
        assert u_min < u_max, "u_min should be less than u_max"
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.err_int = 0
        self.err_dif = 0
        self.err_prev = 0
        self.err_hist = []
        self.t_prev = 0
        self.u_min = u_min
        self.u_max = u_max

    def control(self, err, t):
        dt = t - self.t_prev
        self.err_hist.append(err)
        self.err_int += err
        if len(self.err_hist) > self.kS:
            self.err_int -= self.err_hist.pop(0)
        self.err_dif = err - self.err_prev
        u = (self.kP * err) + (self.kI * self.err_int * dt) + (self.kD * self.err_dif / dt)
        self.err_prev = err
        self.t_prev = t
        return max(self.u_min, min(u, self.u_max))


class Node:
    def __init__(self, position: POSITION_TYPE, parent: "Node"):
        self.position = position
        self.neighbors = []
        self.parent = parent

    def distance_to(self, other_node: "Node") -> float:
        return np.linalg.norm(self.position - other_node.position)

    def to_dict(self) -> Dict:
        return {"x": self.position[0], "y": self.position[1]}

    def __str__(self) -> str:
        return (
            f"Node<pos: {round(self.position[0], 4)}, {round(self.position[1], 4)}, #neighbors: {len(self.neighbors)}>"
        )


class RrtPlanner:

    def __init__(self, obstacles: List[OBS_TYPE], map_aabb: Tuple):
        self.obstacles = obstacles
        self.map_aabb = map_aabb
        self.graph_publisher = rospy.Publisher("/rrt_graph", MarkerArray, queue_size=10)
        self.plan_visualization_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        self.delta = 0.1
        self.obstacle_padding = 0.15
        self.goal_threshold = GOAL_THRESHOLD

    def visualize_plan(self, path: List[Dict]):
        marker_array = MarkerArray()
        for i, waypoint in enumerate(path):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position = Point(waypoint["x"], waypoint["y"], 0.0)
            marker.pose.orientation = Quaternion(0, 0, 0, 1)
            marker.scale = Vector3(0.075, 0.075, 0.1)
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
            marker_array.markers.append(marker)
        self.plan_visualization_pub.publish(marker_array)

    def visualize_graph(self, graph: List[Node]):
        marker_array = MarkerArray()
        for i, node in enumerate(graph):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale = Vector3(0.05, 0.05, 0.05)
            marker.pose.position = Point(node.position[0], node.position[1], 0.01)
            marker.pose.orientation = Quaternion(0, 0, 0, 1)
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.5)
            marker_array.markers.append(marker)
        self.graph_publisher.publish(marker_array)

    def _randomly_sample_q(self) -> Node:
        # Choose uniform randomly sampled points
        ######### Your code starts here #########
        x_min, x_max, y_min, y_max = self.map_aabb

        # Sample a random point; keep resampling if it's in collision
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        q_rand = Node(np.array([x, y]), None)

        while self._is_in_collision(q_rand):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            q_rand = Node(np.array([x, y]), None)

        return q_rand
        ######### Your code ends here #########

    def _nearest_vertex(self, graph: List[Node], q: Node) -> Node:
        # Determine vertex nearest to sampled point
        ######### Your code starts here #########

        ## iterate through all vertices in graph and check dist?
        nearest = None
        min_dist = inf
        for node in graph:
            dist = q.distance_to(node)
            if dist < min_dist:
                min_dist = dist
                nearest = node


        return nearest
        ######### Your code ends here #########

    def _is_in_collision(self, q_rand: Node):
        x = q_rand.position[0]
        y = q_rand.position[1]
        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = obs
            x_min -= self.obstacle_padding
            y_min -= self.obstacle_padding
            x_max += self.obstacle_padding
            y_max += self.obstacle_padding
            if (x_min < x and x < x_max) and (y_min < y and y < y_max):
                return True
        return False

    def _extend(self, graph: List[Node], q_rand: Node):

        # Check if sampled point is in collision and add to tree if not
        ######### Your code starts here #########
        if not self._is_in_collision(q_rand):                                                                
            graph.append(q_rand)

        ######### Your code ends here #########

    def generate_plan(self, start: POSITION_TYPE, goal: POSITION_TYPE) -> Tuple[List[POSITION_TYPE], List[Node]]:
        """Public facing API for generating a plan. Returns the plan and the graph.

        Return format:
            plan:
            [
                {"x": start["x"], "y": start["y"]},
                {"x": ...,      "y": ...},
                            ...
                {"x": goal["x"],  "y": goal["y"]},
            ]
            graph:
                [
                    Node<pos: x1, y1, #neighbors: n_1>,
                    ...
                    Node<pos: x_n, y_n, #neighbors: z>,
                ]
        """
        graph = [Node(np.array([start["x"], start["y"]]), None)]
        goal_node = Node(np.array([goal["x"], goal["y"]]), None)
        plan = []

        # Find path from start to goal location through tree
        ######### Your code starts here #########


        ## Change this value for different tests !!
        num_iterations = 9000


        ## as generate the graph from start to end, set parents (aka previous) for the nodes for backtracking
        
        for k in range(num_iterations):
            q_rand = self._randomly_sample_q()
            q_near = self._nearest_vertex(graph, q_rand)
            dist = (q_rand.position - q_near.position)
            unit = dist/ (np.linalg.norm(dist))
            new_pos = q_near.position + (unit)*self.delta
            q_new = Node(new_pos, q_near)
            self._extend(graph, q_new)

            if graph[-1] is q_new and q_new.distance_to(goal_node) < self.goal_threshold:
                goal_node.parent = q_new
                graph.append(goal_node)
                break
                

        current = graph[-1]## end of graph
        while current != None:
            plan.insert(0, current.to_dict())
            current = current.parent
            
        ######### Your code ends here #########
        return plan, graph


# Protip: copy the ObstacleFreeWaypointController class from lab5.py here
######### Your code starts here #########
class ObstacleFreeWaypointController:
    def __init__(self, waypoints: List[Dict]):
        self.waypoints = waypoints
        # Subscriber to the robot's current position (assuming you have Odometry data)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.waypoint_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        sleep(0.5)  # sleep to give time for rviz to subscribe to /waypoints

        self.current_position = None

        # define linear and angular PID controllers here
        ######### Your code starts here #########
        kP = 2.0
        kD = 0.03
        kI = 0.01
        kS = 0.4
        u_min = -1.5
        u_max = 1.5

        self.angular_controller = PIDController(kP, kI, kD, kS, u_min, u_max)
       
        self.v0 = 0.1 #base forward velocity
        ######### Your code ends here #########

    def odom_callback(self, msg):
        # Extracting current position from Odometry message
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

    def calculate_error(self, goal_position: Dict) -> Optional[Tuple]:
        """Return distance and angle error between the current position and the provided goal_position. Returns None if
        the current position is not available.
        """
        if self.current_position is None:
            return None

        # Calculate error in position and orientation
        ######### Your code starts here #########
               
        ######### CODE FROM LAB 5 BELOW #########

        ## update self.goal position to be from the parameter goal position
        dx = goal_position["x"] - self.current_position["x"]
        dy = goal_position["y"] - self.current_position["y"]
        
        #dist = sqrt(dx^2 + dy^2)
        distance_error = sqrt(dx**2+dy**2)
        
        #angle = arctan(dy/dx)
        goal_angle = atan2(dy, dx)
        current_angle = self.current_position["theta"]

        angle_error = goal_angle - current_angle

        # Ensure angle error is within -pi to pi range
        if angle_error > pi:
            angle_error -= 2 * pi
        elif angle_error < -pi:
            angle_error += 2 * pi
        ######### Your code ends here #########

        return distance_error, angle_error

    def control_robot(self):
        rate = rospy.Rate(20)  # 20 Hz
        ctrl_msg = Twist()

        # initialize first waypoint
        current_waypoint_idx = 0

        while not rospy.is_shutdown():

            # Travel through waypoints one at a time, checking if robot is close enough
            ######### Your code starts here #########
            if current_waypoint_idx >= len(self.waypoints):
                # stop robot 
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
                self.robot_ctrl_pub.publish(ctrl_msg)
                rospy.loginfo("All waypoints reached!")
                break
            

            ## select the goal 
            goal = self.waypoints[current_waypoint_idx]
            error = self.calculate_error(goal)

            if error is None:
                rate.sleep()
                continue
            
            distance_error, angle_error = error
            
            #if error is small
            if abs(distance_error) < 0.05:
                rospy.loginfo(f"Reached waypoint {current_waypoint_idx}: {goal}")
                current_waypoint_idx += 1
                continue
            
            t = rospy.get_time()
            omega = self.angular_controller.control(angle_error, t) #calls the control method in the PID controller class that internally computes ω = kP*error + kI*integral + kD*derivative
            ctrl_msg.angular.z = omega
            ctrl_msg.linear.x = self.v0 * max(0, 1 - abs(angle_error) / pi)               
            
            #publish
            self.robot_ctrl_pub.publish(ctrl_msg)

            ######### Your code ends here #########
            rate.sleep()

######### Your code ends here #########


""" Example usage

rosrun development lab10.py --map_filepath src/csci445l/scripts/lab10_map.json
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()
    with open(args.map_filepath, "r") as f:
        map_ = json.load(f)
        goal_position = map_["goal_position"]
        obstacles = map_["obstacles"]
        map_aabb = map_["map_aabb"]
        start_position = {"x": 0.0, "y": 0.0}

    rospy.init_node("rrt_planner")
    planner = RrtPlanner(obstacles, map_aabb)
    plan, graph = planner.generate_plan(start_position, goal_position)
    planner.visualize_plan(plan)
    planner.visualize_graph(graph)
    controller = ObstacleFreeWaypointController(plan)

    try:
        controller.control_robot()
    except rospy.ROSInterruptException:
        print("Shutting down...")
