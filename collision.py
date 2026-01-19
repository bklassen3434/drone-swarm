"""Collision avoidance utilities for the drone swarm system."""

import math
from typing import List, Tuple, Optional

import config
from drone import Drone
from zone import Zone


def calculate_distance_2d(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def calculate_distance_3d(
    pos1: Tuple[float, float, float],
    pos2: Tuple[float, float, float]
) -> float:
    """Calculate 3D Euclidean distance between two points."""
    return math.sqrt(
        (pos1[0] - pos2[0])**2 +
        (pos1[1] - pos2[1])**2 +
        (pos1[2] - pos2[2])**2
    )


def check_zone_violation(drone: Drone) -> bool:
    """
    Check if a drone is outside its assigned zone boundaries.

    Args:
        drone: The drone to check

    Returns:
        True if drone is violating zone boundaries, False otherwise
    """
    zone = drone.get_current_zone()
    if zone is None:
        return False

    return not zone.contains_point(drone.x, drone.y)


def check_collision_pair(drone1: Drone, drone2: Drone) -> bool:
    """
    Check if two drones are too close to each other.

    Args:
        drone1: First drone
        drone2: Second drone

    Returns:
        True if drones are within minimum safe distance
    """
    if not drone1.is_alive() or not drone2.is_alive():
        return False

    distance = calculate_distance_3d(
        drone1.get_position(),
        drone2.get_position()
    )
    return distance < config.MIN_SAFE_DISTANCE


def check_all_collisions(drones: List[Drone]) -> List[Tuple[int, int]]:
    """
    Check all drone pairs for potential collisions.

    Args:
        drones: List of all drones

    Returns:
        List of tuples containing IDs of colliding drone pairs
    """
    collisions = []
    alive_drones = [d for d in drones if d.is_alive()]

    for i in range(len(alive_drones)):
        for j in range(i + 1, len(alive_drones)):
            if check_collision_pair(alive_drones[i], alive_drones[j]):
                collisions.append((alive_drones[i].id, alive_drones[j].id))

    return collisions


def get_closest_drone(drone: Drone, other_drones: List[Drone]) -> Optional[Tuple[Drone, float]]:
    """
    Find the closest drone to a given drone.

    Args:
        drone: The reference drone
        other_drones: List of other drones to check

    Returns:
        Tuple of (closest drone, distance) or None if no other drones
    """
    closest = None
    min_distance = float('inf')

    for other in other_drones:
        if other.id == drone.id or not other.is_alive():
            continue

        distance = calculate_distance_3d(drone.get_position(), other.get_position())
        if distance < min_distance:
            min_distance = distance
            closest = other

    if closest is None:
        return None
    return (closest, min_distance)


def enforce_zone_boundaries(drone: Drone) -> bool:
    """
    Enforce zone boundaries by clamping drone position.

    Args:
        drone: The drone to constrain

    Returns:
        True if position was modified, False otherwise
    """
    zone = drone.get_current_zone()
    if zone is None:
        return False

    modified = False

    # Clamp x position
    if drone.x < zone.x_min:
        drone.x = zone.x_min
        modified = True
    elif drone.x > zone.x_max:
        drone.x = zone.x_max
        modified = True

    # Clamp y position
    if drone.y < zone.y_min:
        drone.y = zone.y_min
        modified = True
    elif drone.y > zone.y_max:
        drone.y = zone.y_max
        modified = True

    return modified


def assign_safe_altitudes(drones: List[Drone]) -> None:
    """
    Assign different altitudes to each drone for vertical separation.

    Args:
        drones: List of drones to assign altitudes to
    """
    for i, drone in enumerate(drones):
        if i < len(config.DRONE_ALTITUDES):
            drone.altitude = config.DRONE_ALTITUDES[i]
        else:
            # Fallback: increment altitude by 2m for each additional drone
            drone.altitude = config.DRONE_ALTITUDES[-1] + (i - len(config.DRONE_ALTITUDES) + 1) * 2


def is_in_buffer_zone(x: float, zones: List[Zone]) -> bool:
    """
    Check if an x-coordinate falls within a buffer zone between assigned zones.

    Args:
        x: The x-coordinate to check
        zones: List of all zones

    Returns:
        True if x is in a buffer zone (between zones)
    """
    for i in range(len(zones) - 1):
        current_zone = zones[i]
        next_zone = zones[i + 1]

        # Buffer zone is between current zone's x_max and next zone's x_min
        if current_zone.x_max < x < next_zone.x_min:
            return True

    return False
