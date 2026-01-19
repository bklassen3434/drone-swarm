"""Zone and region partitioning logic for the drone swarm system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import config


class ZoneStatus(Enum):
    """Status of a zone's coverage."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


@dataclass
class Zone:
    """Represents a rectangular zone for a drone to photograph."""
    id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    status: ZoneStatus = ZoneStatus.PENDING
    assigned_drone_id: Optional[int] = None
    coverage_progress: float = 0.0  # 0.0 to 1.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within this zone."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def get_start_position(self) -> tuple:
        """Get the starting position for lawnmower pattern (bottom-left)."""
        return (self.x_min + config.LAWNMOWER_STRIP_WIDTH / 2, self.y_min)


@dataclass
class Region:
    """Represents the overall rectangular region to be photographed."""
    width: float = config.REGION_WIDTH
    height: float = config.REGION_HEIGHT
    x_origin: float = 0.0
    y_origin: float = 0.0

    @property
    def x_min(self) -> float:
        return self.x_origin

    @property
    def x_max(self) -> float:
        return self.x_origin + self.width

    @property
    def y_min(self) -> float:
        return self.y_origin

    @property
    def y_max(self) -> float:
        return self.y_origin + self.height


def partition_region(region: Region, num_zones: int, buffer: float = config.ZONE_BUFFER) -> List[Zone]:
    """
    Divide a region into vertical strips with buffer gaps between them.

    Args:
        region: The region to partition
        num_zones: Number of zones to create
        buffer: Gap between zones (no-fly zone)

    Returns:
        List of Zone objects
    """
    zones = []

    # Calculate total buffer space needed
    total_buffer = buffer * (num_zones - 1)

    # Calculate width available for zones
    available_width = region.width - total_buffer
    zone_width = available_width / num_zones

    current_x = region.x_min

    for i in range(num_zones):
        zone = Zone(
            id=i,
            x_min=current_x,
            x_max=current_x + zone_width,
            y_min=region.y_min,
            y_max=region.y_max
        )
        zones.append(zone)

        # Move to next zone (add zone width + buffer)
        current_x += zone_width + buffer

    return zones


def split_zone(zone: Zone, num_parts: int = 2) -> List[Zone]:
    """
    Split an existing zone into smaller vertical strips.
    Used when reassigning a failed drone's zone to remaining drones.

    Args:
        zone: The zone to split
        num_parts: Number of parts to split into

    Returns:
        List of new Zone objects
    """
    new_zones = []
    part_width = zone.width / num_parts

    for i in range(num_parts):
        new_zone = Zone(
            id=zone.id * 100 + i,  # Generate unique ID
            x_min=zone.x_min + i * part_width,
            x_max=zone.x_min + (i + 1) * part_width,
            y_min=zone.y_min,
            y_max=zone.y_max,
            status=zone.status,
            coverage_progress=0.0  # Reset progress for new assignment
        )
        new_zones.append(new_zone)

    return new_zones
