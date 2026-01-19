"""Goal planning for drone navigation in the drone swarm system."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING
from enum import Enum
import math

import config
from coverage_map import CoverageMap, CoverageCell

if TYPE_CHECKING:
    from drone import Drone
    from zone import Zone


class GoalType(Enum):
    """Type of movement goal."""
    COVER_CELL = "cover_cell"  # Move to cover an uncovered cell


@dataclass
class MovementGoal:
    """Represents a target position for the drone to move to."""
    x: float
    y: float
    goal_type: GoalType = GoalType.COVER_CELL
    created_at: float = 0.0  # Simulation time when goal was created

    def distance_to(self, x: float, y: float) -> float:
        """Calculate distance from this goal to a point."""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def is_reached(self, x: float, y: float, threshold: float = None) -> bool:
        """Check if a position has reached this goal."""
        if threshold is None:
            threshold = config.COVERAGE_CELL_SIZE * 0.5
        return self.distance_to(x, y) <= threshold


class GoalPlanner:
    """
    Decides the next best cell for a drone to cover.

    Uses strategies like nearest-uncovered, frontier-expansion, etc.
    """

    def __init__(self, strategy: str = "nearest"):
        """
        Initialize the goal planner.

        Args:
            strategy: Planning strategy to use
                - "nearest": Go to nearest uncovered cell
                - "frontier": Expand from covered area boundary
                - "systematic": Cover in a systematic pattern
        """
        self.strategy = strategy

    def select_next_goal(
        self,
        drone: 'Drone',
        coverage_map: CoverageMap,
        zone: Optional['Zone'] = None,
        claimed_goals: Optional[List[Tuple[float, float]]] = None,
        sim_time: float = 0.0
    ) -> Optional[MovementGoal]:
        """
        Select the next goal for a drone to move toward.

        Args:
            drone: The drone needing a goal
            coverage_map: Current coverage state
            zone: Optional zone to constrain goal selection
            claimed_goals: Goals already claimed by other drones (to avoid)
            sim_time: Current simulation time

        Returns:
            MovementGoal or None if no goal available
        """
        if claimed_goals is None:
            claimed_goals = []

        # Get uncovered cells (optionally within zone)
        if zone is not None:
            uncovered = coverage_map.get_uncovered_in_zone(
                zone.x_min, zone.x_max, zone.y_min, zone.y_max
            )
        else:
            uncovered = coverage_map.get_uncovered_cells()

        if not uncovered:
            return None

        # Filter out claimed goals
        if claimed_goals:
            available = []
            for cell in uncovered:
                is_claimed = False
                for claimed_x, claimed_y in claimed_goals:
                    if abs(cell.x - claimed_x) < config.COVERAGE_CELL_SIZE and \
                       abs(cell.y - claimed_y) < config.COVERAGE_CELL_SIZE:
                        is_claimed = True
                        break
                if not is_claimed:
                    available.append(cell)
            uncovered = available if available else uncovered  # Fall back if all claimed

        if not uncovered:
            return None

        # Select cell based on strategy
        if self.strategy == "nearest":
            target_cell = self._select_nearest(drone.x, drone.y, uncovered)
        elif self.strategy == "frontier":
            target_cell = self._select_frontier(drone.x, drone.y, uncovered, coverage_map)
        elif self.strategy == "systematic":
            target_cell = self._select_systematic(drone, uncovered, zone)
        else:
            target_cell = self._select_nearest(drone.x, drone.y, uncovered)

        if target_cell is None:
            return None

        return MovementGoal(
            x=target_cell.x,
            y=target_cell.y,
            goal_type=GoalType.COVER_CELL,
            created_at=sim_time
        )

    def _select_nearest(
        self,
        x: float,
        y: float,
        candidates: List[CoverageCell]
    ) -> Optional[CoverageCell]:
        """Select the nearest uncovered cell."""
        if not candidates:
            return None

        nearest = None
        min_dist = float('inf')

        for cell in candidates:
            dist = cell.distance_to(x, y)
            if dist < min_dist:
                min_dist = dist
                nearest = cell

        return nearest

    def _select_frontier(
        self,
        x: float,
        y: float,
        candidates: List[CoverageCell],
        coverage_map: CoverageMap
    ) -> Optional[CoverageCell]:
        """Select uncovered cell on the frontier of covered area."""
        # Find cells adjacent to covered cells
        frontier_cells = []

        for cell in candidates:
            # Check if any neighbor is covered
            has_covered_neighbor = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = coverage_map.get_cell_by_grid(
                        cell.grid_x + dc,
                        cell.grid_y + dr
                    )
                    if neighbor and neighbor.covered:
                        has_covered_neighbor = True
                        break
                if has_covered_neighbor:
                    break

            if has_covered_neighbor:
                frontier_cells.append(cell)

        # If we have frontier cells, pick the nearest one
        if frontier_cells:
            return self._select_nearest(x, y, frontier_cells)

        # Otherwise fall back to nearest overall
        return self._select_nearest(x, y, candidates)

    def _select_systematic(
        self,
        drone: 'Drone',
        candidates: List[CoverageCell],
        zone: Optional['Zone']
    ) -> Optional[CoverageCell]:
        """
        Select cell in a systematic pattern (like modified lawnmower).

        Prefers cells in the same row, moving to the next row when complete.
        """
        if not candidates:
            return None

        # Group by row
        rows = {}
        for cell in candidates:
            if cell.grid_y not in rows:
                rows[cell.grid_y] = []
            rows[cell.grid_y].append(cell)

        # Find the row with the smallest y that has uncovered cells
        # (start from bottom, work up)
        sorted_rows = sorted(rows.keys())

        for row_idx in sorted_rows:
            row_cells = rows[row_idx]
            if row_cells:
                # In this row, pick the nearest cell
                return self._select_nearest(drone.x, drone.y, row_cells)

        return self._select_nearest(drone.x, drone.y, candidates)
