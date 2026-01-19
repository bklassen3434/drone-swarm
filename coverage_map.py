"""2D coverage map for tracking photographed cells in the drone swarm system."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set
import math

import config


@dataclass
class CoverageCell:
    """Represents a single cell in the coverage grid."""
    x: float  # Center x coordinate
    y: float  # Center y coordinate
    grid_x: int  # Grid column index
    grid_y: int  # Grid row index
    covered: bool = False
    covered_by: Optional[int] = None  # Drone ID that covered this cell

    def distance_to(self, x: float, y: float) -> float:
        """Calculate distance from this cell's center to a point."""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)


@dataclass
class CoverageMap:
    """
    2D grid tracking which cells have been photographed.

    The map divides the region into cells of COVERAGE_CELL_SIZE.
    Each cell tracks whether it has been covered and by which drone.
    """
    width: float
    height: float
    cell_size: float = config.COVERAGE_CELL_SIZE
    x_origin: float = 0.0
    y_origin: float = 0.0
    cells: List[List[CoverageCell]] = field(default_factory=list)
    _covered_count: int = 0
    _total_cells: int = 0

    def __post_init__(self):
        """Initialize the grid of cells."""
        self._init_grid()

    def _init_grid(self) -> None:
        """Create the 2D grid of coverage cells."""
        self.num_cols = max(1, int(math.ceil(self.width / self.cell_size)))
        self.num_rows = max(1, int(math.ceil(self.height / self.cell_size)))
        self._total_cells = self.num_cols * self.num_rows
        self._covered_count = 0

        self.cells = []
        for row in range(self.num_rows):
            row_cells = []
            for col in range(self.num_cols):
                # Calculate center of cell
                center_x = self.x_origin + (col + 0.5) * self.cell_size
                center_y = self.y_origin + (row + 0.5) * self.cell_size

                cell = CoverageCell(
                    x=center_x,
                    y=center_y,
                    grid_x=col,
                    grid_y=row
                )
                row_cells.append(cell)
            self.cells.append(row_cells)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        col = int((x - self.x_origin) / self.cell_size)
        row = int((y - self.y_origin) / self.cell_size)

        # Clamp to valid range
        col = max(0, min(col, self.num_cols - 1))
        row = max(0, min(row, self.num_rows - 1))

        return col, row

    def get_cell(self, x: float, y: float) -> Optional[CoverageCell]:
        """Get the cell containing the given world coordinates."""
        col, row = self.world_to_grid(x, y)
        if 0 <= row < self.num_rows and 0 <= col < self.num_cols:
            return self.cells[row][col]
        return None

    def get_cell_by_grid(self, grid_x: int, grid_y: int) -> Optional[CoverageCell]:
        """Get cell by grid indices."""
        if 0 <= grid_y < self.num_rows and 0 <= grid_x < self.num_cols:
            return self.cells[grid_y][grid_x]
        return None

    def mark_covered(self, x: float, y: float, drone_id: int,
                     radius: Optional[float] = None) -> List[CoverageCell]:
        """
        Mark a cell (or cells within radius) as covered.

        Args:
            x: World x coordinate
            y: World y coordinate
            drone_id: ID of the drone covering this cell
            radius: Optional radius to mark multiple cells (for photo coverage area)

        Returns:
            List of cells that were newly marked as covered
        """
        newly_covered = []

        if radius is None:
            # Mark single cell
            cell = self.get_cell(x, y)
            if cell and not cell.covered:
                cell.covered = True
                cell.covered_by = drone_id
                self._covered_count += 1
                newly_covered.append(cell)
        else:
            # Mark all cells within radius
            # Calculate grid range to check
            min_col, _ = self.world_to_grid(x - radius, y)
            max_col, _ = self.world_to_grid(x + radius, y)
            _, min_row = self.world_to_grid(x, y - radius)
            _, max_row = self.world_to_grid(x, y + radius)

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    cell = self.get_cell_by_grid(col, row)
                    if cell and not cell.covered:
                        if cell.distance_to(x, y) <= radius:
                            cell.covered = True
                            cell.covered_by = drone_id
                            self._covered_count += 1
                            newly_covered.append(cell)

        return newly_covered

    def get_uncovered_cells(self) -> List[CoverageCell]:
        """Get all cells that have not been covered yet."""
        uncovered = []
        for row in self.cells:
            for cell in row:
                if not cell.covered:
                    uncovered.append(cell)
        return uncovered

    def get_uncovered_in_zone(self, x_min: float, x_max: float,
                               y_min: float, y_max: float) -> List[CoverageCell]:
        """Get uncovered cells within a rectangular zone."""
        uncovered = []

        # Calculate grid bounds for the zone
        min_col, min_row = self.world_to_grid(x_min, y_min)
        max_col, max_row = self.world_to_grid(x_max, y_max)

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell = self.get_cell_by_grid(col, row)
                if cell and not cell.covered:
                    # Verify cell center is actually within zone
                    if x_min <= cell.x <= x_max and y_min <= cell.y <= y_max:
                        uncovered.append(cell)

        return uncovered

    def get_coverage_percentage(self) -> float:
        """Get percentage of cells that have been covered."""
        if self._total_cells == 0:
            return 100.0
        return (self._covered_count / self._total_cells) * 100.0

    def get_coverage_stats(self) -> dict:
        """Get detailed coverage statistics."""
        covered_by_drone = {}
        for row in self.cells:
            for cell in row:
                if cell.covered and cell.covered_by is not None:
                    drone_id = cell.covered_by
                    covered_by_drone[drone_id] = covered_by_drone.get(drone_id, 0) + 1

        return {
            'total_cells': self._total_cells,
            'covered_cells': self._covered_count,
            'uncovered_cells': self._total_cells - self._covered_count,
            'coverage_percentage': self.get_coverage_percentage(),
            'covered_by_drone': covered_by_drone,
            'grid_size': (self.num_cols, self.num_rows),
            'cell_size': self.cell_size
        }
