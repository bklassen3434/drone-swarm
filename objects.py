"""Simulated moving objects for drone tracking."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import random
import math
from uuid import uuid4


@dataclass
class SimulatedObject:
    """A moving object in the simulation that drones can detect and track."""
    object_id: str
    x: float
    y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 2.0  # meters per second
    detected: bool = False
    detected_by: Optional[str] = None  # drone_id that first detected
    current_zone_id: Optional[str] = None

    # Lifetime management
    age: float = 0.0
    lifetime: float = 20.0  # How long before it despawns (seconds)
    spawn_grace_period: float = 3.0  # Seconds before object can be detected

    # Movement behavior
    _target_x: Optional[float] = None
    _target_y: Optional[float] = None
    _pause_duration: float = 0.0

    def can_be_detected(self) -> bool:
        """Check if object has passed spawn grace period."""
        return self.age >= self.spawn_grace_period

    def is_expired(self) -> bool:
        """Check if object has exceeded its lifetime."""
        return self.age >= self.lifetime

    def update(self, dt: float, region_width: float, region_height: float) -> None:
        """Update object position."""
        self.age += dt

        if self._pause_duration > 0:
            self._pause_duration -= dt
            return

        if self._target_x is None or self._reached_target():
            self._pick_new_target(region_width, region_height)
            self._pause_duration = random.uniform(1.0, 5.0)
            return

        dx = self._target_x - self.x
        dy = self._target_y - self.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > 0.1:
            self.velocity_x = (dx / distance) * self.speed
            self.velocity_y = (dy / distance) * self.speed
            move_dist = min(self.speed * dt, distance)
            self.x += (dx / distance) * move_dist
            self.y += (dy / distance) * move_dist

        self.x = max(0, min(region_width, self.x))
        self.y = max(0, min(region_height, self.y))

    def _reached_target(self) -> bool:
        if self._target_x is None:
            return True
        dx = self._target_x - self.x
        dy = self._target_y - self.y
        return math.sqrt(dx * dx + dy * dy) < 1.0

    def _pick_new_target(self, region_width: float, region_height: float) -> None:
        if random.random() < 0.3:
            self._target_x = random.uniform(0, region_width)
            self._target_y = random.uniform(0, region_height)
        else:
            angle = math.atan2(self.velocity_y, self.velocity_x) if (self.velocity_x or self.velocity_y) else random.uniform(0, 2 * math.pi)
            angle += random.uniform(-math.pi / 4, math.pi / 4)
            distance = random.uniform(5, 20)
            self._target_x = max(0, min(region_width, self.x + math.cos(angle) * distance))
            self._target_y = max(0, min(region_height, self.y + math.sin(angle) * distance))

    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, x: float, y: float) -> float:
        dx = self.x - x
        dy = self.y - y
        return math.sqrt(dx * dx + dy * dy)


@dataclass
class ObjectManager:
    """Manages simulated objects in the simulation."""
    region_width: float
    region_height: float
    objects: List[SimulatedObject] = field(default_factory=list)
    _spawn_timer: float = 0.0
    _spawn_interval: float = 15.0
    _max_objects: int = 3

    def update(self, dt: float) -> List[SimulatedObject]:
        """Update all objects, despawn expired ones, and spawn new ones."""
        for obj in self.objects:
            obj.update(dt, self.region_width, self.region_height)

        expired = [obj for obj in self.objects if obj.is_expired()]
        for obj in expired:
            print(f"[Object] {obj.object_id} despawned after {obj.age:.1f}s")
            self.objects.remove(obj)

        self._spawn_timer += dt
        if self._spawn_timer >= self._spawn_interval and len(self.objects) < self._max_objects:
            self._spawn_timer = 0.0
            if random.random() < 0.7:
                self.spawn_object()

        return self.objects

    def spawn_object(self, x: float = None, y: float = None,
                     lifetime: float = None) -> SimulatedObject:
        """Spawn a new object at the given position or random location."""
        if x is None:
            x = random.uniform(5, self.region_width - 5)
        if y is None:
            y = random.uniform(5, self.region_height - 5)
        if lifetime is None:
            lifetime = random.uniform(15.0, 30.0)

        obj = SimulatedObject(
            object_id=f"obj_{uuid4().hex[:6]}",
            x=x,
            y=y,
            speed=random.uniform(1.0, 3.0),
            lifetime=lifetime,
            spawn_grace_period=3.0
        )
        self.objects.append(obj)
        print(f"[Object] {obj.object_id} spawned at ({x:.1f}, {y:.1f})")
        return obj

    def get_objects_in_radius(self, x: float, y: float, radius: float) -> List[SimulatedObject]:
        return [obj for obj in self.objects if obj.distance_to(x, y) <= radius]

    def get_object_by_id(self, object_id: str) -> Optional[SimulatedObject]:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def remove_object(self, object_id: str) -> bool:
        for i, obj in enumerate(self.objects):
            if obj.object_id == object_id:
                self.objects.pop(i)
                return True
        return False
