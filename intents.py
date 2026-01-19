"""Intent-based tasking system for autonomous drone decision making.

This module implements a decentralized intent system where drones:
1. Observe the shared world state
2. Evaluate available tasks
3. Declare their intent to perform a task
4. Execute while broadcasting intent to prevent conflicts

Intents automatically expire if not renewed, ensuring robustness to failures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from uuid import uuid4
import time

import config
from entities import (
    Asset, AssetStatus, AssetType,
    Track, TrackStatus, TrackClassification,
    EnhancedZone, EnhancedZoneStatus, ZonePriority
)

if TYPE_CHECKING:
    from world_state import WorldState


class IntentType(Enum):
    """Types of intents a drone can declare."""
    # Patrol states
    PATROL = "patrol"                   # Patrolling assigned zone
    TAKEOVER_ZONE = "takeover_zone"     # Taking over a failed drone's zone (patrol)
    IDLE = "idle"                       # Available for tasking

    # Kill chain states (mirrors KillChainStage progression)
    TRACK_OBJECT = "track_object"       # Following a detected target (DETECTED/TRACKING)
    IDENTIFY = "identify"               # Gathering observations to classify (TRACKING→IDENTIFIED)
    AWAIT_AUTH = "await_auth"           # Holding position, waiting for authorization
    ENGAGE = "engage"                   # Actively neutralizing authorized target


class TaskType(Enum):
    """Types of tasks that can be identified."""
    PATROL_ZONE = "patrol_zone"         # Patrol own assigned zone
    TRACK_OBJECT = "track_object"       # Track a detected object
    ORPHANED_ZONE = "orphaned_zone"     # Take over zone from failed drone
    ENGAGE_TARGET = "engage_target"     # Engage an authorized target


@dataclass
class Intent:
    """
    A declared intention by a drone to perform a task.

    Intents are broadcast to other drones to coordinate work without
    a central coordinator. They auto-expire if not renewed.
    """
    intent_id: str
    asset_id: str               # Drone declaring the intent
    intent_type: IntentType
    target_id: Optional[str]    # Zone ID, Track ID, or Asset ID
    target_position: Optional[Tuple[float, float]] = None  # For navigation
    priority: int = 5           # 1-10, higher = more important
    started_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + config.INTENT_EXPIRY_TIME)
    version: int = 1

    def is_expired(self) -> bool:
        """Check if intent has expired."""
        return time.time() > self.expires_at

    def renew(self, duration: float = None) -> None:
        """Extend the expiry time."""
        if duration is None:
            duration = config.INTENT_EXPIRY_TIME
        self.expires_at = time.time() + duration
        self.version += 1

    def to_dict(self) -> dict:
        """Serialize intent for network transmission."""
        return {
            'intent_id': self.intent_id,
            'asset_id': self.asset_id,
            'intent_type': self.intent_type.value,
            'target_id': self.target_id,
            'target_position': self.target_position,
            'priority': self.priority,
            'started_at': self.started_at,
            'expires_at': self.expires_at,
            'version': self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Intent':
        """Deserialize intent from dictionary."""
        return cls(
            intent_id=data['intent_id'],
            asset_id=data['asset_id'],
            intent_type=IntentType(data['intent_type']),
            target_id=data.get('target_id'),
            target_position=tuple(data['target_position']) if data.get('target_position') else None,
            priority=data.get('priority', 5),
            started_at=data.get('started_at', time.time()),
            expires_at=data.get('expires_at', time.time() + config.INTENT_EXPIRY_TIME),
            version=data.get('version', 1),
        )


@dataclass
class Task:
    """A potential task identified from world state analysis."""
    task_type: TaskType
    target_id: str
    target_position: Tuple[float, float]
    priority: int               # 1-10
    description: str

    def distance_from(self, x: float, y: float) -> float:
        """Calculate distance from a position to this task."""
        dx = self.target_position[0] - x
        dy = self.target_position[1] - y
        return (dx*dx + dy*dy) ** 0.5


@dataclass
class IntentManager:
    """
    Manages intent-based decision making for a drone.

    Observes world state, identifies tasks, declares intents,
    and coordinates with other drones via shared intent state.
    """
    my_asset_id: str
    world_state: 'WorldState'

    # Current intent
    current_intent: Optional[Intent] = None

    # Track all known intents (including from other drones)
    all_intents: Dict[str, Intent] = field(default_factory=dict)

    # Pending intent to broadcast
    _pending_intent: Optional[Intent] = None
    _intent_changed: bool = False

    # Track when this drone became idle (for longest-idle selection)
    _idle_since: Optional[float] = None

    # ========== Intent Management ==========

    def set_intent(self, intent_type: IntentType, target_id: Optional[str] = None,
                   target_position: Optional[Tuple[float, float]] = None,
                   priority: int = 5) -> Intent:
        """
        Declare a new intent.

        Args:
            intent_type: Type of intent
            target_id: ID of target (zone, track, or asset)
            target_position: Position to navigate to
            priority: Priority level (1-10)

        Returns:
            The new Intent object
        """
        intent = Intent(
            intent_id=f"intent_{uuid4().hex[:8]}",
            asset_id=self.my_asset_id,
            intent_type=intent_type,
            target_id=target_id,
            target_position=target_position,
            priority=priority,
        )

        # Track when drone becomes idle
        if intent_type == IntentType.IDLE:
            if self._idle_since is None:
                self._idle_since = time.time()
        else:
            self._idle_since = None  # No longer idle

        self.current_intent = intent
        self.all_intents[self.my_asset_id] = intent
        self._pending_intent = intent
        self._intent_changed = True

        return intent

    def renew_current_intent(self) -> None:
        """Renew the current intent to prevent expiry."""
        if self.current_intent:
            self.current_intent.renew()
            self.all_intents[self.my_asset_id] = self.current_intent
            self._pending_intent = self.current_intent
            self._intent_changed = True

    def clear_intent(self) -> None:
        """Clear current intent (set to IDLE)."""
        self.set_intent(IntentType.IDLE)

    def apply_intent(self, intent: Intent, from_sync: bool = True) -> bool:
        """
        Apply an intent update (usually from another drone).

        Args:
            intent: The intent to apply
            from_sync: True if from network sync

        Returns:
            True if applied, False if rejected
        """
        existing = self.all_intents.get(intent.asset_id)

        if existing:
            # Only accept newer versions
            if intent.version <= existing.version:
                if intent.version == existing.version:
                    if intent.expires_at <= existing.expires_at:
                        return False
                else:
                    return False

        self.all_intents[intent.asset_id] = intent
        return True

    def cleanup_expired_intents(self) -> int:
        """Remove expired intents. Returns count removed."""
        expired = [
            asset_id for asset_id, intent in self.all_intents.items()
            if intent.is_expired() and asset_id != self.my_asset_id
        ]
        for asset_id in expired:
            del self.all_intents[asset_id]
        return len(expired)

    def get_pending_intent(self) -> Optional[Intent]:
        """Get pending intent for broadcast and clear flag."""
        if self._intent_changed:
            self._intent_changed = False
            return self._pending_intent
        return None

    def has_pending_intent(self) -> bool:
        """Check if there's a pending intent to broadcast."""
        return self._intent_changed

    # ========== Task Discovery ==========

    def get_available_tasks(self, my_position: Tuple[float, float],
                            my_zones: List = None,
                            detected_objects: List = None) -> List[Task]:
        """
        Identify tasks that need doing (patrol or track objects).

        Args:
            my_position: Current drone position (x, y)
            my_zones: Zones assigned to this drone
            detected_objects: List of detected objects in range

        Returns:
            List of available tasks, sorted by priority
        """
        tasks = []

        # 1. Check for objects to track (highest priority)
        if detected_objects:
            for obj in detected_objects:
                # Only track untracked objects in our zone
                if not obj.detected:
                    tasks.append(Task(
                        task_type=TaskType.TRACK_OBJECT,
                        target_id=obj.object_id,
                        target_position=(obj.x, obj.y),
                        priority=9,  # High priority for new detections
                        description=f"Track {obj.object_id}"
                    ))

        # 2. Check for orphaned zones (drone failed) - only if IDLE
        tasks.extend(self._find_orphaned_zones())

        # 3. Add own zone patrol if assigned
        if my_zones:
            for zone in my_zones:
                tasks.append(Task(
                    task_type=TaskType.PATROL_ZONE,
                    target_id=f"zone_{zone.id}",
                    target_position=zone.center,
                    priority=5,
                    description=f"Patrol zone {zone.id}"
                ))

        # Filter out tasks already claimed by others
        tasks = self._filter_claimed_tasks(tasks)

        # Sort by priority (descending) then distance (ascending)
        tasks.sort(key=lambda t: (-t.priority, t.distance_from(*my_position)))

        return tasks

    def _find_orphaned_zones(self) -> List[Task]:
        """
        Find zones whose assigned drones have failed/gone offline.

        Simple rules:
        1. Only IDLE drones can be reassigned
        2. Lowest-ID active drone takes orphaned zones (deterministic)
        3. Exactly 1 drone per orphaned zone
        """
        tasks = []

        # Only consider orphaned zones if I am IDLE
        am_i_idle = (self.current_intent is None or
                     self.current_intent.intent_type == IntentType.IDLE)
        if not am_i_idle:
            return tasks

        # Check enhanced zones in world state for orphaned zones
        for zone in self.world_state.zones.values():
            # Check if this zone has a failed/offline drone
            has_failed_drone = False
            for asset_id in zone.assigned_asset_ids:
                asset = self.world_state.get_asset(asset_id)
                if asset and asset.status in (AssetStatus.FAILED, AssetStatus.OFFLINE):
                    has_failed_drone = True
                    break

            if not has_failed_drone:
                continue

            # Check if any ACTIVE drone is already patrolling this zone
            if self._is_zone_being_patrolled(zone):
                continue

            # Only lowest-ID active drone can take orphaned zones (deterministic)
            if self._am_i_lowest_active_drone():
                tasks.append(Task(
                    task_type=TaskType.ORPHANED_ZONE,
                    target_id=zone.zone_id,
                    target_position=zone.center,
                    priority=8,
                    description=f"Take over zone {zone.zone_id}"
                ))

        return tasks

    def _is_zone_being_patrolled(self, zone: 'EnhancedZone') -> bool:
        """Check if any active drone is currently patrolling this zone."""
        for intent in self.all_intents.values():
            if intent.asset_id == self.my_asset_id:
                continue
            if intent.is_expired():
                continue
            # Skip intents from failed drones
            asset = self.world_state.get_asset(intent.asset_id)
            if asset and asset.status in (AssetStatus.FAILED, AssetStatus.OFFLINE):
                continue
            # Check if this drone is patrolling the zone
            if intent.target_id == zone.zone_id:
                if intent.intent_type in (IntentType.PATROL, IntentType.TAKEOVER_ZONE):
                    return True
        return False

    def _am_i_lowest_active_drone(self) -> bool:
        """
        Check if this drone is the lowest-ID active drone.

        Simple deterministic rule: ONLY the lowest-ID active drone can take
        orphaned zones. This guarantees exactly 1 drone claims each zone
        without needing any synchronization.
        """
        for asset in self.world_state.assets.values():
            if asset.asset_type != AssetType.DRONE:
                continue
            if asset.status in (AssetStatus.FAILED, AssetStatus.OFFLINE):
                continue
            if asset.asset_id == self.my_asset_id:
                continue

            # If there's any active drone with LOWER ID, I don't take orphaned zones
            if asset.asset_id < self.my_asset_id:
                return False

        return True

    def _filter_claimed_tasks(self, tasks: List[Task]) -> List[Task]:
        """Remove tasks that other active drones have already claimed."""
        claimed_targets = set()

        for intent in self.all_intents.values():
            if intent.asset_id == self.my_asset_id:
                continue
            if intent.is_expired():
                continue
            # Don't count claims from failed/offline drones
            asset = self.world_state.get_asset(intent.asset_id)
            if asset and asset.status in (AssetStatus.FAILED, AssetStatus.OFFLINE):
                continue
            if intent.target_id:
                claimed_targets.add(intent.target_id)

        return [t for t in tasks if t.target_id not in claimed_targets]

    # ========== Decision Making ==========

    def select_best_task(self, my_position: Tuple[float, float],
                         my_zones: List = None,
                         detected_objects: List = None) -> Optional[Task]:
        """
        Select the best available task.

        Args:
            my_position: Current drone position
            my_zones: Zones assigned to this drone
            detected_objects: List of detected objects in range

        Returns:
            Best task to pursue, or None if no tasks available
        """
        tasks = self.get_available_tasks(my_position, my_zones, detected_objects)

        if not tasks:
            return None

        # Tasks are already sorted by priority and distance
        return tasks[0]

    def should_change_intent(self, my_position: Tuple[float, float],
                             my_zones: List = None,
                             detected_objects: List = None) -> Optional[Task]:
        """
        Check if we should change our current intent.

        Rules:
        - If IDLE or no intent, look for work (patrol or track)
        - If patrolling and detect object, switch to tracking
        - If tracking and object leaves zone, return to patrol

        Returns:
            New task if we should change, None if we should continue current intent
        """
        # If no current intent or it's IDLE, look for work
        if not self.current_intent or self.current_intent.intent_type == IntentType.IDLE:
            return self.select_best_task(my_position, my_zones, detected_objects)

        # If patrolling and we detect an object, switch to tracking
        if self.current_intent.intent_type in (IntentType.PATROL, IntentType.TAKEOVER_ZONE):
            if detected_objects:
                for obj in detected_objects:
                    if not obj.detected:
                        # New untracked object - switch to tracking!
                        return Task(
                            task_type=TaskType.TRACK_OBJECT,
                            target_id=obj.object_id,
                            target_position=(obj.x, obj.y),
                            priority=9,
                            description=f"Track {obj.object_id}"
                        )

        # If tracking and object left our zone (or is lost), return to patrol
        if self.current_intent.intent_type == IntentType.TRACK_OBJECT:
            # Check if tracked object is still in our zone
            # This will be handled in drone.py by checking zone bounds
            pass

        # Otherwise, continue current intent
        return None

    def intent_to_goal_position(self) -> Optional[Tuple[float, float]]:
        """Get the navigation target position from current intent."""
        if not self.current_intent:
            return None

        if self.current_intent.target_position:
            return self.current_intent.target_position

        # Try to resolve position from target_id
        if self.current_intent.target_id:
            if self.current_intent.intent_type == IntentType.TRACK_OBJECT:
                # For tracking, look up track position from world state
                track = self.world_state.get_track(self.current_intent.target_id)
                if track:
                    return (track.x, track.y)

            elif self.current_intent.intent_type in (IntentType.PATROL, IntentType.TAKEOVER_ZONE):
                # For patrolling, return zone center (patrol waypoints handled in drone.py)
                zone = self.world_state.get_zone(self.current_intent.target_id)
                if zone:
                    return zone.center

        return None

    # ========== Statistics ==========

    def get_stats(self) -> Dict:
        """Get statistics about intent state."""
        active_intents = [i for i in self.all_intents.values() if not i.is_expired()]
        by_type = {}
        for intent in active_intents:
            t = intent.intent_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            'current_intent': self.current_intent.intent_type.value if self.current_intent else None,
            'current_target': self.current_intent.target_id if self.current_intent else None,
            'total_intents': len(active_intents),
            'by_type': by_type,
        }


def create_intent_id() -> str:
    """Generate a unique intent ID."""
    return f"intent_{uuid4().hex[:8]}"
