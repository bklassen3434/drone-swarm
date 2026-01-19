"""Drone agent class for the drone swarm system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, TYPE_CHECKING
import time
import math
import random

import config
from zone import Zone, ZoneStatus
from goal_planner import GoalPlanner, MovementGoal
from mesh import MeshNode
from sync import StateSynchronizer, create_message_handlers
from entities import (
    Asset, AssetType, AssetStatus, Track, TrackClassification, TrackStatus,
    ThreatLevel, KillChainStage
)
from world_state import WorldState
from intents import IntentManager, IntentType, Intent, Task, TaskType
from objects import SimulatedObject

if TYPE_CHECKING:
    from coverage_map import CoverageMap
    from objects import ObjectManager


class DroneStatus(Enum):
    """Status of a drone."""
    IDLE = "idle"
    FLYING = "flying"
    PHOTOGRAPHING = "photographing"
    FAILED = "failed"


@dataclass
class Photo:
    """Represents a photo taken by a drone."""
    drone_id: int
    x: float
    y: float
    altitude: float
    timestamp: float


@dataclass
class Drone:
    """Represents an autonomous drone agent."""
    id: int
    x: float = 0.0
    y: float = 0.0
    altitude: float = 10.0
    status: DroneStatus = DroneStatus.IDLE
    assigned_zones: List[Zone] = field(default_factory=list)
    photos: List[Photo] = field(default_factory=list)
    speed: float = config.DRONE_SPEED

    # Goal-based movement state
    current_goal: Optional[MovementGoal] = None
    goal_planner: GoalPlanner = field(default_factory=lambda: GoalPlanner(strategy="nearest"))
    _time_since_last_photo: float = 0.0
    _time_since_goal_update: float = 0.0
    _current_zone_idx: int = 0

    # Mesh communication
    mesh_node: Optional[MeshNode] = field(default=None, repr=False)
    state_sync: Optional[StateSynchronizer] = field(default=None, repr=False)
    _mesh_enabled: bool = False

    # World state
    world_state: Optional[WorldState] = field(default=None, repr=False)
    _my_asset: Optional[Asset] = field(default=None, repr=False)
    _time_since_entity_sync: float = 0.0

    # Intent-based tasking
    intent_manager: Optional[IntentManager] = field(default=None, repr=False)
    _time_since_intent_update: float = 0.0

    # Patrol and tracking state
    _patrol_waypoints: List[Tuple[float, float]] = field(default_factory=list)
    _current_waypoint_idx: int = 0
    _tracked_object_id: Optional[str] = None
    _detection_range: float = 4.0  # Detection radius in meters

    def __post_init__(self):
        """Initialize mesh components after dataclass creation."""
        self._init_mesh()
        self._init_world_state()

    def _init_mesh(self) -> None:
        """Initialize mesh node and state synchronizer."""
        node_id = f"drone_{self.id}"
        self.mesh_node = MeshNode(node_id=node_id, position=(self.x, self.y))
        self.state_sync = StateSynchronizer(node_id=node_id)

        # Register message handlers
        handlers = create_message_handlers(self.state_sync)
        for msg_type, handler in handlers.items():
            self.mesh_node.on_message(msg_type, handler)

    def _init_world_state(self) -> None:
        """Create world state, intent manager, and register self as an asset."""
        node_id = f"drone_{self.id}"
        self.world_state = WorldState(node_id=node_id)

        # Create asset representing this drone
        self._my_asset = Asset(
            asset_id=node_id,
            asset_type=AssetType.DRONE,
            x=self.x,
            y=self.y,
            altitude=self.altitude,
            status=AssetStatus.ACTIVE,
            capabilities=["patrol", "tracking"],
            owner_id=node_id,
        )
        self.world_state.upsert_asset(self._my_asset)

        # Create intent manager
        self.intent_manager = IntentManager(
            my_asset_id=node_id,
            world_state=self.world_state
        )

        # Connect world state and intent manager to synchronizer
        if self.state_sync:
            self.state_sync.set_world_state(self.world_state)
            self.state_sync.set_intent_manager(self.intent_manager)

    def enable_mesh(self, coverage_map: 'CoverageMap') -> None:
        """
        Enable mesh communication with the given coverage map.

        Args:
            coverage_map: The coverage map to synchronize
        """
        self._mesh_enabled = True
        if self.state_sync:
            self.state_sync.set_coverage_map(coverage_map)

    def assign_zone(self, zone: Zone) -> None:
        """Assign a zone to this drone."""
        zone.assigned_drone_id = self.id
        zone.status = ZoneStatus.IN_PROGRESS
        self.assigned_zones.append(zone)

        # Position at zone start if idle
        if self.status == DroneStatus.IDLE and len(self.assigned_zones) == 1:
            start_x, start_y = zone.get_start_position()
            self.x = start_x
            self.y = start_y
            self.status = DroneStatus.FLYING

    def get_current_zone(self) -> Optional[Zone]:
        """Get the zone currently being worked on."""
        if self._current_zone_idx < len(self.assigned_zones):
            return self.assigned_zones[self._current_zone_idx]
        return None

    def take_photo(self) -> Photo:
        """Take a photo at current position."""
        photo = Photo(
            drone_id=self.id,
            x=self.x,
            y=self.y,
            altitude=self.altitude,
            timestamp=time.time()
        )
        self.photos.append(photo)
        return photo

    def set_goal(self, goal: MovementGoal) -> None:
        """Set a new movement goal."""
        self.current_goal = goal

    def clear_goal(self) -> None:
        """Clear the current goal."""
        self.current_goal = None

    def has_goal(self) -> bool:
        """Check if drone has an active goal."""
        return self.current_goal is not None

    def is_goal_reached(self, threshold: float = None) -> bool:
        """Check if current goal has been reached."""
        if self.current_goal is None:
            return True
        if threshold is None:
            threshold = config.GOAL_REACHED_THRESHOLD
        return self.current_goal.is_reached(self.x, self.y, threshold)

    def move_toward_goal(self, dt: float) -> None:
        """Move the drone toward its current goal."""
        if self.current_goal is None:
            return

        # Calculate direction to goal
        dx = self.current_goal.x - self.x
        dy = self.current_goal.y - self.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 0.01:  # Essentially at goal
            return

        # Normalize direction
        dx /= distance
        dy /= distance

        # Calculate movement distance
        move_dist = min(self.speed * dt, distance)

        # Update position
        self.x += dx * move_dist
        self.y += dy * move_dist

    def update(self, dt: float, coverage_map: 'CoverageMap',
               object_manager: Optional['ObjectManager'] = None) -> Optional[Photo]:
        """
        Update drone position for one timestep using intent-based movement.

        Args:
            dt: Time delta in seconds
            coverage_map: The coverage map (kept for compatibility)
            object_manager: Manager for simulated objects to detect

        Returns:
            Photo if taken, None otherwise
        """
        if self.status == DroneStatus.FAILED:
            return None

        # Update mesh node position
        if self.mesh_node:
            self.mesh_node.update_position(self.x, self.y)

        # Process incoming mesh messages
        if self._mesh_enabled and self.mesh_node:
            self.mesh_node.process_messages()

        zone = self.get_current_zone()
        self.status = DroneStatus.FLYING
        photo_taken = None

        # Detect nearby objects (only those past grace period)
        detected_objects = []
        if object_manager:
            nearby = object_manager.get_objects_in_radius(
                self.x, self.y, self._detection_range
            )
            # Filter to only detectable objects
            detected_objects = [obj for obj in nearby if obj.can_be_detected()]

        # Intent-based decision making (patrol/track)
        if self.intent_manager:
            self._update_intent_based_patrol(dt, zone, detected_objects)

        # Move toward goal
        if self.has_goal():
            self.move_toward_goal(dt)

        # Take photos at intervals (for documentation, not coverage)
        self._time_since_last_photo += dt
        if self._time_since_last_photo >= config.PHOTO_INTERVAL:
            self._time_since_last_photo = 0.0
            photo_taken = self.take_photo()
            self.status = DroneStatus.PHOTOGRAPHING

        # Update world state with my position
        if self._my_asset and self.world_state:
            self._my_asset.update_position(self.x, self.y, self.altitude)
            self.world_state.upsert_asset(self._my_asset)

        # Broadcast entity and intent updates periodically
        self._time_since_entity_sync += dt
        if self._time_since_entity_sync >= config.WORLD_STATE_SYNC_INTERVAL:
            self._time_since_entity_sync = 0.0
            if self._mesh_enabled and self.state_sync and self.mesh_node:
                # Broadcast entity updates
                batch_msg = self.state_sync.create_batch_update_message()
                if batch_msg:
                    self.mesh_node.broadcast(batch_msg)

                # Broadcast intent updates
                intent_msg = self.state_sync.create_intent_message()
                if intent_msg:
                    self.mesh_node.broadcast(intent_msg)

            # Cleanup stale entities and intents
            if self.world_state:
                self.world_state.cleanup_stale_entities()
            if self.intent_manager:
                self.intent_manager.cleanup_expired_intents()

        return photo_taken

    def _update_intent_based_patrol(self, dt: float, zone: Optional[Zone],
                                     detected_objects: List[SimulatedObject]) -> None:
        """Intent-based decision making for patrol and tracking."""
        if self.intent_manager is None:
            return

        my_pos = (self.x, self.y)

        # Update intent periodically
        self._time_since_intent_update += dt
        needs_intent_update = (
            self._time_since_intent_update >= config.INTENT_UPDATE_INTERVAL or
            self.intent_manager.current_intent is None or
            self.intent_manager.current_intent.intent_type == IntentType.IDLE
        )

        if needs_intent_update:
            self._time_since_intent_update = 0.0

            # Check if we should change our intent (detect new objects, etc.)
            new_task = self.intent_manager.should_change_intent(
                my_pos, self.assigned_zones, detected_objects
            )

            if new_task:
                # We have a new task to pursue
                self._apply_task_as_intent(new_task, detected_objects)
            elif self.intent_manager.current_intent:
                # Renew current intent to prevent expiry
                self.intent_manager.renew_current_intent()

        # Update navigation goal based on current intent
        self._update_goal_from_patrol_intent(zone, detected_objects)

    def _apply_task_as_intent(self, task: Task,
                              detected_objects: List[SimulatedObject] = None) -> None:
        """Convert a task into an intent and declare it."""
        intent_type_map = {
            TaskType.PATROL_ZONE: IntentType.PATROL,
            TaskType.ORPHANED_ZONE: IntentType.TAKEOVER_ZONE,
            TaskType.TRACK_OBJECT: IntentType.TRACK_OBJECT,
            TaskType.ENGAGE_TARGET: IntentType.ENGAGE,
        }

        intent_type = intent_type_map.get(task.task_type, IntentType.PATROL)

        # If tracking a new object, mark it as detected by us
        if task.task_type == TaskType.TRACK_OBJECT and detected_objects:
            for obj in detected_objects:
                if obj.object_id == task.target_id:
                    obj.detected = True
                    obj.detected_by = f"drone_{self.id}"
                    self._tracked_object_id = obj.object_id
                    # Create/update a track in world state
                    self._create_track_for_object(obj)
                    break

        # If starting patrol, generate waypoints
        if intent_type in (IntentType.PATROL, IntentType.TAKEOVER_ZONE):
            zone = self.get_current_zone()
            if zone:
                self._generate_patrol_waypoints(zone)
            # Also generate for takeover zone
            if task.target_id and self.world_state:
                enhanced_zone = self.world_state.get_zone(task.target_id)
                if enhanced_zone:
                    self._generate_patrol_waypoints_from_bounds(
                        enhanced_zone.x_min, enhanced_zone.x_max,
                        enhanced_zone.y_min, enhanced_zone.y_max
                    )

        self.intent_manager.set_intent(
            intent_type=intent_type,
            target_id=task.target_id,
            target_position=task.target_position,
            priority=task.priority
        )

    def _update_goal_from_patrol_intent(self, zone: Optional[Zone],
                                         detected_objects: List[SimulatedObject]) -> None:
        """Update navigation goal based on current intent (patrol or track)."""
        if self.intent_manager is None or self.intent_manager.current_intent is None:
            return

        intent = self.intent_manager.current_intent

        if intent.intent_type == IntentType.TRACK_OBJECT:
            # Initial tracking mode - follow the object
            self._update_tracking_goal(intent, detected_objects, zone)

        elif intent.intent_type == IntentType.IDENTIFY:
            # Identifying - continue following while gathering observations
            self._update_tracking_goal(intent, detected_objects, zone)

        elif intent.intent_type == IntentType.AWAIT_AUTH:
            # Awaiting authorization - hold position near target
            self._update_tracking_goal(intent, detected_objects, zone)

        elif intent.intent_type == IntentType.ENGAGE:
            # Engaging - move to intercept target
            self._update_tracking_goal(intent, detected_objects, zone)

        elif intent.intent_type in (IntentType.PATROL, IntentType.TAKEOVER_ZONE):
            # Patrol mode - follow waypoints
            self._update_patrol_goal(intent, zone)

        elif intent.intent_type == IntentType.IDLE:
            self.clear_goal()

    def _update_tracking_goal(self, intent: Intent,
                              detected_objects: List[SimulatedObject],
                              zone: Optional[Zone]) -> None:
        """Update goal to track a detected object and advance kill chain."""
        # Find the tracked object in detection range
        tracked_obj = None
        for obj in detected_objects:
            if obj.object_id == intent.target_id:
                tracked_obj = obj
                break

        if tracked_obj:
            # Object is in detection range - update with fresh data
            intent.target_position = (tracked_obj.x, tracked_obj.y)
            self.intent_manager._intent_changed = True

            # Update the track in world state and advance kill chain
            if self.world_state:
                track = self._get_or_create_track_for_object(tracked_obj)
                if track:
                    track.x = tracked_obj.x
                    track.y = tracked_obj.y
                    track.update_observation(tracked_obj.x, tracked_obj.y, f"drone_{self.id}")

                    # Advance kill chain based on observations
                    self._advance_kill_chain(track, tracked_obj)

                    self.world_state.upsert_track(track)

            # Navigate toward the object
            self.set_goal(MovementGoal(x=tracked_obj.x, y=tracked_obj.y))

        elif self.world_state and intent.target_id:
            # Object not in detection range - use track's last known position
            track = self.world_state.get_track(intent.target_id)
            if track and track.status != TrackStatus.LOST:
                # Move toward last known position to reacquire
                predicted_x, predicted_y = track.predict_position(0.5)  # Predict 0.5s ahead
                self.set_goal(MovementGoal(x=predicted_x, y=predicted_y))
                intent.target_position = (predicted_x, predicted_y)
                self.intent_manager._intent_changed = True
            else:
                # Track is lost - return to patrol
                self._return_to_patrol()
        else:
            # No track info available - return to patrol
            self._return_to_patrol()

    def _get_or_create_track_for_object(self, obj: SimulatedObject) -> Optional[Track]:
        """Get existing track or create new one for object."""
        if not self.world_state:
            return None

        # Look for existing track by object_id (track_id = object_id)
        existing_track = self.world_state.get_track(obj.object_id)
        if existing_track:
            return existing_track

        # Create new track
        return self._create_track_for_object(obj)

    def _advance_kill_chain(self, track: Track, obj: SimulatedObject) -> None:
        """
        Advance the kill chain based on observations.

        Kill Chain: DETECTED → TRACKING → IDENTIFIED → AWAITING_AUTH → AUTHORIZED → ENGAGED
        Intent:     TRACK    → IDENTIFY → IDENTIFY  → AWAIT_AUTH    → ENGAGE     → (done)
        """
        # Stage 1: DETECTED → TRACKING (automatic after first observation)
        if track.kill_chain_stage == KillChainStage.DETECTED:
            track.advance_kill_chain(KillChainStage.TRACKING)
            print(f"[KillChain] {track.track_id}: DETECTED → TRACKING")
            # Intent stays TRACK_OBJECT

        # Stage 2: TRACKING → IDENTIFIED (after enough observations + classification)
        elif track.kill_chain_stage == KillChainStage.TRACKING:
            # Switch to IDENTIFY intent while gathering observations
            if self.intent_manager and self.intent_manager.current_intent:
                if self.intent_manager.current_intent.intent_type == IntentType.TRACK_OBJECT:
                    self.intent_manager.set_intent(
                        intent_type=IntentType.IDENTIFY,
                        target_id=track.track_id,
                        target_position=(track.x, track.y),
                        priority=8
                    )
                    print(f"[Intent] Drone {self.id}: TRACK_OBJECT → IDENTIFY")

            if track.observation_count >= 5 and track.confidence >= 0.7:
                # Classify the threat based on object type
                track.threat_level = self._assess_threat(obj)
                track.advance_kill_chain(KillChainStage.IDENTIFIED)
                print(f"[KillChain] {track.track_id}: TRACKING → IDENTIFIED (threat={track.threat_level.value})")

        # Stage 3: IDENTIFIED → AWAITING_AUTH (if hostile or suspect)
        elif track.kill_chain_stage == KillChainStage.IDENTIFIED:
            if track.threat_level == ThreatLevel.HOSTILE:
                if not track.authorization_requested:
                    track.request_authorization()
                    # Switch to AWAIT_AUTH intent
                    if self.intent_manager:
                        self.intent_manager.set_intent(
                            intent_type=IntentType.AWAIT_AUTH,
                            target_id=track.track_id,
                            target_position=(track.x, track.y),
                            priority=9
                        )
                    print(f"[KillChain] {track.track_id}: IDENTIFIED → AWAITING_AUTH (requesting human authorization)")
                    print(f"[Intent] Drone {self.id}: IDENTIFY → AWAIT_AUTH")

        # Stage 4: AUTHORIZED → Switch to ENGAGE intent
        elif track.kill_chain_stage == KillChainStage.AUTHORIZED:
            if self.intent_manager and self.intent_manager.current_intent:
                if self.intent_manager.current_intent.intent_type != IntentType.ENGAGE:
                    self.intent_manager.set_intent(
                        intent_type=IntentType.ENGAGE,
                        target_id=track.track_id,
                        target_position=(track.x, track.y),
                        priority=10  # Highest priority
                    )
                    print(f"[Intent] Drone {self.id}: → ENGAGE (authorization received)")

        # Stage 5: ENGAGED → NEUTRALIZED (handled by simulation)

    def _assess_threat(self, obj: SimulatedObject) -> ThreatLevel:
        """All detected objects require authorization."""
        return ThreatLevel.HOSTILE

    def _return_to_patrol(self) -> None:
        """Return to patrol mode after tracking ends."""
        self._tracked_object_id = None

        # Set to IDLE - simulation will detect this and rebalance
        if self.intent_manager:
            self.intent_manager.set_intent(
                intent_type=IntentType.IDLE,
                target_id=None,
                target_position=None
            )

    def _update_patrol_goal(self, intent: Intent, zone: Optional[Zone]) -> None:
        """Update goal to follow patrol waypoints."""
        # Get the zone we're patrolling
        work_zone = self._get_work_zone(zone, intent)

        if not work_zone:
            return

        # Generate waypoints if we don't have any
        if not self._patrol_waypoints:
            self._generate_patrol_waypoints(work_zone)

        # Check if we've reached current waypoint
        if self.is_goal_reached() or not self.has_goal():
            # Move to next waypoint
            self._current_waypoint_idx = (self._current_waypoint_idx + 1) % len(self._patrol_waypoints)
            next_wp = self._patrol_waypoints[self._current_waypoint_idx]
            self.set_goal(MovementGoal(x=next_wp[0], y=next_wp[1]))

            # Update intent position
            intent.target_position = next_wp
            self.intent_manager._intent_changed = True

    def _get_work_zone(self, zone: Optional[Zone], intent: Intent):
        """Get the zone we're currently working in."""
        work_zone = zone

        if intent.target_id and self.world_state:
            enhanced_zone = self.world_state.get_zone(intent.target_id)
            if enhanced_zone:
                from zone import Zone as BasicZone
                work_zone = BasicZone(
                    id=hash(enhanced_zone.zone_id) % 1000,
                    x_min=enhanced_zone.x_min,
                    x_max=enhanced_zone.x_max,
                    y_min=enhanced_zone.y_min,
                    y_max=enhanced_zone.y_max,
                )

        return work_zone

    def _generate_patrol_waypoints(self, zone: Zone) -> None:
        """Generate patrol waypoints for a zone (lawnmower pattern)."""
        self._generate_patrol_waypoints_from_bounds(
            zone.x_min, zone.x_max, zone.y_min, zone.y_max
        )

    def _generate_patrol_waypoints_from_bounds(self, x_min: float, x_max: float,
                                                y_min: float, y_max: float) -> None:
        """
        Generate figure-8 patrol pattern waypoints.

        Creates a figure-8 pattern that provides good coverage with
        smooth continuous motion. The pattern crosses through the center
        ensuring the middle of the zone is regularly monitored.
        """
        margin = 1.5  # Stay away from zone edges

        # Zone center and dimensions
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        half_width = (x_max - x_min) / 2 - margin
        half_height = (y_max - y_min) / 2 - margin

        # Generate figure-8 pattern using parametric points
        # Top loop + bottom loop, crossing at center
        num_points = 16  # Points per loop
        self._patrol_waypoints = []

        for i in range(num_points):
            # Angle from 0 to 2*pi
            t = (i / num_points) * 2 * math.pi

            # Figure-8 parametric equations (lemniscate-like)
            # x = sin(t), y = sin(t)*cos(t) scaled to zone
            x = cx + half_width * math.sin(t)
            y = cy + half_height * math.sin(t) * math.cos(t) * 2

            # Clamp to zone bounds
            x = max(x_min + margin, min(x_max - margin, x))
            y = max(y_min + margin, min(y_max - margin, y))

            self._patrol_waypoints.append((x, y))

        self._current_waypoint_idx = 0

    def _create_track_for_object(self, obj: SimulatedObject) -> Optional[Track]:
        """Create a track entry in world state for a detected object."""
        if not self.world_state:
            return None

        # Check if we already have a track for this object
        existing_track = self.world_state.get_track(obj.object_id)
        if existing_track:
            existing_track.update_observation(obj.x, obj.y, f"drone_{self.id}")
            self.world_state.upsert_track(existing_track)
            return existing_track

        # Create new track using object_id as track_id for correlation
        track = self.world_state.create_track(
            x=obj.x,
            y=obj.y,
            observer_id=f"drone_{self.id}",
            confidence=0.8,
            track_id=obj.object_id
        )
        return track

    def _handoff_tracking(self, obj: SimulatedObject, current_zone) -> None:
        """Handoff tracking when object leaves our zone."""
        # Mark the track as needing pickup by another drone
        if self.world_state:
            track = self.world_state.get_track(obj.object_id)
            if track:
                # Keep the track but we're no longer actively tracking
                track.status = TrackStatus.TENTATIVE  # Another drone should pick this up

        # Broadcast that we're releasing this track
        if self._mesh_enabled and self.mesh_node and self.state_sync:
            intent_msg = self.state_sync.create_intent_message()
            if intent_msg:
                self.mesh_node.broadcast(intent_msg)

        # Return to patrol
        self._tracked_object_id = None
        self.intent_manager.set_intent(
            intent_type=IntentType.PATROL,
            target_id=f"zone_{current_zone.id}" if hasattr(current_zone, 'id') else None,
            target_position=current_zone.center if hasattr(current_zone, 'center') else None
        )
        self._generate_patrol_waypoints(current_zone)

    def fail(self) -> None:
        """Mark drone as failed."""
        self.status = DroneStatus.FAILED
        self.clear_goal()
        # Update asset status in world state
        if self._my_asset:
            self._my_asset.update_status(AssetStatus.FAILED)

    def is_alive(self) -> bool:
        """Check if drone is operational."""
        return self.status != DroneStatus.FAILED

    def get_position(self) -> Tuple[float, float, float]:
        """Get current 3D position."""
        return (self.x, self.y, self.altitude)

    def get_position_2d(self) -> Tuple[float, float]:
        """Get current 2D position (x, y only)."""
        return (self.x, self.y)

    def get_goal_position(self) -> Optional[Tuple[float, float]]:
        """Get current goal position if any."""
        if self.current_goal is not None:
            return (self.current_goal.x, self.current_goal.y)
        return None

    def is_complete(self) -> bool:
        """Check if drone has completed all assigned zones."""
        return all(z.status == ZoneStatus.COMPLETE for z in self.assigned_zones)

    # ========== World State Query Methods ==========

    def get_nearby_assets(self, radius: float = 50.0) -> List[Asset]:
        """
        Get all assets within a radius of this drone.

        Args:
            radius: Search radius in meters

        Returns:
            List of Asset objects within range (excluding self)
        """
        if self.world_state is None:
            return []
        assets = self.world_state.get_assets_in_radius(self.x, self.y, radius)
        # Exclude self
        my_id = f"drone_{self.id}"
        return [a for a in assets if a.asset_id != my_id]

    def get_nearby_tracks(self, radius: float = 30.0) -> List[Track]:
        """
        Get all tracks within a radius of this drone.

        Args:
            radius: Search radius in meters

        Returns:
            List of Track objects within range
        """
        if self.world_state is None:
            return []
        return self.world_state.get_tracks_in_radius(self.x, self.y, radius)

    def report_track(self, x: float, y: float,
                     classification: TrackClassification = TrackClassification.UNKNOWN,
                     confidence: float = 0.5) -> Optional[Track]:
        """
        Report a detected object as a track.

        Creates a new track or updates an existing one if found nearby.

        Args:
            x: X position of detected object
            y: Y position of detected object
            classification: Classification of the object
            confidence: Confidence in the detection (0.0 to 1.0)

        Returns:
            The created or updated Track, or None if world state not available
        """
        if self.world_state is None:
            return None

        my_id = f"drone_{self.id}"

        # Check if there's an existing track nearby (within 5m)
        nearby_tracks = self.world_state.get_tracks_in_radius(x, y, 5.0)
        for track in nearby_tracks:
            if track.classification == classification or classification == TrackClassification.UNKNOWN:
                # Update existing track
                track.update_observation(x, y, my_id, confidence_delta=0.1)
                self.world_state.upsert_track(track)
                return track

        # Create new track
        track = self.world_state.create_track(
            x=x, y=y,
            observer_id=my_id,
            classification=classification,
            confidence=confidence
        )
        return track

    def get_world_state_stats(self) -> dict:
        """Get statistics about this drone's world state."""
        if self.world_state is None:
            return {}
        return self.world_state.get_stats()

    # ========== Intent Query Methods ==========

    def get_current_intent(self) -> Optional[Intent]:
        """Get this drone's current intent."""
        if self.intent_manager is None:
            return None
        return self.intent_manager.current_intent

    def get_intent_stats(self) -> dict:
        """Get statistics about intents."""
        if self.intent_manager is None:
            return {}
        return self.intent_manager.get_stats()

    def get_all_intents(self) -> List[Intent]:
        """Get all known intents (from all drones)."""
        if self.intent_manager is None:
            return []
        return list(self.intent_manager.all_intents.values())

    def get_available_tasks(self) -> List[Task]:
        """Get available tasks this drone could pursue."""
        if self.intent_manager is None:
            return []
        return self.intent_manager.get_available_tasks(
            my_position=(self.x, self.y),
            my_zones=self.assigned_zones
        )
