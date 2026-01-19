"""Core entity types for the distributed world model.

This module defines the data structures for all trackable entities in the swarm:
- Assets: Drones, sensors, relay nodes
- Tracks: Detected/tracked objects in the environment
- EnhancedZones: Zones with richer semantics than the base Zone class

All entities use UUID identification, version numbers for CRDT-style merge,
and timestamps for conflict resolution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from uuid import uuid4
import time


class AssetType(Enum):
    """Types of autonomous assets in the swarm."""
    DRONE = "drone"
    GROUND_SENSOR = "ground_sensor"
    RELAY_NODE = "relay_node"


class AssetStatus(Enum):
    """Operational status of an asset."""
    ACTIVE = "active"
    IDLE = "idle"
    FAILED = "failed"
    OFFLINE = "offline"


@dataclass
class Asset:
    """
    Any autonomous system in the swarm.

    Assets are the core entities that make up the swarm - drones, ground sensors,
    relay nodes, etc. Each asset tracks its position, status, and capabilities.
    """
    asset_id: str
    asset_type: AssetType
    x: float
    y: float
    altitude: float
    status: AssetStatus
    capabilities: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    version: int = 1
    owner_id: str = ""  # Node that owns/controls this asset

    def update_position(self, x: float, y: float, altitude: float) -> None:
        """Update position and increment version."""
        self.x = x
        self.y = y
        self.altitude = altitude
        self.last_seen = time.time()
        self.version += 1

    def update_status(self, status: AssetStatus) -> None:
        """Update status and increment version."""
        self.status = status
        self.last_seen = time.time()
        self.version += 1

    def to_dict(self) -> dict:
        """Serialize asset to dictionary for network transmission."""
        return {
            'asset_id': self.asset_id,
            'asset_type': self.asset_type.value,
            'x': self.x,
            'y': self.y,
            'altitude': self.altitude,
            'status': self.status.value,
            'capabilities': self.capabilities,
            'last_seen': self.last_seen,
            'version': self.version,
            'owner_id': self.owner_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Asset':
        """Deserialize asset from dictionary."""
        return cls(
            asset_id=data['asset_id'],
            asset_type=AssetType(data['asset_type']),
            x=data['x'],
            y=data['y'],
            altitude=data['altitude'],
            status=AssetStatus(data['status']),
            capabilities=data.get('capabilities', []),
            last_seen=data.get('last_seen', time.time()),
            version=data.get('version', 1),
            owner_id=data.get('owner_id', ''),
        )


class TrackClassification(Enum):
    """Classification of tracked objects."""
    UNKNOWN = "unknown"


class ThreatLevel(Enum):
    """Threat assessment - all objects require authorization."""
    UNKNOWN = "unknown"
    HOSTILE = "hostile"


class KillChainStage(Enum):
    """
    Kill chain stages for target engagement.

    DETECTED → TRACKING → IDENTIFIED → AWAITING_AUTH → AUTHORIZED → ENGAGED
    """
    DETECTED = "detected"           # Initial detection
    TRACKING = "tracking"           # Actively following
    IDENTIFIED = "identified"       # Classification + threat level assigned
    AWAITING_AUTH = "awaiting_auth" # Hostile confirmed, waiting for human authorization
    AUTHORIZED = "authorized"       # Human authorized engagement
    ENGAGED = "engaged"             # Engagement in progress
    NEUTRALIZED = "neutralized"     # Target neutralized


class TrackStatus(Enum):
    """Status of a track."""
    TENTATIVE = "tentative"  # Low confidence, few observations
    CONFIRMED = "confirmed"  # High confidence, multiple observations
    LOST = "lost"            # Not seen recently


@dataclass
class Track:
    """
    A detected/tracked object in the environment.

    Tracks represent objects detected by assets (drones, sensors) that need
    to be monitored. They have position, velocity, confidence scores, and
    observation history.

    Kill Chain: DETECTED → TRACKING → IDENTIFIED → AWAITING_AUTH → AUTHORIZED → ENGAGED
    """
    track_id: str
    classification: TrackClassification
    x: float
    y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    confidence: float = 0.5  # 0.0 to 1.0
    status: TrackStatus = TrackStatus.TENTATIVE
    observation_count: int = 1
    first_observed_by: str = ""  # Asset ID
    last_observed_by: str = ""
    last_observed: float = field(default_factory=time.time)
    version: int = 1

    # Kill chain fields
    threat_level: ThreatLevel = ThreatLevel.UNKNOWN
    kill_chain_stage: KillChainStage = KillChainStage.DETECTED
    authorization_requested: bool = False
    authorization_time: Optional[float] = None  # When auth was requested
    engaged_by: Optional[str] = None  # Asset ID that engaged

    def update_observation(self, x: float, y: float, observer_id: str,
                           confidence_delta: float = 0.1) -> None:
        """Update track with new observation."""
        # Simple velocity estimation
        dt = time.time() - self.last_observed
        if dt > 0 and dt < 5.0:  # Reasonable time delta
            self.velocity_x = (x - self.x) / dt
            self.velocity_y = (y - self.y) / dt

        self.x = x
        self.y = y
        self.last_observed = time.time()
        self.last_observed_by = observer_id
        self.observation_count += 1
        self.confidence = min(1.0, self.confidence + confidence_delta)
        self.version += 1

        # Promote to confirmed after enough observations
        if self.observation_count >= 3 and self.confidence >= 0.7:
            self.status = TrackStatus.CONFIRMED

    def predict_position(self, dt: float) -> tuple:
        """Predict position after dt seconds using velocity."""
        return (
            self.x + self.velocity_x * dt,
            self.y + self.velocity_y * dt
        )

    def advance_kill_chain(self, new_stage: KillChainStage) -> None:
        """Advance the kill chain stage."""
        self.kill_chain_stage = new_stage
        self.version += 1

    def request_authorization(self) -> None:
        """Request human authorization for engagement."""
        self.authorization_requested = True
        self.authorization_time = time.time()
        self.kill_chain_stage = KillChainStage.AWAITING_AUTH
        self.version += 1

    def authorize(self) -> None:
        """Human authorizes engagement."""
        self.kill_chain_stage = KillChainStage.AUTHORIZED
        self.version += 1

    def deny_authorization(self) -> None:
        """Human denies engagement - return to tracking."""
        self.authorization_requested = False
        self.authorization_time = None
        self.kill_chain_stage = KillChainStage.TRACKING
        self.threat_level = ThreatLevel.UNKNOWN  # Downgrade threat
        self.version += 1

    def engage(self, asset_id: str) -> None:
        """Mark track as engaged by an asset."""
        self.kill_chain_stage = KillChainStage.ENGAGED
        self.engaged_by = asset_id
        self.version += 1

    def to_dict(self) -> dict:
        """Serialize track to dictionary for network transmission."""
        return {
            'track_id': self.track_id,
            'classification': self.classification.value,
            'x': self.x,
            'y': self.y,
            'velocity_x': self.velocity_x,
            'velocity_y': self.velocity_y,
            'confidence': self.confidence,
            'status': self.status.value,
            'observation_count': self.observation_count,
            'first_observed_by': self.first_observed_by,
            'last_observed_by': self.last_observed_by,
            'last_observed': self.last_observed,
            'version': self.version,
            'threat_level': self.threat_level.value,
            'kill_chain_stage': self.kill_chain_stage.value,
            'authorization_requested': self.authorization_requested,
            'authorization_time': self.authorization_time,
            'engaged_by': self.engaged_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Track':
        """Deserialize track from dictionary."""
        return cls(
            track_id=data['track_id'],
            classification=TrackClassification(data['classification']),
            x=data['x'],
            y=data['y'],
            velocity_x=data.get('velocity_x', 0.0),
            velocity_y=data.get('velocity_y', 0.0),
            confidence=data.get('confidence', 0.5),
            status=TrackStatus(data['status']),
            observation_count=data.get('observation_count', 1),
            first_observed_by=data.get('first_observed_by', ''),
            last_observed_by=data.get('last_observed_by', ''),
            last_observed=data.get('last_observed', time.time()),
            version=data.get('version', 1),
            threat_level=ThreatLevel(data.get('threat_level', 'unknown')),
            kill_chain_stage=KillChainStage(data.get('kill_chain_stage', 'detected')),
            authorization_requested=data.get('authorization_requested', False),
            authorization_time=data.get('authorization_time'),
            engaged_by=data.get('engaged_by'),
        )


class ZonePriority(Enum):
    """Priority level for zones."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EnhancedZoneStatus(Enum):
    """Status of an enhanced zone."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETE = "complete"


@dataclass
class EnhancedZone:
    """
    Zone with richer semantics than the existing Zone class.

    EnhancedZone supports multiple assignees, priority levels, and
    distributed state synchronization via version numbers.
    """
    zone_id: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    name: str = ""
    priority: ZonePriority = ZonePriority.NORMAL
    status: EnhancedZoneStatus = EnhancedZoneStatus.PENDING
    assigned_asset_ids: List[str] = field(default_factory=list)
    coverage_progress: float = 0.0  # 0.0 to 1.0
    version: int = 1
    last_updated: float = field(default_factory=time.time)

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

    def assign_asset(self, asset_id: str) -> None:
        """Assign an asset to this zone."""
        if asset_id not in self.assigned_asset_ids:
            self.assigned_asset_ids.append(asset_id)
            self.version += 1
            self.last_updated = time.time()
            if self.status == EnhancedZoneStatus.PENDING:
                self.status = EnhancedZoneStatus.ASSIGNED

    def unassign_asset(self, asset_id: str) -> None:
        """Remove an asset from this zone."""
        if asset_id in self.assigned_asset_ids:
            self.assigned_asset_ids.remove(asset_id)
            self.version += 1
            self.last_updated = time.time()
            if not self.assigned_asset_ids and self.status == EnhancedZoneStatus.ASSIGNED:
                self.status = EnhancedZoneStatus.PENDING

    def update_progress(self, progress: float) -> None:
        """Update coverage progress."""
        self.coverage_progress = max(0.0, min(1.0, progress))
        self.version += 1
        self.last_updated = time.time()
        if self.coverage_progress >= 1.0:
            self.status = EnhancedZoneStatus.COMPLETE

    def to_dict(self) -> dict:
        """Serialize zone to dictionary for network transmission."""
        return {
            'zone_id': self.zone_id,
            'x_min': self.x_min,
            'x_max': self.x_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'name': self.name,
            'priority': self.priority.value,
            'status': self.status.value,
            'assigned_asset_ids': self.assigned_asset_ids,
            'coverage_progress': self.coverage_progress,
            'version': self.version,
            'last_updated': self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EnhancedZone':
        """Deserialize zone from dictionary."""
        return cls(
            zone_id=data['zone_id'],
            x_min=data['x_min'],
            x_max=data['x_max'],
            y_min=data['y_min'],
            y_max=data['y_max'],
            name=data.get('name', ''),
            priority=ZonePriority(data.get('priority', 'normal')),
            status=EnhancedZoneStatus(data.get('status', 'pending')),
            assigned_asset_ids=data.get('assigned_asset_ids', []),
            coverage_progress=data.get('coverage_progress', 0.0),
            version=data.get('version', 1),
            last_updated=data.get('last_updated', time.time()),
        )


def should_accept_update(existing_version: int, existing_timestamp: float,
                         incoming_version: int, incoming_timestamp: float) -> bool:
    """
    Determine if an incoming update should replace existing data.

    Uses last-write-wins with version numbers:
    - Higher version always wins
    - Same version: later timestamp wins

    Args:
        existing_version: Version of existing entity
        existing_timestamp: Timestamp of existing entity
        incoming_version: Version of incoming entity
        incoming_timestamp: Timestamp of incoming entity

    Returns:
        True if incoming update should be accepted
    """
    if incoming_version > existing_version:
        return True
    if incoming_version == existing_version:
        return incoming_timestamp > existing_timestamp
    return False


def create_asset_id() -> str:
    """Generate a unique asset ID."""
    return str(uuid4())


def create_track_id() -> str:
    """Generate a unique track ID."""
    return f"track_{uuid4().hex[:8]}"


def create_zone_id() -> str:
    """Generate a unique zone ID."""
    return f"zone_{uuid4().hex[:8]}"
