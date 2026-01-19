"""Distributed world model maintained by each drone.

This module provides the WorldState class which tracks all entities (assets,
tracks, zones) and provides efficient spatial queries via a grid-based index.

Each drone maintains its own WorldState and synchronizes updates with peers
via the mesh network using CRDT-style conflict resolution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import time
import math

import config
from entities import (
    Asset, AssetType, AssetStatus,
    Track, TrackStatus, TrackClassification,
    EnhancedZone, EnhancedZoneStatus,
    should_accept_update, create_track_id
)


@dataclass
class SpatialIndex:
    """
    Grid-based spatial index for efficient range queries.

    Divides the world into a grid of cells. Each cell tracks which entities
    are within it. This allows O(1) insertion/removal and efficient range
    queries that only check relevant cells.
    """
    cell_size: float = 10.0
    _cells: Dict[Tuple[int, int], Set[str]] = field(default_factory=dict)
    _entity_cells: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Get the cell coordinates for a position."""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, entity_id: str, x: float, y: float) -> None:
        """Insert an entity at a position."""
        # Remove from old cell if exists
        if entity_id in self._entity_cells:
            old_cell = self._entity_cells[entity_id]
            if old_cell in self._cells:
                self._cells[old_cell].discard(entity_id)

        # Add to new cell
        cell = self._get_cell(x, y)
        if cell not in self._cells:
            self._cells[cell] = set()
        self._cells[cell].add(entity_id)
        self._entity_cells[entity_id] = cell

    def remove(self, entity_id: str) -> None:
        """Remove an entity from the index."""
        if entity_id in self._entity_cells:
            cell = self._entity_cells[entity_id]
            if cell in self._cells:
                self._cells[cell].discard(entity_id)
            del self._entity_cells[entity_id]

    def update(self, entity_id: str, x: float, y: float) -> None:
        """Update an entity's position (alias for insert)."""
        self.insert(entity_id, x, y)

    def query_radius(self, x: float, y: float, radius: float) -> Set[str]:
        """
        Query for all entities within a radius of a point.

        Returns set of entity IDs. Caller should filter by actual distance
        if exact radius is needed (this returns all entities in cells that
        could contain matches).
        """
        result = set()

        # Calculate which cells could contain matches
        min_cell_x = int((x - radius) // self.cell_size)
        max_cell_x = int((x + radius) // self.cell_size)
        min_cell_y = int((y - radius) // self.cell_size)
        max_cell_y = int((y + radius) // self.cell_size)

        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                cell = (cx, cy)
                if cell in self._cells:
                    result.update(self._cells[cell])

        return result

    def query_rect(self, x_min: float, y_min: float,
                   x_max: float, y_max: float) -> Set[str]:
        """Query for all entities within a rectangle."""
        result = set()

        min_cell_x = int(x_min // self.cell_size)
        max_cell_x = int(x_max // self.cell_size)
        min_cell_y = int(y_min // self.cell_size)
        max_cell_y = int(y_max // self.cell_size)

        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                cell = (cx, cy)
                if cell in self._cells:
                    result.update(self._cells[cell])

        return result

    def clear(self) -> None:
        """Clear all entries from the index."""
        self._cells.clear()
        self._entity_cells.clear()


@dataclass
class WorldState:
    """
    Complete world model maintained by each drone.

    Tracks all known assets, tracks, and zones. Provides efficient queries
    via spatial indexing. Supports CRDT-style updates for distributed sync.
    """
    node_id: str

    # Entity registries
    assets: Dict[str, Asset] = field(default_factory=dict)
    tracks: Dict[str, Track] = field(default_factory=dict)
    zones: Dict[str, EnhancedZone] = field(default_factory=dict)

    # Spatial indices (initialized in __post_init__)
    _asset_index: SpatialIndex = field(default_factory=lambda: SpatialIndex(
        cell_size=config.SPATIAL_INDEX_CELL_SIZE
    ))
    _track_index: SpatialIndex = field(default_factory=lambda: SpatialIndex(
        cell_size=config.SPATIAL_INDEX_CELL_SIZE
    ))

    # Reference to existing coverage map
    coverage_map: Optional[Any] = None  # CoverageMap, avoid circular import

    # Change tracking for sync
    _pending_updates: List[Tuple[str, str, Any]] = field(default_factory=list)
    _last_sync_time: float = field(default_factory=time.time)

    # ========== Asset operations ==========

    def upsert_asset(self, asset: Asset, from_sync: bool = False) -> bool:
        """
        Insert or update an asset using CRDT-style merge.

        Args:
            asset: The asset to upsert
            from_sync: True if this update came from network sync

        Returns:
            True if the asset was inserted/updated, False if rejected
        """
        existing = self.assets.get(asset.asset_id)

        if existing is not None:
            # Check if we should accept this update
            if not should_accept_update(
                existing.version, existing.last_seen,
                asset.version, asset.last_seen
            ):
                return False

        # Accept the update
        self.assets[asset.asset_id] = asset
        self._asset_index.update(asset.asset_id, asset.x, asset.y)

        # Track for sync if this is a local update
        if not from_sync:
            self._pending_updates.append(('asset', asset.asset_id, asset.to_dict()))

        return True

    def remove_asset(self, asset_id: str) -> bool:
        """Remove an asset from the world state."""
        if asset_id in self.assets:
            self._asset_index.remove(asset_id)
            del self.assets[asset_id]
            return True
        return False

    def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Get an asset by ID."""
        return self.assets.get(asset_id)

    def get_assets_in_radius(self, x: float, y: float, radius: float) -> List[Asset]:
        """Get all assets within a radius of a point."""
        candidate_ids = self._asset_index.query_radius(x, y, radius)
        result = []

        for asset_id in candidate_ids:
            asset = self.assets.get(asset_id)
            if asset:
                # Check actual distance
                dx = asset.x - x
                dy = asset.y - y
                if math.sqrt(dx*dx + dy*dy) <= radius:
                    result.append(asset)

        return result

    def get_active_assets(self, asset_type: Optional[AssetType] = None) -> List[Asset]:
        """Get all active assets, optionally filtered by type."""
        result = []
        for asset in self.assets.values():
            if asset.status in (AssetStatus.ACTIVE, AssetStatus.IDLE):
                if asset_type is None or asset.asset_type == asset_type:
                    result.append(asset)
        return result

    def get_assets_by_type(self, asset_type: AssetType) -> List[Asset]:
        """Get all assets of a specific type."""
        return [a for a in self.assets.values() if a.asset_type == asset_type]

    # ========== Track operations ==========

    def upsert_track(self, track: Track, from_sync: bool = False) -> bool:
        """
        Insert or update a track using CRDT-style merge.

        Args:
            track: The track to upsert
            from_sync: True if this update came from network sync

        Returns:
            True if the track was inserted/updated, False if rejected
        """
        existing = self.tracks.get(track.track_id)

        if existing is not None:
            if not should_accept_update(
                existing.version, existing.last_observed,
                track.version, track.last_observed
            ):
                return False

        self.tracks[track.track_id] = track
        self._track_index.update(track.track_id, track.x, track.y)

        if not from_sync:
            self._pending_updates.append(('track', track.track_id, track.to_dict()))

        return True

    def remove_track(self, track_id: str) -> bool:
        """Remove a track from the world state."""
        if track_id in self.tracks:
            self._track_index.remove(track_id)
            del self.tracks[track_id]
            return True
        return False

    def get_track(self, track_id: str) -> Optional[Track]:
        """Get a track by ID."""
        return self.tracks.get(track_id)

    def get_tracks_in_radius(self, x: float, y: float, radius: float) -> List[Track]:
        """Get all tracks within a radius of a point."""
        candidate_ids = self._track_index.query_radius(x, y, radius)
        result = []

        for track_id in candidate_ids:
            track = self.tracks.get(track_id)
            if track:
                dx = track.x - x
                dy = track.y - y
                if math.sqrt(dx*dx + dy*dy) <= radius:
                    result.append(track)

        return result

    def get_active_tracks(self) -> List[Track]:
        """Get all tracks that are not lost."""
        return [t for t in self.tracks.values() if t.status != TrackStatus.LOST]

    def get_confirmed_tracks(self) -> List[Track]:
        """Get all confirmed tracks."""
        return [t for t in self.tracks.values() if t.status == TrackStatus.CONFIRMED]

    def create_track(self, x: float, y: float, observer_id: str,
                     classification: TrackClassification = TrackClassification.UNKNOWN,
                     confidence: float = 0.5,
                     track_id: Optional[str] = None) -> Track:
        """Create a new track at the given position.

        Args:
            x: X position
            y: Y position
            observer_id: ID of the asset that observed this
            classification: Track classification
            confidence: Initial confidence (0.0-1.0)
            track_id: Optional track ID (defaults to auto-generated)
        """
        track = Track(
            track_id=track_id if track_id else create_track_id(),
            classification=classification,
            x=x,
            y=y,
            confidence=confidence,
            first_observed_by=observer_id,
            last_observed_by=observer_id,
        )
        self.upsert_track(track)
        return track

    # ========== Zone operations ==========

    def upsert_zone(self, zone: EnhancedZone, from_sync: bool = False) -> bool:
        """
        Insert or update a zone using CRDT-style merge.

        Args:
            zone: The zone to upsert
            from_sync: True if this update came from network sync

        Returns:
            True if the zone was inserted/updated, False if rejected
        """
        existing = self.zones.get(zone.zone_id)

        if existing is not None:
            if not should_accept_update(
                existing.version, existing.last_updated,
                zone.version, zone.last_updated
            ):
                return False

        self.zones[zone.zone_id] = zone

        if not from_sync:
            self._pending_updates.append(('zone', zone.zone_id, zone.to_dict()))

        return True

    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone from the world state."""
        if zone_id in self.zones:
            del self.zones[zone_id]
            return True
        return False

    def get_zone(self, zone_id: str) -> Optional[EnhancedZone]:
        """Get a zone by ID."""
        return self.zones.get(zone_id)

    def get_zones_at_point(self, x: float, y: float) -> List[EnhancedZone]:
        """Get all zones that contain a point."""
        return [z for z in self.zones.values() if z.contains_point(x, y)]

    def get_active_zones(self) -> List[EnhancedZone]:
        """Get all zones that are not complete."""
        return [
            z for z in self.zones.values()
            if z.status != EnhancedZoneStatus.COMPLETE
        ]

    # ========== Maintenance ==========

    def cleanup_stale_entities(self) -> Dict[str, int]:
        """
        Mark stale assets as offline and stale tracks as lost.
        Remove very old tracks entirely.

        Returns:
            Dict with counts of entities affected: {'assets_offline': N, 'tracks_lost': N, 'tracks_removed': N}
        """
        current_time = time.time()
        result = {'assets_offline': 0, 'tracks_lost': 0, 'tracks_removed': 0}

        # Mark stale assets as offline
        for asset in self.assets.values():
            if asset.owner_id == self.node_id:
                # Don't mark our own assets as offline
                continue
            if asset.status in (AssetStatus.ACTIVE, AssetStatus.IDLE):
                if current_time - asset.last_seen > config.ASSET_STALE_TIMEOUT:
                    asset.status = AssetStatus.OFFLINE
                    asset.version += 1
                    result['assets_offline'] += 1

        # Mark stale tracks as lost, remove very old ones
        tracks_to_remove = []
        for track in self.tracks.values():
            age = current_time - track.last_observed
            if track.status != TrackStatus.LOST and age > config.TRACK_STALE_TIMEOUT:
                track.status = TrackStatus.LOST
                track.version += 1
                result['tracks_lost'] += 1
            elif age > config.TRACK_EXPIRE_TIMEOUT:
                tracks_to_remove.append(track.track_id)

        for track_id in tracks_to_remove:
            self.remove_track(track_id)
            result['tracks_removed'] += 1

        return result

    def get_pending_updates(self) -> List[Tuple[str, str, Any]]:
        """
        Get pending updates for sync and clear the pending list.

        Returns:
            List of (entity_type, entity_id, entity_data) tuples
        """
        updates = self._pending_updates
        self._pending_updates = []
        return updates

    def has_pending_updates(self) -> bool:
        """Check if there are pending updates to sync."""
        return len(self._pending_updates) > 0

    def clear_pending_updates(self) -> None:
        """Clear pending updates without returning them."""
        self._pending_updates = []

    # ========== Statistics ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the world state."""
        return {
            'num_assets': len(self.assets),
            'num_active_assets': len(self.get_active_assets()),
            'num_tracks': len(self.tracks),
            'num_active_tracks': len(self.get_active_tracks()),
            'num_confirmed_tracks': len(self.get_confirmed_tracks()),
            'num_zones': len(self.zones),
            'num_active_zones': len(self.get_active_zones()),
            'pending_updates': len(self._pending_updates),
        }
