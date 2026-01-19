"""State synchronization for distributed coverage tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, TYPE_CHECKING
import time

from messages import (
    Message, MessageType,
    create_state_sync,
    create_entity_batch,
    create_intent_declare
)
from coverage_map import CoverageMap, CoverageCell
from entities import Asset, Track, EnhancedZone
from intents import Intent, IntentManager

if TYPE_CHECKING:
    from world_state import WorldState


@dataclass
class StateSynchronizer:
    """
    Keeps coverage map, world state, and intents in sync across mesh nodes.

    Each drone maintains its own coverage map, world state, and intent manager,
    synchronizing updates with peers via the mesh network.
    """
    node_id: str
    coverage_map: Optional[CoverageMap] = None
    world_state: Optional['WorldState'] = None
    intent_manager: Optional[IntentManager] = None
    seen_updates: Set[str] = field(default_factory=set)
    _pending_sync: List[tuple] = field(default_factory=list)
    _last_batch_time: float = field(default_factory=time.time)

    def set_coverage_map(self, coverage_map: CoverageMap) -> None:
        """Set the coverage map to synchronize."""
        self.coverage_map = coverage_map

    def record_coverage(self, cells: List[CoverageCell], drone_id: int) -> Message:
        """
        Record newly covered cells and create sync message.

        Args:
            cells: List of CoverageCell objects that were just covered
            drone_id: ID of the drone that covered them

        Returns:
            Message to broadcast to peers
        """
        covered_data = [
            (cell.grid_x, cell.grid_y, drone_id)
            for cell in cells
        ]
        return create_state_sync(self.node_id, covered_data)

    def apply_sync_message(self, message: Message) -> int:
        """
        Apply coverage updates from peer.

        Returns:
            Number of cells newly marked as covered
        """
        if message.message_id in self.seen_updates:
            return 0

        self.seen_updates.add(message.message_id)

        if self.coverage_map is None:
            return 0

        covered_cells = message.payload.get('covered_cells', [])
        newly_covered = 0

        for grid_x, grid_y, drone_id in covered_cells:
            cell = self.coverage_map.get_cell_by_grid(grid_x, grid_y)
            if cell and not cell.covered:
                cell.covered = True
                cell.covered_by = drone_id
                self.coverage_map._covered_count += 1
                newly_covered += 1

        return newly_covered

    def cleanup_old_updates(self) -> None:
        """Remove old update IDs to prevent memory growth."""
        if len(self.seen_updates) > 1000:
            self.seen_updates = set(list(self.seen_updates)[-500:])

    # ========== World State synchronization ==========

    def set_world_state(self, world_state: 'WorldState') -> None:
        """Set the world state to synchronize."""
        self.world_state = world_state

    def create_batch_update_message(self) -> Optional[Message]:
        """
        Create a batch update message from pending world state updates.

        Returns:
            Message if there are pending updates, None otherwise
        """
        if self.world_state is None:
            return None

        if not self.world_state.has_pending_updates():
            return None

        updates = self.world_state.get_pending_updates()
        if not updates:
            return None

        self._last_batch_time = time.time()
        return create_entity_batch(self.node_id, updates)

    def apply_entity_update(self, message: Message) -> bool:
        """
        Apply a single entity update from peer.

        Returns:
            True if the update was applied, False if rejected/ignored
        """
        if message.message_id in self.seen_updates:
            return False

        self.seen_updates.add(message.message_id)

        if self.world_state is None:
            return False

        entity_type = message.payload.get('entity_type')
        entity_data = message.payload.get('entity_data')

        if not entity_type or not entity_data:
            return False

        return self._apply_entity_data(entity_type, entity_data)

    def apply_entity_batch(self, message: Message) -> int:
        """
        Apply a batch of entity updates from peer.

        Returns:
            Number of updates successfully applied
        """
        if message.message_id in self.seen_updates:
            return 0

        self.seen_updates.add(message.message_id)

        if self.world_state is None:
            return 0

        updates = message.payload.get('updates', [])
        applied = 0

        for update in updates:
            entity_type = update.get('entity_type')
            entity_data = update.get('entity_data')

            if entity_type and entity_data:
                if self._apply_entity_data(entity_type, entity_data):
                    applied += 1

        return applied

    def apply_track_alert(self, message: Message) -> bool:
        """
        Apply a high-priority track alert from peer.

        Returns:
            True if the track was applied, False if rejected/ignored
        """
        if message.message_id in self.seen_updates:
            return False

        self.seen_updates.add(message.message_id)

        if self.world_state is None:
            return False

        track_data = message.payload.get('track_data')
        if not track_data:
            return False

        return self._apply_entity_data('track', track_data)

    def _apply_entity_data(self, entity_type: str, entity_data: Dict[str, Any]) -> bool:
        """
        Apply entity data to world state.

        Args:
            entity_type: Type of entity ('asset', 'track', 'zone')
            entity_data: Serialized entity data

        Returns:
            True if applied successfully
        """
        if self.world_state is None:
            return False

        try:
            if entity_type == 'asset':
                asset = Asset.from_dict(entity_data)
                return self.world_state.upsert_asset(asset, from_sync=True)
            elif entity_type == 'track':
                track = Track.from_dict(entity_data)
                return self.world_state.upsert_track(track, from_sync=True)
            elif entity_type == 'zone':
                zone = EnhancedZone.from_dict(entity_data)
                return self.world_state.upsert_zone(zone, from_sync=True)
            else:
                return False
        except (KeyError, ValueError):
            # Invalid entity data
            return False

    # ========== Intent synchronization ==========

    def set_intent_manager(self, intent_manager: IntentManager) -> None:
        """Set the intent manager to synchronize."""
        self.intent_manager = intent_manager

    def create_intent_message(self) -> Optional[Message]:
        """
        Create an intent declaration message if there's a pending intent.

        Returns:
            Message if there's a pending intent, None otherwise
        """
        if self.intent_manager is None:
            return None

        pending_intent = self.intent_manager.get_pending_intent()
        if pending_intent is None:
            return None

        return create_intent_declare(self.node_id, pending_intent.to_dict())

    def apply_intent_declare(self, message: Message) -> bool:
        """
        Apply an intent declaration from peer.

        Returns:
            True if the intent was applied, False if rejected/ignored
        """
        if message.message_id in self.seen_updates:
            return False

        self.seen_updates.add(message.message_id)

        if self.intent_manager is None:
            return False

        intent_data = message.payload.get('intent_data')
        if not intent_data:
            return False

        try:
            intent = Intent.from_dict(intent_data)
            return self.intent_manager.apply_intent(intent, from_sync=True)
        except (KeyError, ValueError):
            return False

    def apply_intent_release(self, message: Message) -> bool:
        """
        Apply an intent release from peer.

        Returns:
            True if processed, False if ignored
        """
        if message.message_id in self.seen_updates:
            return False

        self.seen_updates.add(message.message_id)

        if self.intent_manager is None:
            return False

        # When a drone releases intent, we can remove it from our tracking
        # The sender_id tells us which drone released their intent
        if message.sender_id in self.intent_manager.all_intents:
            del self.intent_manager.all_intents[message.sender_id]
            return True

        return False


def create_message_handlers(synchronizer: StateSynchronizer):
    """
    Create message handlers for the mesh node.

    Returns a dict mapping MessageType to handler functions.
    """
    def handle_state_sync(message: Message):
        synchronizer.apply_sync_message(message)

    def handle_entity_update(message: Message):
        synchronizer.apply_entity_update(message)

    def handle_entity_batch(message: Message):
        synchronizer.apply_entity_batch(message)

    def handle_track_alert(message: Message):
        synchronizer.apply_track_alert(message)

    def handle_intent_declare(message: Message):
        synchronizer.apply_intent_declare(message)

    def handle_intent_release(message: Message):
        synchronizer.apply_intent_release(message)

    return {
        MessageType.STATE_SYNC: handle_state_sync,
        MessageType.ENTITY_UPDATE: handle_entity_update,
        MessageType.ENTITY_BATCH: handle_entity_batch,
        MessageType.TRACK_ALERT: handle_track_alert,
        MessageType.INTENT_DECLARE: handle_intent_declare,
        MessageType.INTENT_RELEASE: handle_intent_release,
    }
