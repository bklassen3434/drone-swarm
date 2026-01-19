"""Message types for mesh communication in the drone swarm system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict
from uuid import uuid4
import time


class MessageType(Enum):
    """Types of messages sent between mesh nodes."""
    STATE_SYNC = "state_sync"         # Coverage map update
    ENTITY_UPDATE = "entity_update"   # Single entity update
    ENTITY_BATCH = "entity_batch"     # Batch of entity updates
    TRACK_ALERT = "track_alert"       # High-priority track notification
    INTENT_DECLARE = "intent_declare" # Declare intent to perform task
    INTENT_RELEASE = "intent_release" # Release/cancel current intent


@dataclass
class Message:
    """
    A message sent between mesh nodes.

    Messages are propagated through the mesh via gossip protocol.
    The TTL (time-to-live) limits how many hops a message can travel.
    """
    sender_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ttl: int = 3  # Max hops for gossip propagation
    message_id: str = field(default_factory=lambda: str(uuid4()))

    def decrement_ttl(self) -> 'Message':
        """Create copy with decremented TTL for forwarding."""
        return Message(
            sender_id=self.sender_id,
            message_type=self.message_type,
            payload=self.payload,
            timestamp=self.timestamp,
            ttl=self.ttl - 1,
            message_id=self.message_id
        )

    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        return self.ttl <= 0

    def __hash__(self):
        return hash(self.message_id)

    def __eq__(self, other):
        if isinstance(other, Message):
            return self.message_id == other.message_id
        return False


def create_state_sync(node_id: str, covered_cells: list) -> Message:
    """
    Create a state sync message for coverage updates.

    Args:
        node_id: ID of the sending node
        covered_cells: List of (grid_x, grid_y, drone_id) tuples for newly covered cells
    """
    return Message(
        sender_id=node_id,
        message_type=MessageType.STATE_SYNC,
        payload={
            'covered_cells': covered_cells
        }
    )


def create_entity_update(node_id: str, entity_type: str, entity_data: Dict[str, Any]) -> Message:
    """
    Create an entity update message for a single entity.

    Args:
        node_id: ID of the sending node
        entity_type: Type of entity ('asset', 'track', 'zone')
        entity_data: Serialized entity data (from entity.to_dict())

    Returns:
        Message to broadcast to peers
    """
    return Message(
        sender_id=node_id,
        message_type=MessageType.ENTITY_UPDATE,
        payload={
            'entity_type': entity_type,
            'entity_data': entity_data
        }
    )


def create_entity_batch(node_id: str, updates: list) -> Message:
    """
    Create a batch entity update message.

    Args:
        node_id: ID of the sending node
        updates: List of (entity_type, entity_id, entity_data) tuples

    Returns:
        Message to broadcast to peers
    """
    return Message(
        sender_id=node_id,
        message_type=MessageType.ENTITY_BATCH,
        payload={
            'updates': [
                {'entity_type': t, 'entity_id': i, 'entity_data': d}
                for t, i, d in updates
            ]
        }
    )


def create_track_alert(node_id: str, track_data: Dict[str, Any], alert_reason: str) -> Message:
    """
    Create a high-priority track alert message.

    Track alerts have a higher TTL (5) to ensure they propagate further
    through the mesh for important detections.

    Args:
        node_id: ID of the sending node
        track_data: Serialized track data (from track.to_dict())
        alert_reason: Reason for the alert (e.g., 'new_detection', 'classification_change')

    Returns:
        Message to broadcast to peers (with TTL=5)
    """
    return Message(
        sender_id=node_id,
        message_type=MessageType.TRACK_ALERT,
        payload={
            'track_data': track_data,
            'alert_reason': alert_reason
        },
        ttl=5  # Higher TTL for important alerts
    )


def create_intent_declare(node_id: str, intent_data: Dict[str, Any]) -> Message:
    """
    Create an intent declaration message.

    Broadcasts a drone's intent to perform a task, allowing other drones
    to coordinate and avoid conflicts.

    Args:
        node_id: ID of the sending node
        intent_data: Serialized intent data (from intent.to_dict())

    Returns:
        Message to broadcast to peers
    """
    return Message(
        sender_id=node_id,
        message_type=MessageType.INTENT_DECLARE,
        payload={
            'intent_data': intent_data
        }
    )


def create_intent_release(node_id: str, intent_id: str) -> Message:
    """
    Create an intent release message.

    Signals that a drone is releasing/canceling its current intent,
    making the task available for others.

    Args:
        node_id: ID of the sending node
        intent_id: ID of the intent being released

    Returns:
        Message to broadcast to peers
    """
    return Message(
        sender_id=node_id,
        message_type=MessageType.INTENT_RELEASE,
        payload={
            'intent_id': intent_id
        }
    )
