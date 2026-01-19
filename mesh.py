"""Mesh networking for peer-to-peer drone communication."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple
import math
import time

from messages import Message, MessageType
import config


@dataclass
class PeerInfo:
    """Information about a known peer in the mesh."""
    node_id: str
    position: Tuple[float, float]
    last_seen: float
    signal_strength: float  # Simulated based on distance (0.0 to 1.0)

    def is_stale(self, timeout: float = None) -> bool:
        """Check if peer hasn't been heard from recently."""
        if timeout is None:
            timeout = config.PEER_TIMEOUT
        return (time.time() - self.last_seen) > timeout


@dataclass
class MeshNode:
    """
    Each drone runs a mesh node for peer-to-peer communication.

    The mesh node handles:
    - Peer discovery based on simulated radio range
    - Message broadcasting (gossip protocol)
    - Direct messaging to specific peers
    - Message deduplication to prevent loops
    """
    node_id: str
    position: Tuple[float, float] = (0.0, 0.0)
    peers: Dict[str, PeerInfo] = field(default_factory=dict)
    message_queue: List[Message] = field(default_factory=list)
    handlers: Dict[MessageType, List[Callable]] = field(default_factory=dict)
    seen_messages: Set[str] = field(default_factory=set)
    _last_discovery: float = 0.0
    _mesh_network: Optional['MeshNetwork'] = field(default=None, repr=False)

    def update_position(self, x: float, y: float) -> None:
        """Update node position (affects peer discovery and signal strength)."""
        self.position = (x, y)

    def set_network(self, network: 'MeshNetwork') -> None:
        """Connect this node to the mesh network."""
        self._mesh_network = network

    def discover_peers(self, all_nodes: List['MeshNode']) -> None:
        """
        Find peers within radio range.

        This simulates radio-based peer discovery where nodes can only
        see other nodes within MESH_RANGE distance.
        """
        current_time = time.time()
        my_x, my_y = self.position

        for node in all_nodes:
            if node.node_id == self.node_id:
                continue

            # Calculate distance to peer
            peer_x, peer_y = node.position
            distance = math.sqrt((my_x - peer_x) ** 2 + (my_y - peer_y) ** 2)

            if distance <= config.MESH_RANGE:
                # Calculate signal strength (inverse of distance, normalized)
                signal = max(0.0, 1.0 - (distance / config.MESH_RANGE))

                self.peers[node.node_id] = PeerInfo(
                    node_id=node.node_id,
                    position=node.position,
                    last_seen=current_time,
                    signal_strength=signal
                )
            elif node.node_id in self.peers:
                # Peer moved out of range - mark as stale but don't remove yet
                # (will be cleaned up if not re-discovered)
                pass

        # Clean up stale peers
        stale_peers = [
            peer_id for peer_id, peer in self.peers.items()
            if peer.is_stale()
        ]
        for peer_id in stale_peers:
            del self.peers[peer_id]

    def broadcast(self, message: Message) -> None:
        """
        Send message to all peers (gossip protocol).

        Messages are forwarded to all reachable peers, who then forward
        to their peers (up to TTL hops).
        """
        if self._mesh_network is None:
            return

        # Mark message as seen to prevent loops
        self.seen_messages.add(message.message_id)

        # Send to all peers
        for peer_id in self.peers:
            self._mesh_network.deliver(peer_id, message)

    def send_to(self, peer_id: str, message: Message) -> bool:
        """
        Direct message to specific peer (if reachable).

        Returns True if peer is reachable, False otherwise.
        """
        if peer_id not in self.peers:
            return False

        if self._mesh_network is None:
            return False

        self.seen_messages.add(message.message_id)
        self._mesh_network.deliver(peer_id, message)
        return True

    def receive(self, message: Message) -> None:
        """
        Receive a message from the network.

        Messages are queued for processing.
        """
        # Check for duplicates
        if message.message_id in self.seen_messages:
            return

        self.seen_messages.add(message.message_id)
        self.message_queue.append(message)

        # Forward message if TTL > 0 (gossip propagation)
        if message.ttl > 1:
            forwarded = message.decrement_ttl()
            # Forward to peers except original sender
            for peer_id in self.peers:
                if peer_id != message.sender_id:
                    if self._mesh_network:
                        self._mesh_network.deliver(peer_id, forwarded)

    def on_message(self, msg_type: MessageType, handler: Callable) -> None:
        """Register handler for message type."""
        if msg_type not in self.handlers:
            self.handlers[msg_type] = []
        self.handlers[msg_type].append(handler)

    def process_messages(self) -> None:
        """Process all pending messages through registered handlers."""
        messages_to_process = self.message_queue.copy()
        self.message_queue.clear()

        for message in messages_to_process:
            if message.message_type in self.handlers:
                for handler in self.handlers[message.message_type]:
                    handler(message)

    def get_peer_count(self) -> int:
        """Get number of active peers."""
        return len(self.peers)

    def get_peer_positions(self) -> List[Tuple[str, Tuple[float, float]]]:
        """Get list of (peer_id, position) tuples."""
        return [(peer_id, peer.position) for peer_id, peer in self.peers.items()]

    def cleanup_old_messages(self) -> None:
        """Remove old message IDs to prevent memory growth."""
        # In a real system, we'd track timestamps for each message ID
        # For simplicity, we just limit the size of seen_messages
        if len(self.seen_messages) > 1000:
            # Keep only the most recent half
            self.seen_messages = set(list(self.seen_messages)[-500:])


@dataclass
class MeshNetwork:
    """
    Simulates the mesh network connecting all nodes.

    This class handles message routing between nodes, simulating
    the radio-based communication with range limits.
    """
    nodes: Dict[str, MeshNode] = field(default_factory=dict)
    debug: bool = False

    def register(self, node: MeshNode) -> None:
        """Register a node with the network."""
        self.nodes[node.node_id] = node
        node.set_network(self)

    def unregister(self, node_id: str) -> None:
        """Remove a node from the network."""
        if node_id in self.nodes:
            del self.nodes[node_id]

    def deliver(self, target_id: str, message: Message) -> bool:
        """
        Deliver a message to a target node.

        Returns True if delivered, False if target not found.
        """
        if target_id not in self.nodes:
            return False

        target = self.nodes[target_id]
        target.receive(message)

        if self.debug:
            print(f"[Mesh] {message.sender_id} -> {target_id}: {message.message_type.value}")

        return True

    def update_all(self) -> None:
        """
        Update peer discovery for all nodes.

        Call this periodically to simulate radio-based discovery.
        """
        node_list = list(self.nodes.values())
        for node in node_list:
            node.discover_peers(node_list)

    def get_connectivity_stats(self) -> Dict:
        """Get statistics about mesh connectivity."""
        if not self.nodes:
            return {'nodes': 0, 'avg_peers': 0, 'fully_connected': True}

        peer_counts = [node.get_peer_count() for node in self.nodes.values()]
        total_nodes = len(self.nodes)

        return {
            'nodes': total_nodes,
            'avg_peers': sum(peer_counts) / total_nodes if total_nodes > 0 else 0,
            'min_peers': min(peer_counts) if peer_counts else 0,
            'max_peers': max(peer_counts) if peer_counts else 0,
            'fully_connected': all(c == total_nodes - 1 for c in peer_counts)
        }
