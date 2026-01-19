# Drone Swarm Simulation

A decentralized drone swarm simulation featuring autonomous patrol, object tracking, and human-in-the-loop engagement authorization. Inspired by modern autonomous defense systems.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Decentralized Mesh Network**: Drones communicate peer-to-peer using gossip protocol - no central coordinator required
- **Autonomous Patrol**: Drones patrol assigned zones using figure-8 patterns for optimal coverage
- **Object Detection & Tracking**: Automatic detection and tracking of moving objects
- **Kill Chain Workflow**: Full engagement pipeline with human authorization requirement
  - DETECTED → TRACKING → IDENTIFIED → AWAITING_AUTH → AUTHORIZED → ENGAGED → NEUTRALIZED
- **Fault Tolerance**: Automatic zone rebalancing when drones fail
- **Real-time Visualization**: Live dashboard showing drone status, tracked objects, and authorization queue

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Simulation Layer                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │ Drone 0 │  │ Drone 1 │  │ Drone 2 │  │ Object Manager  │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └─────────────────┘ │
│       │            │            │                            │
│       └────────────┼────────────┘                            │
│                    │                                         │
│            ┌───────┴───────┐                                 │
│            │  Mesh Network │  (Gossip Protocol)              │
│            └───────────────┘                                 │
└─────────────────────────────────────────────────────────────┘

Each Drone Contains:
┌─────────────────────────────────────┐
│  ┌─────────────┐  ┌──────────────┐  │
│  │ World State │  │Intent Manager│  │
│  │  (CRDT)     │  │  (Tasks)     │  │
│  └─────────────┘  └──────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  │
│  │  Mesh Node  │  │ Goal Planner │  │
│  └─────────────┘  └──────────────┘  │
└─────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drone-swarm.git
cd drone-swarm

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Run

```bash
python main.py
```

### Command Line Options

```bash
python main.py --help

Options:
  --drones N        Number of drones (default: 3)
  --width W         Region width in meters (default: 60)
  --height H        Region height in meters (default: 40)
  --time T          Max simulation time in seconds (default: 300)
  --no-viz          Run without visualization
  --fast            Run as fast as possible (not real-time)
  --debug-mesh      Show mesh network debug info
  --test            Test mode with scheduled drone failure
  --fail ID --at T  Fail specific drone at time T
```

### Examples

```bash
# Run with 5 drones
python main.py --drones 5

# Test fault tolerance (drone 1 fails at 10s)
python main.py --test

# Run headless for performance testing
python main.py --no-viz --fast --time 60

# Large region with more drones
python main.py --drones 7 --width 100 --height 80
```

### Controls

During simulation:
- **Y** - Authorize engagement on pending target
- **N** - Deny engagement (drone returns to patrol)
- **Q** - Quit simulation

## Project Structure

```
drone_swarm/
├── main.py           # Entry point
├── simulation.py     # Main simulation and visualization
├── drone.py          # Drone agent with patrol/tracking behavior
├── intents.py        # Intent-based task system
├── entities.py       # Core data types (Asset, Track, Zone)
├── world_state.py    # Distributed world model (CRDT-style)
├── mesh.py           # Peer-to-peer mesh network
├── messages.py       # Network message types
├── sync.py           # State synchronization
├── objects.py        # Simulated moving objects
├── zone.py           # Zone partitioning
├── coverage_map.py   # Coverage tracking
├── collision.py      # Collision avoidance
├── goal_planner.py   # Navigation planning
├── config.py         # Configuration constants
└── drone.jpg         # Drone icon for visualization
```

## Key Concepts

### Intent-Based Tasking

Drones declare their intentions (PATROL, TRACK, IDENTIFY, AWAIT_AUTH, ENGAGE) which are broadcast to peers. This enables coordination without central control.

### Kill Chain

Objects progress through stages requiring human authorization before engagement:

1. **DETECTED** - Object first seen by a drone
2. **TRACKING** - Drone actively following the object
3. **IDENTIFIED** - Sufficient observations to classify
4. **AWAITING_AUTH** - Waiting for human Y/N decision
5. **AUTHORIZED** - Human approved engagement
6. **ENGAGED** - Drone executing engagement
7. **NEUTRALIZED** - Target eliminated

### World State Synchronization

Each drone maintains its own world state using CRDT-style conflict resolution:
- Higher version numbers win
- Timestamps break ties
- Eventually consistent across the swarm

### Zone Rebalancing

When a drone starts tracking or fails, remaining patrol drones automatically rebalance to cover all zones.

## Configuration

Edit `config.py` to adjust parameters:

```python
# Drone behavior
DRONE_SPEED = 5.0           # meters per second
MESH_RANGE = 30.0           # communication range

# Kill chain timing
INTENT_UPDATE_INTERVAL = 0.5
WORLD_STATE_SYNC_INTERVAL = 0.5

# Detection
TRACK_STALE_TIMEOUT = 10.0  # seconds before track marked lost
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by autonomous swarm systems and modern defense architectures.
