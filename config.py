"""Configuration constants for the drone swarm system."""

# Region dimensions (meters)
REGION_WIDTH = 50.0  # X-axis
REGION_HEIGHT = 30.0  # Y-axis

# Drone configuration
NUM_DRONES = 3
DRONE_SPEED = 5.0  # meters per second
PHOTO_INTERVAL = 2.0  # seconds between photos
DRONE_ALTITUDES = [10.0, 12.0, 14.0]  # Different altitudes for safety

# Zone configuration
ZONE_BUFFER = 0.0  # No buffer - use altitude separation for collision avoidance

# Collision avoidance
MIN_SAFE_DISTANCE = 5.0  # minimum distance between any two drones

# Heartbeat configuration
HEARTBEAT_INTERVAL = 2.0  # seconds
HEARTBEAT_TIMEOUT = 3  # missed heartbeats before considered failed

# Simulation
SIMULATION_TIMESTEP = 0.1  # seconds per simulation tick
LAWNMOWER_STRIP_WIDTH = 5.0  # width of each pass in lawnmower pattern

# Coverage map
COVERAGE_CELL_SIZE = 5.0  # size of each coverage cell in meters
PHOTO_COVERAGE_RADIUS = 3.0  # radius of area covered by one photo

# Goal planning
GOAL_REACHED_THRESHOLD = 2.5  # distance at which goal is considered reached
GOAL_REPLAN_INTERVAL = 1.0  # seconds between goal replanning

# Mesh communication
MESH_RANGE = 40.0           # Radio range in meters (covers 50x30 region)
MESH_UPDATE_INTERVAL = 0.2  # Seconds between mesh updates
MESSAGE_TTL = 3             # Max hops for gossip propagation
PEER_TIMEOUT = 2.0          # Seconds before peer considered lost

# World State configuration
WORLD_STATE_SYNC_INTERVAL = 0.5  # Seconds between entity sync broadcasts
ASSET_STALE_TIMEOUT = 5.0        # Seconds before asset marked offline
TRACK_STALE_TIMEOUT = 10.0       # Seconds before track marked lost
TRACK_EXPIRE_TIMEOUT = 30.0      # Seconds before track removed entirely
SPATIAL_INDEX_CELL_SIZE = 10.0   # Cell size for spatial indexing (meters)

# Intent-based tasking configuration
INTENT_EXPIRY_TIME = 3.0         # Seconds before intent expires if not renewed
INTENT_UPDATE_INTERVAL = 1.0     # Seconds between intent re-evaluation
INTENT_RENEW_INTERVAL = 2.0      # Seconds between intent renewals
