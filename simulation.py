"""2D simulation and visualization for the drone swarm system."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from typing import Optional, List, Dict
import numpy as np
import math
import os

import config
from zone import Region, Zone, ZoneStatus, partition_region
from drone import Drone, DroneStatus, Photo
from coverage_map import CoverageMap
from collision import check_all_collisions, assign_safe_altitudes
from mesh import MeshNetwork
from entities import (
    EnhancedZone, EnhancedZoneStatus, ZonePriority,
    ThreatLevel, KillChainStage, TrackStatus
)
from objects import ObjectManager, SimulatedObject


class Simulation:
    """
    2D simulation with matplotlib visualization.

    All drones communicate via mesh network - no central coordinator.
    Collision avoidance via altitude separation.
    """

    DRONE_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    ZONE_COLORS = ['#fee5d9', '#d4e4f4', '#d5e8d4', '#e8d4e8', '#ffecd4']

    def __init__(
        self,
        region: Optional[Region] = None,
        num_drones: int = config.NUM_DRONES,
        debug_mesh: bool = False
    ):
        """
        Initialize the simulation.

        Args:
            region: Optional region (defaults to config values)
            num_drones: Number of drones to simulate
            debug_mesh: If True, print mesh debug information
        """
        self.region = region or Region()
        self.num_drones = num_drones
        self.debug_mesh = debug_mesh

        # Core components
        self.drones: List[Drone] = []
        self.zones: List[Zone] = []
        self.coverage_map: Optional[CoverageMap] = None
        self.mesh_network: Optional[MeshNetwork] = None
        self.object_manager: Optional[ObjectManager] = None
        self.all_photos: List[Photo] = []

        # Visualization
        self.fig = None
        self.ax = None
        self.ax_dashboard = None
        self.drone_artists = []
        self.object_artists = []  # For drawing detected objects
        self.zone_artists = []  # For drawing dynamic zones
        self.dashboard_text = None

        # Simulation state
        self.simulation_time = 0.0
        self.is_running = False
        self.failure_scheduled = None

        # Dynamic zone management
        self.patrol_drone_ids: set = set()  # IDs of drones currently patrolling
        self.tracking_drone_ids: set = set()  # IDs of drones currently tracking

        # Kill chain / authorization management
        self.pending_authorizations: List[str] = []  # Queue of track IDs awaiting authorization (FIFO)
        self.authorized_tracks: set = set()  # Track IDs that have been authorized
        self.neutralized_count: int = 0  # Count of neutralized targets

        # Initialize everything
        self._initialize()

    def _initialize(self) -> None:
        """Initialize drones, zones, coverage map, object manager, and mesh network."""
        # Create coverage map (kept for compatibility, but not used for coverage anymore)
        self.coverage_map = CoverageMap(
            width=self.region.width,
            height=self.region.height,
            cell_size=config.COVERAGE_CELL_SIZE,
            x_origin=self.region.x_origin,
            y_origin=self.region.y_origin
        )

        # Create object manager for simulated moving objects
        self.object_manager = ObjectManager(
            region_width=self.region.width,
            region_height=self.region.height
        )
        # Spawn initial objects
        self.object_manager.spawn_object()
        self.object_manager.spawn_object()

        # Create drones
        self.drones = []
        for i in range(self.num_drones):
            drone = Drone(
                id=i,
                altitude=config.DRONE_ALTITUDES[i % len(config.DRONE_ALTITUDES)]
            )
            self.drones.append(drone)

        # Assign safe altitudes (collision avoidance)
        assign_safe_altitudes(self.drones)

        # Set up mesh network
        self.mesh_network = MeshNetwork(debug=self.debug_mesh)
        for drone in self.drones:
            if drone.mesh_node:
                self.mesh_network.register(drone.mesh_node)
            drone.enable_mesh(self.coverage_map)

        # All drones start as patrol drones
        self.patrol_drone_ids = set(d.id for d in self.drones)
        self.tracking_drone_ids = set()

        # Create initial zones for all drones and assign them
        self._rebalance_patrol_zones()

    def _rebalance_patrol_zones(self) -> None:
        """
        Rebalance patrol zones among currently patrolling drones.

        Called when:
        - A drone starts tracking (leaves patrol pool)
        - A drone finishes tracking (rejoins patrol pool)
        - Initialization
        """
        patrol_drones = [d for d in self.drones if d.id in self.patrol_drone_ids and d.is_alive()]

        if not patrol_drones:
            self.zones = []
            return

        # Create new zones based on number of patrol drones
        self.zones = partition_region(self.region, len(patrol_drones))

        # Assign each patrol drone to a zone
        for i, drone in enumerate(patrol_drones):
            if i < len(self.zones):
                zone = self.zones[i]
                zone.assigned_drone_id = drone.id

                # Clear old zones and assign new one
                drone.assigned_zones = [zone]

                # Position drone in their new zone if they're far away
                if not zone.contains_point(drone.x, drone.y):
                    # Move toward zone center
                    drone.x = zone.center[0]
                    drone.y = zone.center[1]

                # Generate new patrol waypoints for the new zone
                drone._generate_patrol_waypoints(zone)
                drone._current_waypoint_idx = 0

                # Update drone's intent to patrol new zone
                if drone.intent_manager:
                    from intents import IntentType
                    drone.intent_manager.set_intent(
                        intent_type=IntentType.PATROL,
                        target_id=f"zone_{zone.id}",
                        target_position=zone.center
                    )

        # Register zones in all drones' world states
        for drone in self.drones:
            if drone.is_alive():
                for i, zone in enumerate(self.zones):
                    self._register_zone_in_world_state(drone, zone, zone.assigned_drone_id)

        print(f"[Rebalance] {len(patrol_drones)} patrol drones, {len(self.tracking_drone_ids)} tracking")

    def _drone_starts_tracking(self, drone_id: int, object_id: str) -> None:
        """Called when a drone starts tracking an object."""
        if drone_id in self.patrol_drone_ids:
            self.patrol_drone_ids.remove(drone_id)
            self.tracking_drone_ids.add(drone_id)
            print(f"[Tracking] Drone {drone_id} started tracking {object_id}")
            self._rebalance_patrol_zones()

    def _drone_stops_tracking(self, drone_id: int) -> None:
        """Called when a drone stops tracking and returns to patrol."""
        if drone_id in self.tracking_drone_ids:
            self.tracking_drone_ids.remove(drone_id)
            self.patrol_drone_ids.add(drone_id)
            print(f"[Tracking] Drone {drone_id} returned to patrol")
            self._rebalance_patrol_zones()

    def _check_pending_authorizations(self) -> None:
        """Check for tracks awaiting authorization and update pending list."""
        # Check ALL drones' world states for tracks awaiting authorization
        # (each drone tracks its own target, so we need to check all)
        for drone in self.drones:
            if not drone.is_alive() or not drone.world_state:
                continue

            for track in drone.world_state.tracks.values():
                if track.kill_chain_stage == KillChainStage.AWAITING_AUTH:
                    if track.track_id not in self.pending_authorizations:
                        # Also check it's not already authorized or neutralized
                        if track.track_id not in self.authorized_tracks:
                            self.pending_authorizations.append(track.track_id)
                            queue_pos = len(self.pending_authorizations)
                            print(f"[Authorization] {track.track_id} added to queue (position {queue_pos})")

        # Clean up queue: remove entries for objects that no longer exist or are neutralized
        valid_entries = []
        for track_id in self.pending_authorizations:
            # Check if object still exists
            obj_exists = self.object_manager and self.object_manager.get_object_by_id(track_id) is not None

            # Check if track is still in AWAITING_AUTH (not yet authorized/neutralized)
            track_still_waiting = False
            for drone in self.drones:
                if drone.is_alive() and drone.world_state:
                    track = drone.world_state.get_track(track_id)
                    if track and track.kill_chain_stage == KillChainStage.AWAITING_AUTH:
                        track_still_waiting = True
                        break

            if obj_exists and track_still_waiting:
                valid_entries.append(track_id)
            elif track_id in self.pending_authorizations:
                print(f"[Authorization] Removed {track_id} from queue (object despawned or already processed)")

        self.pending_authorizations = valid_entries

    def _authorize_next(self) -> None:
        """Authorize the next pending track."""
        if not self.pending_authorizations:
            print("[Authorization] No pending requests")
            return

        track_id = self.pending_authorizations.pop(0)
        self.authorized_tracks.add(track_id)
        remaining = len(self.pending_authorizations)

        # Update track in all drones' world states
        for drone in self.drones:
            if drone.is_alive() and drone.world_state:
                track = drone.world_state.get_track(track_id)
                if track:
                    track.authorize()
                    drone.world_state.upsert_track(track)

        print(f"[Authorization] AUTHORIZED: Engagement on {track_id} ({remaining} remaining in queue)")

    def _deny_next(self) -> None:
        """Deny the next pending authorization."""
        if not self.pending_authorizations:
            print("[Authorization] No pending requests")
            return

        track_id = self.pending_authorizations.pop(0)
        remaining = len(self.pending_authorizations)

        # Update track in all drones' world states
        for drone in self.drones:
            if drone.is_alive() and drone.world_state:
                track = drone.world_state.get_track(track_id)
                if track:
                    track.deny_authorization()
                    drone.world_state.upsert_track(track)

        # Find the drone tracking this target and have it return to patrol
        for drone in self.drones:
            if drone.is_alive():
                intent = drone.get_current_intent()
                if intent and intent.target_id == track_id:
                    drone._return_to_patrol()
                    self._drone_stops_tracking(drone.id)
                    break

        print(f"[Authorization] DENIED: {track_id} - drone returning to patrol ({remaining} remaining in queue)")

    def _process_authorized_tracks(self) -> None:
        """Process tracks that have been authorized for engagement."""
        ref_drone = next((d for d in self.drones if d.is_alive() and d.world_state), None)
        if not ref_drone:
            return

        for track in list(ref_drone.world_state.tracks.values()):
            if track.kill_chain_stage == KillChainStage.AUTHORIZED:
                # Find the drone tracking this target
                for drone in self.drones:
                    if drone.is_alive():
                        intent = drone.get_current_intent()
                        if intent and intent.target_id == track.track_id:
                            # Simulate engagement
                            track.engage(f"drone_{drone.id}")
                            print(f"[Engagement] Drone {drone.id} engaging {track.track_id}")

                            # After a moment, neutralize (we'll do this instantly for simulation)
                            track.kill_chain_stage = KillChainStage.NEUTRALIZED
                            self.neutralized_count += 1
                            print(f"[Engagement] {track.track_id} NEUTRALIZED")

                            # Remove the object from simulation
                            if self.object_manager:
                                self.object_manager.remove_object(track.track_id)

                            # Drone returns to patrol and trigger zone rebalancing
                            drone._return_to_patrol()
                            self._drone_stops_tracking(drone.id)
                            break

                # Update track in all world states
                for drone in self.drones:
                    if drone.is_alive() and drone.world_state:
                        drone.world_state.upsert_track(track)

    def _register_zone_in_world_state(self, drone: Drone, zone: Zone, assigned_drone_id: int) -> None:
        """Register a zone in a drone's world state for failure detection."""
        if drone.world_state is None:
            return

        enhanced_zone = EnhancedZone(
            zone_id=f"zone_{zone.id}",
            x_min=zone.x_min,
            x_max=zone.x_max,
            y_min=zone.y_min,
            y_max=zone.y_max,
            name=f"Zone {zone.id}",
            priority=ZonePriority.NORMAL,
            status=EnhancedZoneStatus.IN_PROGRESS,
            assigned_asset_ids=[f"drone_{assigned_drone_id}"] if assigned_drone_id is not None else [],
            coverage_progress=0.0,
        )
        drone.world_state.upsert_zone(enhanced_zone)

    def setup_plot(self) -> None:
        """Set up the matplotlib figure and axes with dashboard."""
        # Create figure with main plot and dashboard panel
        self.fig = plt.figure(figsize=(18, 9))

        # Main simulation view (left, takes 65% of width)
        self.ax = self.fig.add_axes([0.05, 0.1, 0.55, 0.85])
        self.ax.set_xlim(-5, self.region.width + 5)
        self.ax.set_ylim(-5, self.region.height + 5)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Drone Swarm - Mesh Network Communication')
        self.ax.grid(True, alpha=0.3)

        # Dashboard panel (right, takes 30% of width)
        self.ax_dashboard = self.fig.add_axes([0.65, 0.1, 0.33, 0.85])
        self.ax_dashboard.set_xlim(0, 1)
        self.ax_dashboard.set_ylim(0, 1)
        self.ax_dashboard.axis('off')
        self.ax_dashboard.set_title('Dashboard', fontsize=14, fontweight='bold')

        self._draw_zones()
        self._init_drone_markers()
        self._init_object_markers()

        # Connect key press event for quitting
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        # Dashboard text area (no text on main simulation view)
        self.dashboard_text = self.ax_dashboard.text(
            0.02, 0.98, '', transform=self.ax_dashboard.transAxes,
            verticalalignment='top', fontsize=9,
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        )

        self._add_legend()

    def _on_key_press(self, event) -> None:
        """Handle key press events."""
        if event.key == 'q':
            self.is_running = False
            plt.close(self.fig)
        elif event.key == 'y':
            self._authorize_next()
        elif event.key == 'n':
            self._deny_next()

    def _draw_zones(self) -> None:
        """Draw initial zone rectangles (will be updated dynamically)."""
        self._update_zone_visualization()

    def _update_zone_visualization(self) -> None:
        """Update zone visualization (called when zones change)."""
        # Remove old zone artists
        for artist_dict in self.zone_artists:
            if 'rect' in artist_dict:
                artist_dict['rect'].remove()
            if 'label' in artist_dict:
                artist_dict['label'].remove()
        self.zone_artists = []

        # Draw current zones
        for i, zone in enumerate(self.zones):
            # Color based on assigned drone
            if zone.assigned_drone_id is not None:
                color = self.DRONE_COLORS[zone.assigned_drone_id % len(self.DRONE_COLORS)]
            else:
                color = '#cccccc'

            rect = patches.Rectangle(
                (zone.x_min, zone.y_min),
                zone.width, zone.height,
                linewidth=2, edgecolor='black',
                facecolor=color, alpha=0.2,
                zorder=1
            )
            self.ax.add_patch(rect)

            # Label with assigned drone
            drone_str = f"D{zone.assigned_drone_id}" if zone.assigned_drone_id is not None else "?"
            label = self.ax.text(
                zone.center[0], zone.center[1],
                f'{drone_str}',
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='gray', alpha=0.7
            )

            self.zone_artists.append({
                'rect': rect,
                'label': label,
                'zone_id': zone.id
            })

    def _load_drone_image(self, color: str, size: int = 40) -> np.ndarray:
        """Load drone image and colorize it."""
        # Get path to drone.jpg relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, 'drone.jpg')

        # Load image
        img = Image.open(image_path).convert('RGBA')

        # Resize
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img)

        # Colorize: replace black pixels with the drone's color
        # Convert hex color to RGB
        hex_color = color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Find dark pixels (the drone silhouette) and colorize them
        # The image has black drone on light background
        dark_mask = (img_array[:, :, 0] < 100) & (img_array[:, :, 1] < 100) & (img_array[:, :, 2] < 100)

        # Set drone color
        img_array[dark_mask, 0] = rgb[0]
        img_array[dark_mask, 1] = rgb[1]
        img_array[dark_mask, 2] = rgb[2]
        img_array[dark_mask, 3] = 255  # Full opacity for drone

        # Make background transparent
        light_mask = ~dark_mask
        img_array[light_mask, 3] = 0  # Transparent background

        return img_array

    def _init_drone_markers(self) -> None:
        """Initialize drone marker artists using drone.jpg image."""
        self.drone_artists = []
        self.drone_images = {}  # Store colored images

        for i, drone in enumerate(self.drones):
            color = self.DRONE_COLORS[i % len(self.DRONE_COLORS)]

            # Load and colorize drone image
            img_array = self._load_drone_image(color, size=35)
            self.drone_images[i] = img_array

            # Create OffsetImage
            imagebox = OffsetImage(img_array, zoom=1.0)
            imagebox.image.axes = self.ax

            # Create AnnotationBbox at drone position
            ab = AnnotationBbox(imagebox, (drone.x, drone.y),
                               frameon=False, zorder=10)
            self.ax.add_artist(ab)

            # Trail and label
            trail, = self.ax.plot([], [], color=color, alpha=0.5, linewidth=1)
            label = self.ax.text(
                drone.x, drone.y + 4,
                f'D{drone.id}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=color
            )

            self.drone_artists.append({
                'image_box': ab,
                'imagebox': imagebox,
                'trail': trail,
                'label': label,
                'trail_x': [drone.x],
                'trail_y': [drone.y],
                'last_x': drone.x,
                'last_y': drone.y,
            })

    def _init_object_markers(self) -> None:
        """Initialize markers for simulated objects."""
        self.object_artists = []
        # We'll create/update these dynamically as objects spawn

    def _add_legend(self) -> None:
        """Add legend to the plot."""
        legend_elements = []
        for i in range(min(len(self.drones), len(self.DRONE_COLORS))):
            color = self.DRONE_COLORS[i]
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor='black',
                              label=f'Drone {i}')
            )
        # Add object types to legend
        legend_elements.append(
            patches.Circle((0, 0), radius=1, facecolor='yellow', edgecolor='orange',
                          label='Object (undetected)')
        )
        legend_elements.append(
            patches.Circle((0, 0), radius=1, facecolor='red', edgecolor='darkred',
                          label='Object (tracking)')
        )
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def update_visualization(self) -> None:
        """Update the visualization with current drone positions."""
        # Update zone visualization (handles dynamic zone changes)
        self._update_zone_visualization()

        for i, drone in enumerate(self.drones):
            artist = self.drone_artists[i]

            if drone.is_alive():
                # Update image position
                artist['image_box'].xybox = (drone.x, drone.y)
                artist['imagebox'].image.set_alpha(1.0)

                artist['label'].set_position((drone.x, drone.y + 4))
                artist['label'].set_alpha(1.0)
                artist['label'].set_text(f'D{drone.id}')

                artist['trail_x'].append(drone.x)
                artist['trail_y'].append(drone.y)
                if len(artist['trail_x']) > 100:
                    artist['trail_x'] = artist['trail_x'][-100:]
                    artist['trail_y'] = artist['trail_y'][-100:]
                artist['trail'].set_data(artist['trail_x'], artist['trail_y'])

                artist['last_x'] = drone.x
                artist['last_y'] = drone.y
            else:
                artist['imagebox'].image.set_alpha(0.3)
                artist['label'].set_alpha(0.3)
                artist['label'].set_text(f'D{drone.id} (FAILED)')

        # Update object visualization
        self._update_object_visualization()

        # Update dashboard
        if self.dashboard_text:
            dashboard_content = self._generate_dashboard_content()
            self.dashboard_text.set_text(dashboard_content)

    def _update_object_visualization(self) -> None:
        """Update the visualization of simulated objects."""
        if not self.object_manager:
            return

        # Remove old object artists
        for artist_dict in self.object_artists:
            if 'marker' in artist_dict:
                artist_dict['marker'].remove()
            if 'label' in artist_dict:
                artist_dict['label'].remove()
        self.object_artists = []

        # Create new artists for each object
        for obj in self.object_manager.objects:
            # Color based on detection status
            if obj.detected:
                color = 'red'
                edge_color = 'darkred'
            else:
                color = 'yellow'
                edge_color = 'orange'

            marker_size = 10

            # Create marker
            marker = self.ax.scatter(
                obj.x, obj.y,
                s=marker_size * 20,
                c=color,
                edgecolors=edge_color,
                linewidth=2,
                marker='o',
                zorder=8
            )

            # Create label
            status_str = "[TRACKING]" if obj.detected else ""
            label = self.ax.text(
                obj.x, obj.y - 2,
                f"{obj.object_id[-6:]} {status_str}",
                ha='center', va='top', fontsize=7,
                color='darkred' if obj.detected else 'darkorange',
                fontweight='bold'
            )

            self.object_artists.append({
                'marker': marker,
                'label': label,
                'object_id': obj.object_id
            })

    def _generate_dashboard_content(self) -> str:
        """Generate detailed dashboard content for patrol/tracking mode."""
        lines = []

        # Header with simulation time
        lines.append(f"{'═' * 40}")
        lines.append(f"  TIME: {self.simulation_time:.1f}s  |  NEUTRALIZED: {self.neutralized_count}")
        lines.append(f"{'═' * 40}")
        lines.append("")

        # PENDING AUTHORIZATION - show only the FIRST one (must handle sequentially)
        if self.pending_authorizations:
            track_id = self.pending_authorizations[0]  # First in queue
            queue_size = len(self.pending_authorizations)

            lines.append("╔═══════════════════════════════════════╗")
            lines.append("║      AUTHORIZATION REQUIRED           ║")
            lines.append("╠═══════════════════════════════════════╣")

            ref_drone = next((d for d in self.drones if d.is_alive() and d.world_state), None)
            if ref_drone:
                track = ref_drone.world_state.get_track(track_id)
                if track:
                    lines.append(f"║  Target: {track_id}")
                    lines.append(f"║  Confidence: {track.confidence:.0%}")
                    lines.append(f"║  Stage: {track.kill_chain_stage.value.upper()}")

            lines.append("║")
            lines.append("║      Y = AUTHORIZE  |  N = DENY       ║")
            if queue_size > 1:
                lines.append(f"║      ({queue_size - 1} more in queue)              ║")
            lines.append("╚═══════════════════════════════════════╝")
            lines.append("")

        # KILL CHAIN STATUS
        ref_drone = next((d for d in self.drones if d.is_alive() and d.world_state), None)
        if ref_drone and ref_drone.world_state.tracks:
            lines.append("┌─── KILL CHAIN ────────────────────────┐")
            for track in list(ref_drone.world_state.tracks.values())[:5]:
                stage = track.kill_chain_stage.value.upper()
                lines.append(f"│ {track.track_id[-8:]}: {stage}")
            lines.append("└───────────────────────────────────────┘")
            lines.append("")

        # DETECTED OBJECTS SECTION
        if self.object_manager:
            objects = self.object_manager.objects
            detected = [o for o in objects if o.detected]
            undetected = [o for o in objects if not o.detected]

            lines.append("┌─── OBJECTS ───────────────────────────┐")
            lines.append(f"│ Total: {len(objects)} | Tracked: {len(detected)} | Hidden: {len(undetected)}")
            for obj in objects:
                status = "TRACKING" if obj.detected else "hidden"
                lines.append(f"│ {obj.object_id[-8:]} @ ({obj.x:5.1f},{obj.y:5.1f}) {status}")
            lines.append("└───────────────────────────────────────┘")
            lines.append("")

        # DRONE STATUS (patrol vs tracking)
        lines.append("┌─── DRONE STATUS ──────────────────────┐")
        lines.append(f"│ Patrolling: {len(self.patrol_drone_ids)} | Engaged: {len(self.tracking_drone_ids)}")
        lines.append("│")
        for drone in self.drones:
            intent = drone.get_current_intent()
            status_icon = "✓" if drone.is_alive() else "✗"
            if intent:
                # Map intent to display string with icon
                intent_display = {
                    'patrol': '🔄 PATROL',
                    'takeover_zone': '🔄 TAKEOVER',
                    'track_object': '👁️  TRACK',
                    'identify': '🔍 IDENTIFY',
                    'await_auth': '⏳ AWAIT_AUTH',
                    'engage': '🎯 ENGAGE',
                    'idle': '💤 IDLE',
                }
                intent_str = intent_display.get(intent.intent_type.value, intent.intent_type.value.upper())
                target = intent.target_id or ""
                if len(target) > 12:
                    target = target[-12:]
                lines.append(f"│ {status_icon} D{drone.id}: {intent_str}")
                if target:
                    lines.append(f"│      └─ {target}")
            else:
                lines.append(f"│ {status_icon} D{drone.id}: 💤 IDLE")
        lines.append("└───────────────────────────────────────┘")
        lines.append("")

        # ZONES SECTION (who's patrolling)
        lines.append("┌─── ZONES ─────────────────────────────┐")
        for zone in self.zones:
            assigned = f"D{zone.assigned_drone_id}" if zone.assigned_drone_id is not None else "none"
            # Count objects in this zone
            objects_in_zone = 0
            if self.object_manager:
                for obj in self.object_manager.objects:
                    if obj.current_zone_id == f"zone_{zone.id}":
                        objects_in_zone += 1
            obj_str = f" ({objects_in_zone} obj)" if objects_in_zone > 0 else ""
            lines.append(f"│ Zone {zone.id}: Patrolled by {assigned}{obj_str}")
        lines.append("└───────────────────────────────────────┘")
        lines.append("")

        # MESH CONNECTIVITY
        if self.mesh_network:
            mesh_stats = self.mesh_network.get_connectivity_stats()
            lines.append("┌─── MESH NETWORK ──────────────────────┐")
            lines.append(f"│ Nodes: {mesh_stats.get('nodes', 0)}")
            lines.append(f"│ Avg peers: {mesh_stats.get('avg_peers', 0):.1f}")
            lines.append(f"│ Connected: {mesh_stats.get('fully_connected', False)}")
            lines.append("└───────────────────────────────────────┘")

        # Add press q to quit reminder
        lines.append("")
        lines.append("  Press 'q' to quit")

        return '\n'.join(lines)

    def step(self, dt: float = config.SIMULATION_TIMESTEP) -> dict:
        """Advance simulation by one timestep."""
        self.simulation_time += dt

        status = {
            'photos_taken': [],
            'collisions': [],
            'failed_drones': [],
            'completed': False,
            'mesh_stats': None,
            'objects_detected': []
        }

        # Update mesh network (peer discovery)
        if self.mesh_network:
            self.mesh_network.update_all()
            if self.debug_mesh:
                status['mesh_stats'] = self.mesh_network.get_connectivity_stats()

        # Update simulated objects
        if self.object_manager:
            self.object_manager.update(dt)
            # Update which zone each object is in
            for obj in self.object_manager.objects:
                obj.current_zone_id = self._get_zone_for_position(obj.x, obj.y)

        # Track state changes for rebalancing
        drones_started_tracking = []
        drones_stopped_tracking = []

        # Intent types that count as "engaged with target" (not patrolling)
        engaged_intents = {'track_object', 'identify', 'await_auth', 'engage'}

        # Update each drone
        for drone in self.drones:
            if not drone.is_alive():
                continue

            # Remember previous intent
            prev_intent = drone.get_current_intent()
            was_engaged = prev_intent and prev_intent.intent_type.value in engaged_intents

            photo = drone.update(dt, coverage_map=self.coverage_map,
                                object_manager=self.object_manager)

            if photo:
                self.all_photos.append(photo)
                status['photos_taken'].append(photo)

            # Check if intent changed
            curr_intent = drone.get_current_intent()
            is_engaged = curr_intent and curr_intent.intent_type.value in engaged_intents

            if not was_engaged and is_engaged:
                # Drone started tracking/engaging
                target_id = curr_intent.target_id if curr_intent else "unknown"
                drones_started_tracking.append((drone.id, target_id))
            elif was_engaged and not is_engaged:
                # Drone stopped tracking (object despawned or lost)
                drones_stopped_tracking.append(drone.id)

            # Also check if drone is engaged but object no longer exists
            if is_engaged and curr_intent and self.object_manager:
                tracked_obj = self.object_manager.get_object_by_id(curr_intent.target_id)
                if tracked_obj is None:
                    # Object despawned - drone should return to patrol
                    print(f"[Tracking] Drone {drone.id}'s target {curr_intent.target_id} despawned")

                    # Update the track to LOST so it's no longer in AWAITING_AUTH
                    if drone.world_state:
                        track = drone.world_state.get_track(curr_intent.target_id)
                        if track:
                            track.status = TrackStatus.LOST
                            # If it was awaiting auth, mark as no longer pending
                            if track.kill_chain_stage == KillChainStage.AWAITING_AUTH:
                                track.kill_chain_stage = KillChainStage.TRACKING  # Downgrade
                            drone.world_state.upsert_track(track)

                    drone._return_to_patrol()
                    drones_stopped_tracking.append(drone.id)

        # Process tracking state changes (triggers rebalancing)
        for drone_id, object_id in drones_started_tracking:
            self._drone_starts_tracking(drone_id, object_id)

        for drone_id in drones_stopped_tracking:
            self._drone_stops_tracking(drone_id)

        # Kill chain management
        self._check_pending_authorizations()
        self._process_authorized_tracks()

        # Check for collisions (warning only - altitude separation should prevent)
        collisions = check_all_collisions(self.drones)
        if collisions:
            status['collisions'] = collisions

        # Check for scheduled failure
        if self.failure_scheduled is not None:
            if self.simulation_time >= self.failure_scheduled[0]:
                drone_id = self.failure_scheduled[1]
                self.simulate_drone_failure(drone_id)
                self.failure_scheduled = None

        # In patrol mode, simulation runs continuously (no completion condition)
        # Only ends when max_time is reached or user quits
        return status

    def _get_zone_for_position(self, x: float, y: float) -> Optional[str]:
        """Get the zone ID for a given position."""
        for zone in self.zones:
            if zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max:
                return f"zone_{zone.id}"
        return None

    def simulate_drone_failure(self, drone_id: int) -> None:
        """Simulate a drone failure and propagate to other drones."""
        if 0 <= drone_id < len(self.drones):
            failed_drone = self.drones[drone_id]
            failed_drone.fail()
            print(f"[Simulation] Drone {drone_id} has failed")

            # Propagate the failed status to other drones' world states
            # This simulates other drones detecting the failure
            for drone in self.drones:
                if drone.id != drone_id and drone.is_alive():
                    if drone.world_state and failed_drone._my_asset:
                        # Update the failed drone's asset in other drones' world state
                        drone.world_state.upsert_asset(failed_drone._my_asset, from_sync=True)

    def schedule_failure(self, time_seconds: float, drone_id: int) -> None:
        """Schedule a drone failure for testing."""
        self.failure_scheduled = (time_seconds, drone_id)
        print(f"[Simulation] Scheduled failure of drone {drone_id} at {time_seconds}s")

    def get_status(self) -> Dict:
        """Get current status of all drones and zones."""
        return {
            'drones': [
                {
                    'id': d.id,
                    'position': d.get_position(),
                    'status': d.status.value,
                    'photos_taken': len(d.photos),
                    'zones_assigned': len(d.assigned_zones),
                    'is_complete': d.is_complete(),
                    'current_goal': d.get_goal_position(),
                    'intent': d.get_current_intent().intent_type.value if d.get_current_intent() else None
                }
                for d in self.drones
            ],
            'zones': [
                {
                    'id': z.id,
                    'status': z.status.value,
                    'assigned_to': z.assigned_drone_id,
                    'progress': self._calculate_zone_coverage(z)
                }
                for z in self.zones
            ],
            'total_photos': len(self.all_photos),
            'alive_drones': len([d for d in self.drones if d.is_alive()]),
            'completed_zones': len([z for z in self.zones if z.status == ZoneStatus.COMPLETE])
        }

    def _calculate_zone_coverage(self, zone: Zone) -> float:
        """Calculate coverage percentage for a zone from the coverage map."""
        if self.coverage_map is None:
            return zone.coverage_progress

        covered = 0
        total = 0
        for row in self.coverage_map.cells:
            for cell in row:
                if zone.x_min <= cell.x <= zone.x_max and zone.y_min <= cell.y <= zone.y_max:
                    total += 1
                    if cell.covered:
                        covered += 1

        return covered / total if total > 0 else 0.0

    def get_coverage_stats(self) -> Dict:
        """Calculate coverage statistics."""
        total_area = self.region.width * self.region.height
        flyable_area = sum(z.width * z.height for z in self.zones)

        if self.coverage_map is not None:
            map_stats = self.coverage_map.get_coverage_stats()
            covered_cells = map_stats['covered_cells']
            cell_area = self.coverage_map.cell_size ** 2
            covered_area = covered_cells * cell_area
            flyable_coverage = (covered_area / flyable_area) * 100 if flyable_area > 0 else 0
            total_coverage = (covered_area / total_area) * 100 if total_area > 0 else 0

            return {
                'total_area': total_area,
                'flyable_area': flyable_area,
                'covered_area': covered_area,
                'coverage_percentage': total_coverage,
                'flyable_coverage_percentage': flyable_coverage,
                'buffer_area': total_area - flyable_area,
                'total_photos': len(self.all_photos),
                'coverage_map_stats': map_stats
            }
        else:
            covered_area = sum(z.width * z.height * z.coverage_progress for z in self.zones)
            flyable_coverage = (covered_area / flyable_area) * 100 if flyable_area > 0 else 0
            total_coverage = (covered_area / total_area) * 100 if total_area > 0 else 0

            return {
                'total_area': total_area,
                'flyable_area': flyable_area,
                'covered_area': covered_area,
                'coverage_percentage': total_coverage,
                'flyable_coverage_percentage': flyable_coverage,
                'buffer_area': total_area - flyable_area,
                'total_photos': len(self.all_photos)
            }

    def run(
        self,
        max_time: float = 120.0,
        realtime: bool = True,
        show_animation: bool = True
    ) -> dict:
        """Run the simulation."""
        self.is_running = True
        self.simulation_time = 0.0

        if show_animation:
            self.setup_plot()

            def animate(frame):
                if not self.is_running:
                    return

                status = self.step()
                self.update_visualization()

                if status['completed'] or self.simulation_time >= max_time:
                    self.is_running = False
                    self._print_final_stats()

            anim = FuncAnimation(
                self.fig, animate,
                interval=int(config.SIMULATION_TIMESTEP * 1000) if realtime else 1,
                cache_frame_data=False
            )
            plt.show(block=True)
        else:
            while self.is_running and self.simulation_time < max_time:
                status = self.step()
                if status['completed']:
                    self.is_running = False

            print(f"\n[Simulation] Completed at {self.simulation_time:.1f}s")
            self._print_final_stats()

        return {
            'final_time': self.simulation_time,
            'coverage': self.get_coverage_stats(),
            'status': self.get_status()
        }

    def _print_final_stats(self) -> None:
        """Print final simulation statistics."""
        status = self.get_status()

        print("\n" + "=" * 50)
        print("SIMULATION COMPLETE")
        print("=" * 50)
        print("Mode: PATROL/TRACK (Decentralized)")
        print(f"Total time: {self.simulation_time:.1f} seconds")
        print(f"Total photos taken: {len(self.all_photos)}")
        print(f"Active drones: {status['alive_drones']}/{len(self.drones)}")

        # Object tracking stats
        if self.object_manager:
            total_obj = len(self.object_manager.objects)
            detected = len([o for o in self.object_manager.objects if o.detected])
            print(f"Objects tracked: {detected}/{total_obj}")

        if self.mesh_network:
            mesh_stats = self.mesh_network.get_connectivity_stats()
            print(f"Mesh connectivity: {mesh_stats['avg_peers']:.1f} avg peers")

        print("=" * 50)


if __name__ == '__main__':
    sim = Simulation(num_drones=3, debug_mesh=True)
    sim.schedule_failure(5.0, 1)
    sim.run(max_time=60.0, show_animation=False)
