import math
import numpy as np
from collections import namedtuple
from typing import Tuple, Optional, Dict, List, Any
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
import os

# Constants
CAMERA_FORWARD_DIST = 0.066  # Distance from camera to center of rotation
ROBOT_LENGTH = 0.18  # Total robot length

# Named tuple for lane position information
LanePosition0 = namedtuple('LanePosition', 'dist dot_dir angle_deg angle_rad')

class LanePosition(LanePosition0):
    def as_json_dict(self):
        """ Serialization-friendly format. """
        return dict(dist=self.dist,
                   dot_dir=self.dot_dir,
                   angle_deg=self.angle_deg,
                   angle_rad=self.angle_rad)

class NotInLane(Exception):
    ''' Raised when the Duckiebot is not in a lane. '''
    pass

class MapInterpreter:
    def __init__(self, map_dir: str):
        """
        Initialize map interpreter with map data from YAML files
        Args:
            map_dir: Directory containing frames.yaml, tiles.yaml, and tile_maps.yaml
        """
        # Load YAML files
        with open(f"{map_dir}/frames.yaml", 'r') as f:
            self.frames_data = yaml.safe_load(f)
        with open(f"{map_dir}/tiles.yaml", 'r') as f:
            self.tiles_data = yaml.safe_load(f)
        with open(f"{map_dir}/tile_maps.yaml", 'r') as f:
            self.tile_maps_data = yaml.safe_load(f)
            
        # Get tile size
        self.road_tile_size = self.tile_maps_data['tile_maps']['map_0']['tile_size']['x']
        
        # Initialize grid
        self.grid = {}  # Dictionary mapping (i,j) to tile info
        self.drivable_tiles = []
        
        self._process_tiles()
        
    def _process_tiles(self):
        """Process all tiles in the map and create the grid"""
        for tile_name, tile_info in self.tiles_data['tiles'].items():
            # Extract coordinates from tile name (e.g., "map_0/tile_1_2" -> (1,2))
            _, coords = tile_name.split('/')
            i, j = map(int, coords.split('_')[1:])
            
            # Get tile type and pose
            tile_type = tile_info['type']
            pose = self.frames_data['frames'][tile_name]['pose']
            
            # Determine if tile is drivable
            drivable = tile_type in ['straight', 'curve', '3way']
            
            tile_obj = {
                'coords': (i, j),
                'kind': tile_type,
                'angle': pose['yaw'],
                'drivable': drivable,
                'pose': pose
            }
            
            self.grid[(i, j)] = tile_obj
            
            if drivable:
                tile_obj['curves'] = self._get_curve(i, j)
                self.drivable_tiles.append(tile_obj)
    
    def get_tile(self, i: int, j: int) -> Optional[Dict]:
        """Get tile at grid position (i,j)"""
        return self.grid.get((i, j))
    
    def _get_curve(self, i: int, j: int) -> np.ndarray:
        """
        Get the Bezier curve control points for a given tile
        Returns array of control points for each possible curve
        """
        tile = self.get_tile(i, j)
        assert tile is not None
        
        kind = tile['kind']
        angle = tile['angle']
        
        # Define control points for different tile types
        if kind == 'straight':
            pts = np.array([
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, -0.25],
                    [-0.20, 0, 0.25],
                    [-0.20, 0, 0.50],
                ],
                [
                    [0.20, 0, 0.50],
                    [0.20, 0, 0.25],
                    [0.20, 0, -0.25],
                    [0.20, 0, -0.50],
                ]
            ]) * self.road_tile_size
            
        elif kind == 'curve':
            pts = np.array([
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, 0.00],
                    [0.00, 0, 0.20],
                    [0.50, 0, 0.20],
                ],
                [
                    [0.50, 0, -0.20],
                    [0.30, 0, -0.20],
                    [0.20, 0, -0.30],
                    [0.20, 0, -0.50],
                ]
            ]) * self.road_tile_size
            
        elif kind == '3way':
            pts = np.array([
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, -0.25],
                    [-0.20, 0, 0.25],
                    [-0.20, 0, 0.50],
                ],
                [
                    [0.20, 0, 0.50],
                    [0.20, 0, 0.25],
                    [0.20, 0, -0.25],
                    [0.20, 0, -0.50],
                ],
                [
                    [0.50, 0, 0.20],
                    [0.25, 0, 0.20],
                    [-0.25, 0, 0.20],
                    [-0.50, 0, 0.20],
                ]
            ]) * self.road_tile_size
            
        # Rotate and align each curve with its place in global frame
        mat = gen_rot_matrix(np.array([0, 1, 0]), angle)
        pts = np.matmul(pts, mat)
        
        # Add tile position offset
        pose = tile['pose']
        pts += np.array([pose['x'] * self.road_tile_size, 0, pose['y'] * self.road_tile_size])
        
        return pts

    def get_reference_trajectory(self) -> List[np.ndarray]:
        """
        Get the reference trajectory as a list of points along a continuous Bezier curve.
        The curve is created by connecting the control points from each tile.
        Returns a list of 3D points (x, y, z) representing the center line of the road.
        """
        # Find all drivable tiles
        drivable_tiles = [t for t in self.drivable_tiles if t['drivable']]
        if not drivable_tiles:
            return []
            
        # Start with a straight tile if possible
        start_tile = next((t for t in drivable_tiles if t['kind'] == 'straight'), drivable_tiles[0])
        
        # Collect all control points for the Bezier curve
        control_points = []
        visited = set()
        current_tile = start_tile
        current_angle = current_tile['angle']
        
        while current_tile and current_tile['coords'] not in visited:
            visited.add(current_tile['coords'])
            
            if current_tile['kind'] == 'straight':
                # For straight tiles, create control points for a straight line
                pose = current_tile['pose']
                center = np.array([
                    pose['x'] * self.road_tile_size,
                    0,
                    pose['y'] * self.road_tile_size
                ])
                dir_vec = get_dir_vec(current_angle)
                
                # Add control points for straight section
                control_points.extend([
                    center - 0.25 * self.road_tile_size * dir_vec,
                    center,
                    center + 0.25 * self.road_tile_size * dir_vec
                ])
                
                # Find next tile
                next_coords = self._get_next_tile_coords(current_tile['coords'], current_angle)
                next_tile = self.get_tile(*next_coords)
                if next_tile and next_tile['drivable']:
                    current_tile = next_tile
                    current_angle = current_tile['angle']
                else:
                    break
                    
            elif current_tile['kind'] == 'curve':
                # For curve tiles, use the Bezier control points directly
                curves = current_tile['curves']
                # Find the curve that matches our current direction
                dir_vec = get_dir_vec(current_angle)
                curve_headings = curves[:, -1, :] - curves[:, 0, :]
                curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
                dot_prods = np.dot(curve_headings, dir_vec)
                curve = curves[np.argmax(dot_prods)]
                
                # Add control points for curve
                control_points.extend(curve)
                
                # Update angle based on curve end point
                end_tangent = bezier_tangent(curve, 1.0)
                end_tangent = end_tangent / np.linalg.norm(end_tangent)
                current_angle = math.atan2(-end_tangent[2], end_tangent[0])
                
                # Find next tile
                end_point = bezier_point(curve, 1.0)
                next_coords = self._get_next_tile_coords_from_point(end_point, current_angle)
                next_tile = self.get_tile(*next_coords)
                if next_tile and next_tile['drivable']:
                    current_tile = next_tile
                else:
                    break
                    
            elif current_tile['kind'] == '3way':
                # For 3way intersections, add control points for each possible path
                pose = current_tile['pose']
                center = np.array([
                    pose['x'] * self.road_tile_size,
                    0,
                    pose['y'] * self.road_tile_size
                ])
                
                # Add center point
                control_points.append(center)
                
                # Try to continue in the current direction first
                next_coords = self._get_next_tile_coords(current_tile['coords'], current_angle)
                next_tile = self.get_tile(*next_coords)
                
                if next_tile and next_tile['drivable']:
                    current_tile = next_tile
                    current_angle = current_tile['angle']
                else:
                    # Try right turn
                    next_coords = self._get_next_tile_coords(current_tile['coords'], current_angle + np.pi/2)
                    next_tile = self.get_tile(*next_coords)
                    if next_tile and next_tile['drivable']:
                        current_tile = next_tile
                        current_angle = current_tile['angle']
                    else:
                        # Try left turn
                        next_coords = self._get_next_tile_coords(current_tile['coords'], current_angle - np.pi/2)
                        next_tile = self.get_tile(*next_coords)
                        if next_tile and next_tile['drivable']:
                            current_tile = next_tile
                            current_angle = current_tile['angle']
                        else:
                            break
        
        # Create a smooth trajectory by sampling points along the Bezier curve
        trajectory = []
        if len(control_points) >= 4:  # Need at least 4 points for a cubic Bezier curve
            # Convert control points to numpy array and reshape to (N, 3)
            control_points = np.array(control_points).reshape(-1, 3)
            
            # Create segments of cubic Bezier curves
            for i in range(0, len(control_points) - 3, 3):
                segment = control_points[i:i+4]
                # Sample points along this segment
                for t in np.linspace(0, 1, 20):  # 20 points per segment for smoothness
                    point = bezier_point(segment, t)
                    trajectory.append(point)
            
            # Convert trajectory to numpy array and reshape to (N, 3)
            trajectory = np.array(trajectory).reshape(-1, 3)
        
        return trajectory

    def _get_next_tile_coords(self, current_coords: Tuple[int, int], angle: float) -> Tuple[int, int]:
        """Get the coordinates of the next tile based on current position and angle."""
        i, j = current_coords
        # Convert angle to cardinal direction
        angle = angle % (2 * np.pi)
        if angle < np.pi/4 or angle >= 7*np.pi/4:  # East
            return (i + 1, j)
        elif angle < 3*np.pi/4:  # North
            return (i, j + 1)
        elif angle < 5*np.pi/4:  # West
            return (i - 1, j)
        else:  # South
            return (i, j - 1)

    def _get_next_tile_coords_from_point(self, point: np.ndarray, angle: float) -> Tuple[int, int]:
        """Get the coordinates of the next tile based on a point and angle."""
        i = math.floor(point[0] / self.road_tile_size)
        j = math.floor(point[2] / self.road_tile_size)
        return self._get_next_tile_coords((i, j), angle)

    def get_reference_trajectory_with_tangents(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the reference trajectory with tangent vectors at each point.
        Returns a list of tuples (point, tangent) where:
        - point is a 3D point (x, y, z) on the trajectory
        - tangent is the normalized tangent vector at that point
        """
        trajectory = self.get_reference_trajectory()
        trajectory_with_tangents = []
        
        for i in range(len(trajectory)):
            point = trajectory[i]
            
            # Calculate tangent by looking at next point
            if i < len(trajectory) - 1:
                next_point = trajectory[i + 1]
                tangent = next_point - point
                tangent = tangent / np.linalg.norm(tangent)
            else:
                # For last point, use previous tangent
                prev_point = trajectory[i - 1]
                tangent = point - prev_point
                tangent = tangent / np.linalg.norm(tangent)
            
            trajectory_with_tangents.append((point, tangent))
        
        return trajectory_with_tangents

    def plot_trajectory_and_position(self, pos: np.ndarray, angle: float, lane_pos: Optional[LanePosition] = None,
                                   output_dir: str = "plots", filename: str = "trajectory_position.png"):
        """
        Plot the reference trajectory and current position with relevant information and save to file.
        
        Args:
            pos: Current position (x, y, z)
            angle: Current heading angle in radians
            lane_pos: Optional LanePosition object containing distance and angle information
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        # Get reference trajectory
        trajectory = self.get_reference_trajectory()
        trajectory_points = np.array(trajectory)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Plot reference trajectory
        plt.plot(trajectory_points[:, 0], trajectory_points[:, 2], 'b-', label='Reference Trajectory', alpha=0.5)
        
        # Plot current position
        plt.plot(pos[0], pos[2], 'ro', label='Current Position', markersize=10)
        
        # Plot heading direction
        dir_vec = get_dir_vec(angle)
        arrow_length = 0.2
        plt.arrow(pos[0], pos[2], 
                 dir_vec[0] * arrow_length, dir_vec[2] * arrow_length,
                 head_width=0.05, head_length=0.1, fc='r', ec='r',
                 label='Heading')
        
        # If lane position is provided, add distance and angle information
        if lane_pos is not None:
            # Find closest point on trajectory
            closest_point, closest_tangent = self.closest_curve_point(pos, angle)
            if closest_point is not None:
                # Plot line to closest point
                plt.plot([pos[0], closest_point[0]], [pos[2], closest_point[2]], 'g--', 
                        label=f'Distance: {lane_pos.dist:.2f}m')
                
                # Plot angle
                angle_arrow_length = 0.15
                plt.arrow(closest_point[0], closest_point[2],
                         closest_tangent[0] * angle_arrow_length, closest_tangent[2] * angle_arrow_length,
                         head_width=0.05, head_length=0.1, fc='g', ec='g',
                         label=f'Angle: {lane_pos.angle_deg:.1f}°')
        
        # Add legend and labels
        plt.legend()
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('Map Trajectory and Position')
        plt.grid(True)
        plt.axis('equal')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def plot_trajectory_with_tangents(self, output_dir: str = "plots", filename: str = "trajectory_tangents.png"):
        """
        Plot the reference trajectory with tangent vectors at each point and save to file.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        trajectory_with_tangents = self.get_reference_trajectory_with_tangents()
        
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Extract points and tangents
        points = np.array([p for p, _ in trajectory_with_tangents])
        tangents = np.array([t for _, t in trajectory_with_tangents])
        
        # Plot trajectory
        plt.plot(points[:, 0], points[:, 2], 'b-', label='Reference Trajectory')
        
        # Plot tangent vectors
        arrow_length = 0.1
        for point, tangent in zip(points, tangents):
            plt.arrow(point[0], point[2],
                     tangent[0] * arrow_length, tangent[2] * arrow_length,
                     head_width=0.02, head_length=0.05, fc='g', ec='g')
        
        # Add legend and labels
        plt.legend()
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('Reference Trajectory with Tangents')
        plt.grid(True)
        plt.axis('equal')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def plot_tile_centers(self, output_dir: str = "plots", filename: str = "tile_centers.png"):
        """
        Plot the centers of all tiles in 2D coordinates.
        Drivable tiles are colored green, non-drivable tiles are colored red.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Collect tile centers, separating drivable and non-drivable tiles
        drivable_centers = []
        non_drivable_centers = []
        
        for tile in self.grid.values():
            pose = tile['pose']
            center = np.array([
                pose['x'] * self.road_tile_size,
                pose['y'] * self.road_tile_size
            ])
            
            if tile['drivable']:
                drivable_centers.append(center)
            else:
                non_drivable_centers.append(center)
        
        # Convert to numpy arrays
        drivable_centers = np.array(drivable_centers) if drivable_centers else np.array([])
        non_drivable_centers = np.array(non_drivable_centers) if non_drivable_centers else np.array([])
        
        # Plot centers with different colors
        if len(drivable_centers) > 0:
            plt.plot(drivable_centers[:, 0], drivable_centers[:, 1], 'go', 
                    label='Drivable Tiles', markersize=8)
        if len(non_drivable_centers) > 0:
            plt.plot(non_drivable_centers[:, 0], non_drivable_centers[:, 1], 'ro', 
                    label='Non-drivable Tiles', markersize=8)
        
        # Add grid lines
        plt.grid(True)
        
        # Add labels and title
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Tile Centers (Green: Drivable, Red: Non-drivable)')
        plt.legend()
        plt.axis('equal')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def plot_track(self, output_dir: str = "plots", filename: str = "track.png"):
        """
        Plot a complete track by connecting drivable tiles with straight lines and Bezier curves.
        The track follows the center of drivable tiles, using straight lines for straight tiles
        and Bezier curves for curved sections.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # First plot all tile centers for reference
        drivable_centers = []
        non_drivable_centers = []
        
        for tile in self.grid.values():
            pose = tile['pose']
            center = np.array([
                pose['x'] * self.road_tile_size,
                pose['y'] * self.road_tile_size
            ])
            
            if tile['drivable']:
                drivable_centers.append(center)
            else:
                non_drivable_centers.append(center)
        
        # Convert to numpy arrays
        drivable_centers = np.array(drivable_centers) if drivable_centers else np.array([])
        non_drivable_centers = np.array(non_drivable_centers) if non_drivable_centers else np.array([])
        
        # Plot centers with different colors
        if len(drivable_centers) > 0:
            plt.plot(drivable_centers[:, 0], drivable_centers[:, 1], 'go', 
                    label='Drivable Tiles', markersize=8, alpha=0.3)
        if len(non_drivable_centers) > 0:
            plt.plot(non_drivable_centers[:, 0], non_drivable_centers[:, 1], 'ro', 
                    label='Non-drivable Tiles', markersize=8, alpha=0.3)
        
        # Find a starting point (preferably a straight tile)
        start_tile = next((t for t in self.drivable_tiles if t['kind'] == 'straight'), 
                         self.drivable_tiles[0] if self.drivable_tiles else None)
        
        if start_tile is None:
            return
        
        # Track building
        visited = set()
        current_tile = start_tile
        current_angle = current_tile['angle']
        track_points = []
        
        while current_tile and current_tile['coords'] not in visited:
            visited.add(current_tile['coords'])
            pose = current_tile['pose']
            center = np.array([
                pose['x'] * self.road_tile_size,
                pose['y'] * self.road_tile_size
            ])
            
            if current_tile['kind'] == 'straight':
                # For straight tiles, add points along the center line
                dir_vec = get_dir_vec(current_angle)
                start_point = center - 0.25 * self.road_tile_size * dir_vec
                end_point = center + 0.25 * self.road_tile_size * dir_vec
                track_points.extend([start_point, end_point])
                
                # Find next tile
                next_coords = self._get_next_tile_coords(current_tile['coords'], current_angle)
                next_tile = self.get_tile(*next_coords)
                if next_tile and next_tile['drivable']:
                    current_tile = next_tile
                    current_angle = current_tile['angle']
                else:
                    break
                    
            elif current_tile['kind'] == 'curve':
                # For curve tiles, use Bezier curve control points
                curves = current_tile['curves']
                dir_vec = get_dir_vec(current_angle)
                curve_headings = curves[:, -1, :] - curves[:, 0, :]
                curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
                dot_prods = np.dot(curve_headings, dir_vec)
                curve = curves[np.argmax(dot_prods)]
                
                # Sample points along the Bezier curve
                for t in np.linspace(0, 1, 20):
                    point = bezier_point(curve, t)
                    track_points.append(point[:2])  # Only use x,y coordinates
                
                # Update angle based on curve end point
                end_tangent = bezier_tangent(curve, 1.0)
                end_tangent = end_tangent / np.linalg.norm(end_tangent)
                current_angle = math.atan2(-end_tangent[2], end_tangent[0])
                
                # Find next tile
                end_point = bezier_point(curve, 1.0)
                next_coords = self._get_next_tile_coords_from_point(end_point, current_angle)
                next_tile = self.get_tile(*next_coords)
                if next_tile and next_tile['drivable']:
                    current_tile = next_tile
                else:
                    break
                    
            elif current_tile['kind'] == '3way':
                # For 3way intersections, try to continue in current direction
                track_points.append(center[:2])  # Only use x,y coordinates
                
                # Try to continue in the current direction first
                next_coords = self._get_next_tile_coords(current_tile['coords'], current_angle)
                next_tile = self.get_tile(*next_coords)
                
                if next_tile and next_tile['drivable']:
                    current_tile = next_tile
                    current_angle = current_tile['angle']
                else:
                    # Try right turn
                    next_coords = self._get_next_tile_coords(current_tile['coords'], current_angle + np.pi/2)
                    next_tile = self.get_tile(*next_coords)
                    if next_tile and next_tile['drivable']:
                        current_tile = next_tile
                        current_angle = current_tile['angle']
                    else:
                        # Try left turn
                        next_coords = self._get_next_tile_coords(current_tile['coords'], current_angle - np.pi/2)
                        next_tile = self.get_tile(*next_coords)
                        if next_tile and next_tile['drivable']:
                            current_tile = next_tile
                            current_angle = current_tile['angle']
                        else:
                            break
        
        # Convert track points to numpy array and plot
        if track_points:
            track_points = np.array(track_points)
            plt.plot(track_points[:, 0], track_points[:, 1], 'b-', 
                    label='Track', linewidth=2)
        
        # Add grid lines
        plt.grid(True)
        
        # Add labels and title
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Complete Track')
        plt.legend()
        plt.axis('equal')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def plot_robot_on_tiles(self, pos: np.ndarray, angle: float, 
                          output_dir: str = "plots", filename: str = "robot_position.png"):
        """
        Plot the robot's position and angle on the tile centers map.
        
        Args:
            pos: Robot position (x, y, z) in meters
            angle: Robot heading angle in radians
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Collect tile centers, separating drivable and non-drivable tiles
        drivable_centers = []
        non_drivable_centers = []
        
        for tile in self.grid.values():
            pose = tile['pose']
            center = np.array([
                pose['x'] * self.road_tile_size,
                pose['y'] * self.road_tile_size
            ])
            
            if tile['drivable']:
                drivable_centers.append(center)
            else:
                non_drivable_centers.append(center)
        
        # Convert to numpy arrays
        drivable_centers = np.array(drivable_centers) if drivable_centers else np.array([])
        non_drivable_centers = np.array(non_drivable_centers) if non_drivable_centers else np.array([])
        
        # Plot centers with different colors
        if len(drivable_centers) > 0:
            plt.plot(drivable_centers[:, 0], drivable_centers[:, 1], 'go', 
                    label='Drivable Tiles', markersize=8, alpha=0.3)
        if len(non_drivable_centers) > 0:
            plt.plot(non_drivable_centers[:, 0], non_drivable_centers[:, 1], 'ro', 
                    label='Non-drivable Tiles', markersize=8, alpha=0.3)
        
        # Plot robot position
        plt.plot(pos[0], pos[2], 'bo', label='Robot Position', markersize=10)
        
        # Plot robot heading
        dir_vec = get_dir_vec(angle)
        arrow_length = 0.2
        plt.arrow(pos[0], pos[2], 
                 dir_vec[0] * arrow_length, dir_vec[2] * arrow_length,
                 head_width=0.05, head_length=0.1, fc='b', ec='b',
                 label='Robot Heading')
        
        # Add grid lines
        plt.grid(True)
        
        # Add labels and title
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Robot Position on Map')
        plt.legend()
        plt.axis('equal')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

def gen_rot_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Generate rotation matrix for rotation around axis by angle"""
    # This is a simplified version - in practice you'd want a more robust implementation
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])

def get_dir_vec(angle):
    """
    Vector pointing in the direction the agent is looking
    """
    x = math.cos(angle)
    z = -math.sin(angle)
    return np.array([x, 0, z])

def get_right_vec(angle):
    """
    Vector pointing to the right of the agent
    """
    x = math.sin(angle)
    z = math.cos(angle)
    return np.array([x, 0, z])

def _actual_center(pos, angle):
    """
    Calculate the position of the geometric center of the agent
    The value of pos is the center of rotation.
    """
    dir_vec = get_dir_vec(angle)
    return pos + (CAMERA_FORWARD_DIST - (0.18 / 2)) * dir_vec  # 0.18 is ROBOT_LENGTH

class LanePositionCalculator:
    def __init__(self, map_interpreter: MapInterpreter):
        self.map_interpreter = map_interpreter
        self.road_tile_size = map_interpreter.road_tile_size

    def get_grid_coords(self, abs_pos: np.ndarray) -> Tuple[int, int]:
        """
        Compute the tile indices (i,j) for a given (x,_,z) world position
        x-axis maps to increasing i indices
        z-axis maps to increasing j indices
        """
        x, _, z = abs_pos
        i = math.floor(x / self.road_tile_size)
        j = math.floor(z / self.road_tile_size)
        return int(i), int(j)

    def closest_curve_point(self, pos: np.ndarray, angle: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the closest point on the curve to a given point
        Also returns the tangent at that point.
        Returns None, None if not in a lane.
        """
        i, j = self.get_grid_coords(pos)
        tile = self.map_interpreter.get_tile(i, j)
        
        if tile is None or not tile['drivable']:
            return None, None

        # Find curve with largest dotproduct with heading
        curves = tile['curves']
        curve_headings = curves[:, -1, :] - curves[:, 0, :]
        curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
        dir_vec = get_dir_vec(angle)

        dot_prods = np.dot(curve_headings, dir_vec)

        # Closest curve = one with largest dotprod
        cps = curves[np.argmax(dot_prods)]

        # Find closest point and tangent to this curve
        t = bezier_closest(cps, pos)
        point = bezier_point(cps, t)
        tangent = bezier_tangent(cps, t)

        return point, tangent

    def get_lane_pos2(self, pos: np.ndarray, angle: float) -> LanePosition:
        """
        Get the position of the agent relative to the center of the right lane
        Raises NotInLane if the Duckiebot is not in a lane.
        """
        # Get the closest point along the right lane's Bezier curve,
        # and the tangent at that point
        point, tangent = self.closest_curve_point(pos, angle)
        if point is None:
            msg = 'Point not in lane: %s' % pos
            raise NotInLane(msg)

        assert point is not None

        # Compute the alignment of the agent direction with the curve tangent
        dirVec = get_dir_vec(angle)
        dotDir = np.dot(dirVec, tangent)
        dotDir = max(-1, min(1, dotDir))

        # Compute the signed distance to the curve
        # Right of the curve is negative, left is positive
        posVec = pos - point
        upVec = np.array([0, 1, 0])
        rightVec = np.cross(tangent, upVec)
        signedDist = np.dot(posVec, rightVec)

        # Compute the signed angle between the direction and curve tangent
        # Right of the tangent is negative, left is positive
        angle_rad = math.acos(dotDir)

        if np.dot(dirVec, rightVec) < 0:
            angle_rad *= -1

        angle_deg = np.rad2deg(angle_rad)

        return LanePosition(dist=signedDist, dot_dir=dotDir, angle_deg=angle_deg,
                          angle_rad=angle_rad)

    def plot_trajectory_and_position(self, pos: np.ndarray, angle: float, lane_pos: Optional[LanePosition] = None,
                                   output_dir: str = "plots", filename: str = "trajectory_position.png"):
        """
        Plot the reference trajectory and current position with relevant information and save to file.
        
        Args:
            pos: Current position (x, y, z)
            angle: Current heading angle in radians
            lane_pos: Optional LanePosition object containing distance and angle information
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        # Get reference trajectory
        trajectory = self.map_interpreter.get_reference_trajectory()
        if isinstance(trajectory, list):
            trajectory = np.array(trajectory)
        if len(trajectory.shape) == 1:
            trajectory = trajectory.reshape(-1, 3)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Plot reference trajectory
        plt.plot(trajectory[:, 0], trajectory[:, 2], 'b-', label='Reference Trajectory', alpha=0.5)
        
        # Plot current position
        plt.plot(pos[0], pos[2], 'ro', label='Current Position', markersize=10)
        
        # Plot heading direction
        dir_vec = get_dir_vec(angle)
        arrow_length = 0.2
        plt.arrow(pos[0], pos[2], 
                 dir_vec[0] * arrow_length, dir_vec[2] * arrow_length,
                 head_width=0.05, head_length=0.1, fc='r', ec='r',
                 label='Heading')
        
        # If lane position is provided, add distance and angle information
        if lane_pos is not None:
            # Find closest point on trajectory
            closest_point, closest_tangent = self.closest_curve_point(pos, angle)
            if closest_point is not None:
                # Plot line to closest point
                plt.plot([pos[0], closest_point[0]], [pos[2], closest_point[2]], 'g--', 
                        label=f'Distance: {lane_pos.dist:.2f}m')
                
                # Plot angle
                angle_arrow_length = 0.15
                plt.arrow(closest_point[0], closest_point[2],
                         closest_tangent[0] * angle_arrow_length, closest_tangent[2] * angle_arrow_length,
                         head_width=0.05, head_length=0.1, fc='g', ec='g',
                         label=f'Angle: {lane_pos.angle_deg:.1f}°')
        
        # Add legend and labels
        plt.legend()
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('Map Trajectory and Position')
        plt.grid(True)
        plt.axis('equal')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def plot_trajectory_with_tangents(self, output_dir: str = "plots", filename: str = "trajectory_tangents.png"):
        """
        Plot the reference trajectory with tangent vectors at each point and save to file.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        trajectory = self.map_interpreter.get_reference_trajectory()
        if isinstance(trajectory, list):
            trajectory = np.array(trajectory)
        if len(trajectory.shape) == 1:
            trajectory = trajectory.reshape(-1, 3)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Plot trajectory
        plt.plot(trajectory[:, 0], trajectory[:, 2], 'b-', label='Reference Trajectory')
        
        # Calculate and plot tangent vectors
        for i in range(len(trajectory) - 1):
            point = trajectory[i]
            next_point = trajectory[i + 1]
            tangent = next_point - point
            tangent = tangent / np.linalg.norm(tangent)
            
            # Plot tangent vector
            arrow_length = 0.1
            plt.arrow(point[0], point[2],
                     tangent[0] * arrow_length, tangent[2] * arrow_length,
                     head_width=0.02, head_length=0.05, fc='g', ec='g')
        
        # Add legend and labels
        plt.legend()
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('Reference Trajectory with Tangents')
        plt.grid(True)
        plt.axis('equal')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

# Bezier curve utility functions
def bezier_closest(control_points: np.ndarray, point: np.ndarray) -> float:
    """
    Find the parameter t that gives the closest point on the Bezier curve to the given point
    """
    # This is a simplified version - in practice you'd want a more robust implementation
    # that handles edge cases and uses numerical optimization
    t = 0.5  # Start with middle point
    for _ in range(10):  # 10 iterations of Newton's method
        p = bezier_point(control_points, t)
        tangent = bezier_tangent(control_points, t)
        error = p - point
        t = t - np.dot(error, tangent) / np.dot(tangent, tangent)
        t = max(0, min(1, t))  # Clamp to [0,1]
    return t

def bezier_point(control_points: np.ndarray, t: float) -> np.ndarray:
    """
    Calculate point on Bezier curve at parameter t
    """
    n = len(control_points) - 1
    point = np.zeros(3)
    for i, p in enumerate(control_points):
        point += p * math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    return point

def bezier_tangent(control_points: np.ndarray, t: float) -> np.ndarray:
    """
    Calculate tangent vector on Bezier curve at parameter t
    """
    n = len(control_points) - 1
    tangent = np.zeros(3)
    for i in range(n):
        p = control_points[i + 1] - control_points[i]
        tangent += p * math.comb(n - 1, i) * (t ** i) * ((1 - t) ** (n - 1 - i))
    return tangent * n 