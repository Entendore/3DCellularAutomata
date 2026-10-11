"""
3D Cellular Automata visualization in PySide6.
Ultimate Edition with hidden-face culling, hologram mode, and cinematic camera.

Controls:
  Left Drag   rotate camera        Right Drag  pan camera
  Middle Click inject life         Wheel       zoom
  W/S         pitch camera         A/D         yaw camera
  Q/E         zoom in/out          Arrows      pan camera
  Space       pause / resume       R           reset pattern
  N           next rule            C           next palette
  X           toggle symmetry      P           export PNG
  Z           toggle auto-zoom     V           toggle hologram mode
  B           toggle cinematic cam F1          debug overlay
  F2          slow-mo toggle       M           reduced motion
  Esc         quit
"""
import sys
import math
import numpy as np
from PySide6.QtCore import Qt, QTimer, QPointF, QElapsedTimer
from PySide6.QtGui import (QPainter, QColor, QPen, QPolygonF, QLinearGradient,
                           QRadialGradient, QFont, QPixmap)
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QSlider, QLabel,
                               QFrame)

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# CA configuration & Rules
# ---------------------------------------------------------------------------
GRID = 16

RULES = [
    ("B4-5/S4-5", {4, 5}, {4, 5}),    
    ("B5-7/S6-7", {5, 6, 7}, {6, 7}), 
    ("B6/S5-6",   {6}, {5, 6}),
    ("B2/S3-4",   {2}, {3, 4}),       # Chaotic explosion
    ("B3-5/S4-5", {3, 4, 5}, {4, 5})  # Stable clouds
]

# ---------------------------------------------------------------------------
# Color Palettes
# ---------------------------------------------------------------------------
PALETTES = [
    {
        "name": "Neon Cyber",
        "bg_top": (18, 22, 38), "bg_bot": (6, 8, 16),
        "young": (60, 255, 230), "old": (255, 60, 180),
        "particle": (255, 230, 100),
        "bound": (50, 80, 120)
    },
    {
        "name": "Inferno",
        "bg_top": (30, 10, 5), "bg_bot": (8, 0, 0),
        "young": (255, 240, 100), "old": (180, 20, 0),
        "particle": (255, 100, 0),
        "bound": (100, 40, 20)
    },
    {
        "name": "Bio Forest",
        "bg_top": (10, 20, 15), "bg_bot": (2, 6, 4),
        "young": (120, 255, 100), "old": (20, 100, 40),
        "particle": (200, 255, 150),
        "bound": (30, 60, 40)
    },
    {
        "name": "Monochrome",
        "bg_top": (15, 15, 15), "bg_bot": (0, 0, 0),
        "young": (230, 230, 230), "old": (80, 80, 80),
        "particle": (255, 150, 0), 
        "bound": (60, 60, 60)
    }
]

# ---------------------------------------------------------------------------
# Cube geometry & Bounding Box
# ---------------------------------------------------------------------------
CUBE_VERTS = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
], dtype=float) * 0.5

FACES = [
    ([0, 1, 5, 4], np.array([ 0, -1,  0], dtype=float)),
    ([3, 2, 6, 7], np.array([ 0,  1,  0], dtype=float)),
    ([0, 3, 2, 1], np.array([ 0,  0, -1], dtype=float)),
    ([4, 5, 6, 7], np.array([ 0,  0,  1], dtype=float)),
    ([0, 4, 7, 3], np.array([-1,  0,  0], dtype=float)),
    ([1, 2, 6, 5], np.array([ 1,  0,  0], dtype=float)),
]
FACE_NORMALS = np.array([f[1] for f in FACES])
FACES_VERTS_IDX = np.array([f[0] for f in FACES])

# Offsets to check neighbor for each face
FACE_NEIGHBOR_OFFSETS = [
    (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0)
]

BOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0), 
    (4,5),(5,6),(6,7),(7,4), 
    (0,4),(1,5),(2,6),(3,7)  
]
BOX_SCALE = GRID / 2.0

# ---------------------------------------------------------------------------
# Math helpers & Pure CA model
# ---------------------------------------------------------------------------
def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3

def rotate_points(points: np.ndarray, ax: float, ay: float) -> np.ndarray:
    cos_ay, sin_ay = math.cos(ay), math.sin(ay)
    cos_ax, sin_ax = math.cos(ax), math.sin(ax)
    x = points[:, 0] * cos_ay - points[:, 2] * sin_ay
    z = points[:, 0] * sin_ay + points[:, 2] * cos_ay
    y = points[:, 1]
    y2 = y * cos_ax - z * sin_ax
    z2 = y * sin_ax + z * cos_ax
    return np.column_stack([x, y2, z2])

def project_points(points_3d: np.ndarray, cx: float, cy: float, 
                   focal: float = 900.0, z_offset: float = 600.0):
    factor = focal / (focal - points_3d[:, 2] + z_offset)
    px = points_3d[:, 0] * factor + cx
    py = -points_3d[:, 1] * factor + cy
    return np.column_stack([px, py]), points_3d[:, 2]

def count_neighbors(grid: np.ndarray) -> np.ndarray:
    N = grid.shape[0]
    padded = np.zeros((N + 2, N + 2, N + 2), dtype=np.int16)
    padded[1:-1, 1:-1, 1:-1] = grid.astype(np.int16)
    count = np.zeros_like(grid, dtype=np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                count += padded[1 + dx:1 + dx + N,
                                1 + dy:1 + dy + N,
                                1 + dz:1 + dz + N]
    return count


# ---------------------------------------------------------------------------
# View + Controller
# ---------------------------------------------------------------------------
class CAView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 640)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.StrongFocus)

        # CA state
        self.grid = np.zeros((GRID, GRID, GRID), dtype=bool)
        self.age = np.zeros((GRID, GRID, GRID), dtype=np.int32)
        self.born_at = np.full((GRID, GRID, GRID), -1.0, dtype=np.float32)
        self.dying = []
        self.particles = []          
        self.generation = 0
        self.population = 0
        self.rule_index = 0
        self.palette_index = 0
        self.symmetry = False
        self.flash = 0.0
        self.density = 25
        self.hologram = False
        self.cinematic = False

        # Camera
        self.ax = 0.55
        self.ay = 0.6
        self.target_ax = self.ax
        self.target_ay = self.ay
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.target_pan_x = 0.0
        self.target_pan_y = 0.0
        self.auto_rotate = True
        self.auto_zoom = False
        
        self.dragging = False
        self.panning = False
        self.last_pos = None
        self.keys = set()

        # Juice & Effects
        self.pulse = 0.0
        self.shake = 0.0
        self.shake_offset = (0.0, 0.0)
        self.last_pop = 0
        self.scan_z = 0.0             
        self.light_angle = 0.0        
        self.slice_z = GRID

        # Timing
        self.elapsed = QElapsedTimer()
        self.elapsed.start()
        self.last_elapsed = 0
        self.now = 0.0
        self.step_timer = 0.0
        self.step_interval = 0.18

        # Flags
        self.reduced_motion = False
        self.show_debug = False
        self.paused = False
        self.time_scale = 1.0

        self.fps = 0.0
        self._frame_count = 0
        self._fps_timer = 0.0

        # Ground Grid Generation
        grid_lines = []
        for i in range(-GRID//2, GRID//2 + 1):
            grid_lines.append([[i, -GRID/2.0, -GRID/2.0], [i, -GRID/2.0, GRID/2.0]])
            grid_lines.append([[-GRID/2.0, -GRID/2.0, i], [GRID/2.0, -GRID/2.0, i]])
        self.grid_lines_3d = np.array(grid_lines)

        # Fake Shadow Generation
        shadow_pts = []
        for a in np.linspace(0, 2*math.pi, 20):
            shadow_pts.append([math.cos(a) * (GRID/2.5), -GRID/2.0 + 0.01, math.sin(a) * (GRID/2.5)])
        self.shadow_3d = np.array(shadow_pts)

        self.init_pattern()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def get_current_rule(self):
        return RULES[self.rule_index]

    def get_current_palette(self):
        return PALETTES[self.palette_index]

    def init_pattern(self):
        self.grid[:] = False
        self.age[:] = 0
        self.born_at[:] = -1.0
        self.dying.clear()
        self.particles.clear()
        
        center = GRID // 2
        size = max(2, GRID // 4)
        
        c_x, c_y, c_z = center, center, center
        n_cells = int((self.density / 100.0) * (size + 1) ** 3)
        
        for _ in range(n_cells):
            x = np.random.randint(c_x, c_x + size + 1)
            y = np.random.randint(c_y, c_y + size + 1)
            z = np.random.randint(c_z, c_z + size + 1)
            
            if self.symmetry:
                for i in (-1, 1):
                    for j in (-1, 1):
                        for k in (-1, 1):
                            gx = c_x + i*(x-c_x)
                            gy = c_y + j*(y-c_y)
                            gz = c_z + k*(z-c_z)
                            if 0 <= gx < GRID and 0 <= gy < GRID and 0 <= gz < GRID:
                                self.grid[gx, gy, gz] = True
                                self.age[gx, gy, gz] = 1
                                self.born_at[gx, gy, gz] = self.now
            else:
                if 0 <= x < GRID and 0 <= y < GRID and 0 <= z < GRID:
                    self.grid[x, y, z] = True
                    self.age[x, y, z] = 1
                    self.born_at[x, y, z] = self.now

        self.generation = 0
        self.population = int(self.grid.sum())
        self.last_pop = self.population
        self.pulse = 0.6

    def inject_life_at_center(self):
        c = GRID // 2
        for x in range(c-1, c+2):
            for y in range(c-1, c+2):
                for z in range(c-1, c+2):
                    if not self.grid[x, y, z]:
                        self.grid[x, y, z] = True
                        self.age[x, y, z] = 1
                        self.born_at[x, y, z] = self.now
        self.pulse = 1.0
        self.shake = 0.5

    def next_rule(self):
        self.rule_index = (self.rule_index + 1) % len(RULES)
        self.init_pattern()

    def next_palette(self):
        self.palette_index = (self.palette_index + 1) % len(PALETTES)

    def toggle_symmetry(self):
        self.symmetry = not self.symmetry
        self.init_pattern()

    def toggle_auto_zoom(self):
        self.auto_zoom = not self.auto_zoom

    def toggle_hologram(self):
        self.hologram = not self.hologram

    def toggle_cinematic(self):
        self.cinematic = not self.cinematic
        if self.cinematic:
            self.auto_rotate = False
            self.auto_zoom = False

    def export_frame(self):
        try:
            pixmap = self.grab()
            filename = f"ca_frame_{int(self.now*1000)}.png"
            pixmap.save(filename)
            self.flash = 1.0
        except Exception as e:
            print(f"Export failed: {e}")

    # ----- main loop -----
    def tick(self):
        now_real = self.elapsed.elapsed() / 1000.0
        dt = min(now_real - self.last_elapsed, 1.0 / 30.0)
        self.last_elapsed = now_real
        self.now = now_real

        self._frame_count += 1
        self._fps_timer += dt
        if self._fps_timer >= 0.5:
            self.fps = self._frame_count / self._fps_timer
            self._frame_count = 0
            self._fps_timer = 0.0

        self.handle_keyboard(dt)
        dt_scaled = dt * self.time_scale

        if not self.paused:
            self.step_timer += dt_scaled
            steps = 0
            while self.step_timer >= self.step_interval and steps < 3:
                self.step_timer -= self.step_interval
                self.do_step()
                steps += 1

        if self.cinematic:
            t = self.now * 0.15
            self.target_ay = t
            self.target_ax = math.sin(t * 0.5) * 0.6 + 0.2
            self.target_zoom = 1.2 + math.sin(t * 0.8) * 0.3
            self.target_pan_x = math.sin(t * 1.1) * 50
            self.target_pan_y = math.cos(t * 0.9) * 30
        else:
            if self.auto_rotate and not self.dragging and not self.panning:
                self.target_ay += dt * 0.22
                
            if self.auto_zoom and not self.dragging and not self.panning:
                coords = np.argwhere(self.grid)
                if len(coords) > 0:
                    center = GRID / 2.0 - 0.5
                    dists = np.sqrt(np.sum((coords.astype(float) - center)**2, axis=1))
                    max_r = dists.max()
                    desired_zoom = max(0.4, min(2.5, 8.0 / (max_r + 1.0)))
                    self.target_zoom = desired_zoom
                    self.target_pan_x = 0
                    self.target_pan_y = 0

        k = min(1.0, dt * 8.0)
        self.ax += (self.target_ax - self.ax) * k
        self.ay += (self.target_ay - self.ay) * k
        self.zoom += (self.target_zoom - self.zoom) * k
        self.pan_x += (self.target_pan_x - self.pan_x) * k
        self.pan_y += (self.target_pan_y - self.pan_y) * k

        self.scan_z = (self.scan_z + dt * 5.0) % (GRID + 4)
        self.light_angle += dt * 0.8 
        self.pulse *= max(0.0, 1.0 - dt * 4.0)
        self.shake = max(0.0, self.shake - dt * 5.0)
        self.flash = max(0.0, self.flash - dt * 3.0)
        
        if self.shake > 0 and not self.reduced_motion:
            amp = self.shake * 6.0
            self.shake_offset = (np.random.uniform(-amp, amp),
                                 np.random.uniform(-amp, amp))
        else:
            self.shake_offset = (0.0, 0.0)

        self.dying = [d for d in self.dying if self.now - d[1] < 0.30]
        self.update_particles(dt)

        self.update()

    def handle_keyboard(self, dt):
        if self.cinematic: return  # Disable manual control if cinematic
        
        rot_speed = 1.5 * dt
        pan_speed = 150.0 * dt
        zoom_speed = 1.5 * dt
        
        if Qt.Key_W in self.keys: self.target_ax -= rot_speed
        if Qt.Key_S in self.keys: self.target_ax += rot_speed
        if Qt.Key_A in self.keys: self.target_ay -= rot_speed
        if Qt.Key_D in self.keys: self.target_ay += rot_speed
        if Qt.Key_Q in self.keys: self.target_zoom -= zoom_speed
        if Qt.Key_E in self.keys: self.target_zoom += zoom_speed
        
        if Qt.Key_Up in self.keys:    self.target_pan_y += pan_speed
        if Qt.Key_Down in self.keys:  self.target_pan_y -= pan_speed
        if Qt.Key_Left in self.keys:  self.target_pan_x -= pan_speed
        if Qt.Key_Right in self.keys: self.target_pan_x += pan_speed
            
        self.target_ax = max(-1.4, min(1.4, self.target_ax))
        self.target_zoom = max(0.4, min(2.5, self.target_zoom))

    def update_particles(self, dt):
        if not self.particles:
            return
        p = np.array(self.particles, dtype=float)
        p[:, 0:3] += p[:, 3:6] * dt
        p[:, 3:6] *= max(0.0, 1.0 - dt * 2.0)
        p[:, 5] -= 15.0 * dt
        p[:, 6] -= dt
        p = p[p[:, 6] > 0]
        self.particles = p.tolist()

    # ----- CA step -----
    def do_step(self):
        rule_name, birth_set, survive_set = self.get_current_rule()
        n = count_neighbors(self.grid)
        new_grid = np.zeros_like(self.grid)
        new_age = np.zeros_like(self.age, dtype=np.int32)
        new_born_at = np.full_like(self.born_at, -1.0)

        birth_count = np.zeros_like(self.grid, dtype=bool)
        for b in birth_set:
            birth_count |= (n == b)
        birth_mask = (~self.grid) & birth_count
        new_grid |= birth_mask
        new_age[birth_mask] = 1
        new_born_at[birth_mask] = self.now

        survive_count = np.zeros_like(self.grid, dtype=bool)
        for s in survive_set:
            survive_count |= (n == s)
        survive_mask = self.grid & survive_count
        new_grid |= survive_mask
        new_age[survive_mask] = self.age[survive_mask] + 1
        new_born_at[survive_mask] = self.born_at[survive_mask]

        death_mask = self.grid & ~new_grid
        
        self.grid, self.age, self.born_at = new_grid, new_age, new_born_at
        self.generation += 1
        self.population = int(self.grid.sum())
        pop_delta = abs(self.population - self.last_pop)
        self.last_pop = self.population
        
        if pop_delta > 15:
            self.pulse = max(self.pulse, min(1.0, pop_delta / 120.0))
        if pop_delta > 80 and not self.reduced_motion:
            self.shake = min(1.0, pop_delta / 200.0)

        if death_mask.any():
            death_coords = np.argwhere(death_mask)
            for c in death_coords:
                coord = tuple(int(v) for v in c)
                self.dying.append([coord, self.now])
                
                if len(self.particles) < 600:
                    cx, cy, cz = coord
                    tx, ty, tz = cx - GRID/2, cy - GRID/2, cz - GRID/2
                    dist = math.sqrt(tx*tx + ty*ty + tz*tz) + 0.01
                    fx, fy, fz = tx/dist, ty/dist, tz/dist
                    
                    for _ in range(np.random.randint(3, 6)):
                        vx = fx * np.random.uniform(2, 6) + np.random.uniform(-2, 2)
                        vy = fy * np.random.uniform(2, 6) + np.random.uniform(-2, 2)
                        vz = fz * np.random.uniform(2, 6) + np.random.uniform(-2, 2)
                        life = np.random.uniform(0.3, 0.6)
                        self.particles.append([cx, cy, cz, vx, vy, vz, life, life])

        if birth_mask.any() and self.dying:
            birth_set_vals = {tuple(int(v) for v in c) for c in np.argwhere(birth_mask)}
            self.dying = [d for d in self.dying if d[0] not in birth_set_vals]

        if self.population < 4:
            self.init_pattern()

    # ----- input -----
    def mousePressEvent(self, event):
        self.cinematic = False  # Override cinematic on click
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.auto_rotate = False
            self.last_pos = event.position()
        elif event.button() == Qt.RightButton:
            self.panning = True
            self.last_pos = event.position()
        elif event.button() == Qt.MiddleButton:
            self.inject_life_at_center()

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self.dragging:
            dx = pos.x() - self.last_pos.x()
            dy = pos.y() - self.last_pos.y()
            self.target_ay += dx * 0.01
            self.target_ax += dy * 0.01
            self.last_pos = pos
        elif self.panning:
            dx = pos.x() - self.last_pos.x()
            dy = pos.y() - self.last_pos.y()
            self.target_pan_x += dx
            self.target_pan_y -= dy
            self.last_pos = pos

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
        elif event.button() == Qt.RightButton:
            self.panning = False

    def wheelEvent(self, event):
        self.cinematic = False
        delta = event.angleDelta().y() / 120.0
        self.target_zoom += delta * 0.1
        self.target_zoom = max(0.4, min(2.5, self.target_zoom))
        if abs(delta) > 0:
            self.auto_zoom = False

    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key_Space, Qt.Key_R, Qt.Key_Escape, Qt.Key_F1, 
                 Qt.Key_F2, Qt.Key_N, Qt.Key_M, Qt.Key_C, Qt.Key_X, 
                 Qt.Key_P, Qt.Key_Z, Qt.Key_V, Qt.Key_B):
            if k == Qt.Key_Space: self.paused = not self.paused
            elif k == Qt.Key_R: self.init_pattern()
            elif k == Qt.Key_Escape: self.window().close()
            elif k == Qt.Key_F1: self.show_debug = not self.show_debug
            elif k == Qt.Key_F2: self.time_scale = 0.2 if self.time_scale > 0.5 else 1.0
            elif k == Qt.Key_N: self.next_rule()
            elif k == Qt.Key_C: self.next_palette()
            elif k == Qt.Key_X: self.toggle_symmetry()
            elif k == Qt.Key_P: self.export_frame()
            elif k == Qt.Key_Z: self.toggle_auto_zoom()
            elif k == Qt.Key_V: self.toggle_hologram()
            elif k == Qt.Key_B: self.toggle_cinematic()
            elif k == Qt.Key_M: self.reduced_motion = not self.reduced_motion
        else:
            self.keys.add(k)

    def keyReleaseEvent(self, event):
        self.keys.discard(event.key())

    # ----- rendering -----
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        pal = self.get_current_palette()

        # Radial Vignette Background
        radial_bg = QRadialGradient(QPointF(w/2, h/2), max(w, h)/1.2)
        radial_bg.setColorAt(0.0, QColor(*pal["bg_top"]))
        radial_bg.setColorAt(1.0, QColor(*pal["bg_bot"]))
        painter.fillRect(self.rect(), radial_bg)

        cx = w / 2.0 + self.shake_offset[0] + self.pan_x
        cy = h / 2.0 + self.shake_offset[1] + self.pan_y

        scale = min(w, h) / 700.0
        cube_size = 22.0 * scale * self.zoom
        pulse_scale = 1.0 + 0.06 * self.pulse

        # Dual-light setup
        light1 = np.array([math.cos(self.light_angle) * 0.7, 0.8, math.sin(self.light_angle) * 0.7])
        light1 /= np.linalg.norm(light1)
        light2 = np.array([-0.4, 0.3, 0.6])
        light2 /= np.linalg.norm(light2)

        rotated_normals = rotate_points(FACE_NORMALS, self.ax, self.ay)
        visible_faces = [i for i in range(6) if rotated_normals[i, 2] > 0]
        face_intensity = {}
        for i in visible_faces:
            i1 = max(0.0, float(np.dot(rotated_normals[i], light1)))
            i2 = max(0.0, float(np.dot(rotated_normals[i], light2)))
            face_intensity[i] = 0.25 + i1 * 0.65 + i2 * 0.25

        all_items = []
        projected_live = None
        projected_dying = None
        projected_particles = None
        cell_colors_rgb = None

        # ----- Ground Grid -----
        grid_verts_3d = self.grid_lines_3d.reshape(-1, 3) * cube_size
        grid_rot = rotate_points(grid_verts_3d, self.ax, self.ay)
        grid_proj, grid_z = project_points(grid_rot, cx, cy)
        for i in range(len(self.grid_lines_3d)):
            z1, z2 = grid_z[i*2], grid_z[i*2+1]
            avg_z = (z1 + z2) / 2.0
            all_items.append(('grid', avg_z, i))

        # ----- Fake Shadow -----
        shadow_3d_scaled = self.shadow_3d * cube_size
        shadow_rot = rotate_points(shadow_3d_scaled, self.ax, self.ay)
        shadow_proj, shadow_z = project_points(shadow_rot, cx, cy)
        shadow_avg_z = np.mean(shadow_z) + 0.5
        all_items.append(('shadow', shadow_avg_z, shadow_proj))

        # ----- live cells (with clipping) -----
        live_coords = np.argwhere(self.grid)
        if len(live_coords) > 0:
            live_coords = live_coords[live_coords[:, 2] <= self.slice_z]
            
        if len(live_coords) > 0:
            ages_sec = self.now - self.born_at[live_coords[:, 0],
                                               live_coords[:, 1],
                                               live_coords[:, 2]]
            ages_sec = np.maximum(ages_sec, 0.0)
            birth_t = np.clip(ages_sec / 0.25, 0.0, 1.0)
            birth_scales = 1.0 - (1.0 - birth_t) ** 3
            final_sizes = cube_size * birth_scales * pulse_scale

            centers = (live_coords.astype(float)
                       - GRID / 2.0 + 0.5) * cube_size
            corners = (centers[:, None, :]
                       + CUBE_VERTS[None, :, :] * final_sizes[:, None, None])
            rotated = rotate_points(corners.reshape(-1, 3), self.ax, self.ay)

            projected, z_depths = project_points(rotated, cx, cy)
            rotated_live = rotated.reshape(-1, 8, 3)
            projected_live = projected.reshape(-1, 8, 2)

            cell_ages = self.age[live_coords[:, 0],
                                 live_coords[:, 1],
                                 live_coords[:, 2]]
            cell_zs = live_coords[:, 2]

            age_t = np.clip(cell_ages / 22.0, 0.0, 1.0)
            young_rgb = np.array(pal["young"], dtype=float)
            old_rgb = np.array(pal["old"], dtype=float)
            cell_colors_rgb = (young_rgb[None, :] * (1 - age_t[:, None]) + 
                               old_rgb[None, :] * age_t[:, None])

            # Precompute avg z for all faces for all live cells
            # Shape: (N, 6)
            face_z_all = np.mean(rotated_live[:, FACES_VERTS_IDX, 2], axis=2)
            
            # Vectorized Hidden Face Culling
            # exposed_faces_mask: True if face is visible (neighbor is dead)
            exposed_faces_mask = np.ones((len(live_coords), 6), dtype=bool)
            for fi, (dx, dy, dz) in enumerate(FACE_NEIGHBOR_OFFSETS):
                nx = live_coords[:,0] + dx
                ny = live_coords[:,1] + dy
                nz = live_coords[:,2] + dz
                valid = (nx >= 0) & (nx < GRID) & (ny >= 0) & (ny < GRID) & (nz >= 0) & (nz < GRID)
                neighbor_alive = np.zeros(len(live_coords), dtype=bool)
                neighbor_alive[valid] = self.grid[nx[valid], ny[valid], nz[valid]]
                exposed_faces_mask[:, fi] = ~neighbor_alive

            # Determine which faces to actually draw
            cam_vis_mask = np.zeros(6, dtype=bool)
            cam_vis_mask[visible_faces] = True
            
            draw_mask = exposed_faces_mask & cam_vis_mask[None, :]
            ci_arr, fi_arr = np.where(draw_mask)
            
            # Append all visible faces to all_items
            for i in range(len(ci_arr)):
                ci = ci_arr[i]
                fi = fi_arr[i]
                scan_dist = abs(cell_zs[ci] - (self.scan_z - 2.0))
                scan_boost = max(0.0, 1.0 - scan_dist / 1.5)
                all_items.append(('live', float(face_z_all[ci, fi]), fi, ci, int(cell_ages[ci]),
                                  face_intensity[fi], float(birth_scales[ci]), scan_boost))

        # ----- dying cells -----
        if self.dying:
            dying_coords = np.array([d[0] for d in self.dying])
            dying_coords = dying_coords[dying_coords[:, 2] <= self.slice_z]
            
            if len(dying_coords) > 0:
                dying_times = np.array([d[1] for d in self.dying if d[0][2] <= self.slice_z])
                t_arr = np.clip((self.now - dying_times) / 0.30, 0.0, 1.0)
                eased = 1.0 - (1.0 - t_arr) ** 3
                dying_scales = 1.0 - 0.5 * eased
                dying_alphas = 1.0 - eased

                centers = (dying_coords.astype(float)
                           - GRID / 2.0 + 0.5) * cube_size
                final_sizes = cube_size * dying_scales * pulse_scale
                corners = (centers[:, None, :]
                           + CUBE_VERTS[None, :, :] * final_sizes[:, None, None])
                rotated = rotate_points(corners.reshape(-1, 3), self.ax, self.ay)

                projected, _ = project_points(rotated, cx, cy)
                rotated_dying = rotated.reshape(-1, 8, 3)
                projected_dying = projected.reshape(-1, 8, 2)

                # No hidden face culling for dying cells
                fi_arr_d = np.repeat(visible_faces, len(dying_coords))
                ci_arr_d = np.tile(np.arange(len(dying_coords)), len(visible_faces))
                
                face_z_all_d = np.mean(rotated_dying[:, FACES_VERTS_IDX, 2], axis=2)
                
                for i in range(len(ci_arr_d)):
                    ci = ci_arr_d[i]
                    fi = fi_arr_d[i]
                    all_items.append(('dying', float(face_z_all_d[ci, fi]), fi, ci, 0,
                                      face_intensity[fi],
                                      float(dying_alphas[ci]), 0.0))

        # ----- particles prep -----
        if self.particles:
            p_arr = np.array(self.particles, dtype=float)
            p_arr = p_arr[p_arr[:, 2] <= self.slice_z]
            
            if len(p_arr) > 0:
                centers = (p_arr[:, 0:3] - GRID / 2.0 + 0.5) * cube_size
                rotated = rotate_points(centers, self.ax, self.ay)
                projected, z_depths = project_points(rotated, cx, cy)
                projected_particles = projected

                for i in range(len(p_arr)):
                    life_t = p_arr[i, 6] / p_arr[i, 7]
                    size = 2.0 + (1.0 - life_t) * 4.0
                    all_items.append(('particle', float(z_depths[i]), i, size, life_t))

        # ----- bounding box -----
        box_verts_3d = (CUBE_VERTS * BOX_SCALE * cube_size)
        box_rotated = rotate_points(box_verts_3d, self.ax, self.ay)
        box_proj, box_z = project_points(box_rotated, cx, cy)
        
        for i, (v1, v2) in enumerate(BOX_EDGES):
            avg_z = (box_z[v1] + box_z[v2]) / 2.0
            all_items.append(('box', avg_z, i, v1, v2))

        # Depth sort all items
        if all_items:
            z_vals = np.array([item[1] for item in all_items])
            sort_idx = np.argsort(z_vals)
            all_items = [all_items[i] for i in sort_idx]

        bg_color = np.array(pal["bg_bot"], dtype=float)

        # Draw sorted solid items
        if self.hologram:
            painter.setCompositionMode(QPainter.CompositionMode_Plus)

        for item in all_items:
            if item[0] in ('live', 'dying'):
                kind, avg_z, fi, ci, age_gen, intensity, alpha, scan_boost = item
                verts = FACES[fi][0]
                proj = projected_live if kind == 'live' else projected_dying
                pts = [QPointF(proj[ci, v, 0], proj[ci, v, 1]) for v in verts]

                if kind == 'live':
                    r, g, b = cell_colors_rgb[ci]
                    if scan_boost > 0:
                        r = min(255, r + (255 - r) * scan_boost)
                        g = min(255, g + (255 - g) * scan_boost)
                        b = min(255, b + (255 - b) * scan_boost)
                else:
                    r, g, b = 180, 180, 200

                fog_amt = max(0.0, min(1.0, (150.0 - avg_z) / 300.0))
                fog_amt = fog_amt ** 2
                
                r = r * (1 - fog_amt) + bg_color[0] * fog_amt
                g = g * (1 - fog_amt) + bg_color[1] * fog_amt
                b = b * (1 - fog_amt) + bg_color[2] * fog_amt

                if self.hologram:
                    # Hologram mode: Draw bright wireframes
                    edge_r = min(255, r * 1.5 + 50)
                    edge_g = min(255, g * 1.5 + 50)
                    edge_b = min(255, b * 1.5 + 50)
                    edge_a = int(220 * alpha * (1.0 - fog_amt))
                    painter.setPen(QPen(QColor(edge_r, edge_g, edge_b, edge_a), 1.5))
                    painter.setBrush(Qt.NoBrush)
                else:
                    # Solid mode: Draw filled polygons
                    r = min(255.0, r * intensity)
                    g = min(255.0, g * intensity)
                    b = min(255.0, b * intensity)
                    a = int(255 * alpha)
                    
                    edge_r = min(255, r * 1.4 + 30)
                    edge_g = min(255, g * 1.4 + 30)
                    edge_b = min(255, b * 1.4 + 30)
                    edge_a = int(220 * alpha * (1.0 - fog_amt))
                    
                    painter.setPen(QPen(QColor(edge_r, edge_g, edge_b, edge_a), 1))
                    painter.setBrush(QColor(int(r), int(g), int(b), a))
                
                painter.drawPolygon(QPolygonF(pts))
                    
            elif item[0] == 'grid' and not self.hologram:
                _, avg_z, i = item
                fog_amt = max(0.0, min(1.0, (150.0 - avg_z) / 300.0))
                fog_amt = fog_amt ** 2
                alpha = int(120 * (1.0 - fog_amt))
                painter.setPen(QPen(QColor(*pal["bound"], alpha), 1))
                painter.setBrush(Qt.NoBrush)
                p1 = grid_proj[i*2]
                p2 = grid_proj[i*2+1]
                painter.drawLine(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]))
                
            elif item[0] == 'shadow' and not self.hologram:
                _, avg_z, proj_pts = item
                fog_amt = max(0.0, min(1.0, (150.0 - avg_z) / 300.0))
                fog_amt = fog_amt ** 2
                alpha = int(140 * (1.0 - fog_amt))
                if self.ax > 0:
                    pts = [QPointF(p[0], p[1]) for p in proj_pts]
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(0, 0, 0, alpha))
                    painter.drawPolygon(QPolygonF(pts))
                    
            elif item[0] == 'box' and not self.hologram:
                _, avg_z, i, v1, v2 = item
                fog_amt = max(0.0, min(1.0, (150.0 - avg_z) / 300.0))
                fog_amt = fog_amt ** 2
                alpha = int(180 * (1.0 - fog_amt))
                painter.setPen(QPen(QColor(*pal["bound"], alpha), 1.5, Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(QPointF(box_proj[v1, 0], box_proj[v1, 1]),
                                 QPointF(box_proj[v2, 0], box_proj[v2, 1]))
                
            elif item[0] == 'particle':
                pass

        if self.hologram:
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # ----- Additive Bloom Pass (Glows & Particles) -----
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        painter.setPen(Qt.NoPen)
        
        # 1. Newborn cell halos
        if projected_live is not None and len(live_coords) > 0:
            for ci in range(len(live_coords)):
                age_gen = int(cell_ages[ci])
                birth_alpha = float(birth_scales[ci])
                if age_gen <= 2 and birth_alpha < 1.0:
                    center_3d = rotated_live[ci].mean(axis=0)
                    center_2d, _ = project_points(np.array([center_3d]), cx, cy)
                    cpx, cpy = center_2d[0, 0], center_2d[0, 1]
                    
                    halo_r = 25 * (1.0 - birth_alpha) + 8
                    halo_a = int(160 * (1.0 - birth_alpha))
                    
                    glow_col = pal["young"]
                    painter.setBrush(QColor(glow_col[0], glow_col[1], glow_col[2], max(0, halo_a)))
                    painter.drawEllipse(QPointF(cpx, cpy), halo_r, halo_r)

        # 2. Comet Particles
        if projected_particles is not None:
            p_col = pal["particle"]
            for item in all_items:
                if item[0] == 'particle':
                    _, avg_z, i, size, life_t = item
                    px, py = projected_particles[i]
                    alpha = int(255 * life_t)
                    
                    painter.setBrush(QColor(p_col[0], p_col[1], p_col[2], max(0, int(alpha * 0.3))))
                    painter.drawEllipse(QPointF(px, py), size * 3.0, size * 3.0)
                    painter.setBrush(QColor(p_col[0], p_col[1], p_col[2], max(0, alpha)))
                    painter.drawEllipse(QPointF(px, py), size, size)

        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # Export Flash
        if self.flash > 0.01:
            painter.fillRect(self.rect(), QColor(255, 255, 255, int(self.flash * 200)))

        self.draw_hud(painter, w, h)

    def draw_hud(self, painter, w, h):
        painter.setPen(QColor(220, 225, 235, 230))
        painter.setFont(QFont("DejaVu Sans", 10))
        rule_name = self.get_current_rule()[0]
        pal_name = self.get_current_palette()["name"]
        lines = [
            f"Generation : {self.generation}",
            f"Population : {self.population}",
            f"Rule       : {rule_name}",
            f"Palette    : {pal_name}",
            f"Symmetry   : {'ON' if self.symmetry else 'OFF'}",
            f"Auto-Zoom  : {'ON' if self.auto_zoom else 'OFF'}",
            f"Hologram   : {'ON' if self.hologram else 'OFF'}",
            f"Cinematic  : {'ON' if self.cinematic else 'OFF'}",
            f"Slice Z    : {self.slice_z}/{GRID}",
            f"Grid       : {GRID}^3 = {GRID**3} cells",
        ]
        if self.paused: lines.append("[ PAUSED ]")
        if self.time_scale < 0.5: lines.append("[ SLOW-MO ]")
        if self.reduced_motion: lines.append("[ REDUCED MOTION ]")
        if self.show_debug:
            lines.append(f"FPS        : {self.fps:5.1f}")
            lines.append(f"Particles  : {len(self.particles)}")
            lines.append(f"Rot ax,ay  : {self.ax:+.2f}, {self.ay:+.2f}")
            lines.append(f"Zoom       : {self.zoom:.2f}")
        for i, line in enumerate(lines):
            painter.drawText(14, 22 + i * 16, line)

        painter.setPen(QColor(150, 160, 175, 180))
        hint = ("L-Drag: Rot  R-Drag: Pan  Mid-Click: Inject Life  Wheel: Zoom  "
                "WASD: Rot  QE: Zoom  Arrows: Pan  Space: Pause  R: Reset  "
                "N: Rule  C: Palette  X: Symmetry  Z: Auto-Zoom  V: Hologram  B: Cinematic  P: PNG  Esc: Quit")
        painter.drawText(14, h - 14, hint)


# ---------------------------------------------------------------------------
# Main window with control bar
# ---------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Cellular Automata — PySide6 Ultimate Edition")
        self.resize(1100, 820)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.view = CAView()
        layout.addWidget(self.view, 1)

        bar = QFrame()
        bar.setObjectName("ControlBar")
        bar.setStyleSheet("QFrame#ControlBar { background: #0a0d18; border-top: 1px solid #1c2238; }")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(14, 10, 14, 10)
        bar_layout.setSpacing(6)

        self.btn_reset = QPushButton("Reset")
        self.btn_pause = QPushButton("Pause")
        self.btn_rule = QPushButton("Next Rule")
        self.btn_palette = QPushButton("Next Palette")
        self.btn_symmetry = QPushButton("Symmetry")
        self.btn_auto = QPushButton("Auto-Rotate")
        self.btn_autozoom = QPushButton("Auto-Zoom")
        self.btn_hologram = QPushButton("Hologram")
        self.btn_cinematic = QPushButton("Cinematic")
        self.btn_export = QPushButton("Save PNG")
        
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_reset.clicked.connect(self.view.init_pattern)
        self.btn_rule.clicked.connect(self.view.next_rule)
        self.btn_palette.clicked.connect(self.view.next_palette)
        self.btn_symmetry.clicked.connect(self.toggle_symmetry)
        self.btn_auto.clicked.connect(self.toggle_auto)
        self.btn_autozoom.clicked.connect(self.toggle_autozoom)
        self.btn_hologram.clicked.connect(self.toggle_hologram)
        self.btn_cinematic.clicked.connect(self.toggle_cinematic)
        self.btn_export.clicked.connect(self.view.export_frame)
        
        self.btn_auto.setCheckable(True)
        self.btn_auto.setChecked(self.view.auto_rotate)
        self.btn_symmetry.setCheckable(True)
        self.btn_symmetry.setChecked(self.view.symmetry)
        self.btn_autozoom.setCheckable(True)
        self.btn_autozoom.setChecked(self.view.auto_zoom)
        self.btn_hologram.setCheckable(True)
        self.btn_hologram.setChecked(self.view.hologram)
        self.btn_cinematic.setCheckable(True)
        self.btn_cinematic.setChecked(self.view.cinematic)

        self.speed_label = QLabel("Speed")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(2, 30)
        self.speed_slider.setValue(int(self.view.step_interval * 60))
        self.speed_slider.valueChanged.connect(self.on_speed)

        self.density_label = QLabel("Density")
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setRange(10, 50)
        self.density_slider.setValue(self.view.density)
        self.density_slider.valueChanged.connect(self.on_density)

        self.slice_label = QLabel("Slice Z")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, GRID)
        self.slice_slider.setValue(self.view.slice_z)
        self.slice_slider.valueChanged.connect(self.on_slice)

        # Row 1 of buttons
        btn_row1 = QHBoxLayout()
        btn_row1.addWidget(self.btn_reset)
        btn_row1.addWidget(self.btn_pause)
        btn_row1.addWidget(self.btn_rule)
        btn_row1.addWidget(self.btn_palette)
        btn_row1.addWidget(self.btn_symmetry)
        
        # Row 2 of buttons
        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self.btn_auto)
        btn_row2.addWidget(self.btn_autozoom)
        btn_row2.addWidget(self.btn_hologram)
        btn_row2.addWidget(self.btn_cinematic)
        btn_row2.addWidget(self.btn_export)

        bar_layout.addLayout(btn_row1)
        bar_layout.addSpacing(10)
        bar_layout.addLayout(btn_row2)
        bar_layout.addSpacing(20)
        
        bar_layout.addWidget(self.speed_label)
        bar_layout.addWidget(self.speed_slider)
        bar_layout.addSpacing(10)
        
        bar_layout.addWidget(self.density_label)
        bar_layout.addWidget(self.density_slider)
        bar_layout.addSpacing(10)
        
        bar_layout.addWidget(self.slice_label)
        bar_layout.addWidget(self.slice_slider)

        layout.addWidget(bar)

        self.setStyleSheet("""
            QPushButton {
                background: #161b2e; color: #e4e8f0;
                border: 1px solid #2a3147; padding: 6px 10px;
                border-radius: 4px; font-size: 11px;
            }
            QPushButton:hover   { background: #1f263e; border-color: #3a4366; }
            QPushButton:pressed { background: #11162a; }
            QPushButton:checked { background: #2a4a7a; border-color: #5a8fd6; color: #fff; }
            QLabel { color: #98a0b5; font-size: 11px; }
            QSlider::groove:horizontal { background: #161b2e; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #5a8fd6; width: 14px; margin: -6px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #5a8fd6; border-radius: 2px; }
        """)
        
        self.view.setFocus()

    def toggle_pause(self):
        self.view.paused = not self.view.paused
        self.btn_pause.setText("Play" if self.view.paused else "Pause")
        self.view.setFocus()

    def toggle_auto(self):
        self.view.auto_rotate = not self.view.auto_rotate
        self.btn_auto.setChecked(self.view.auto_rotate)
        self.view.setFocus()

    def toggle_symmetry(self):
        self.view.toggle_symmetry()
        self.btn_symmetry.setChecked(self.view.symmetry)
        self.view.setFocus()

    def toggle_autozoom(self):
        self.view.toggle_auto_zoom()
        self.btn_autozoom.setChecked(self.view.auto_zoom)
        self.view.setFocus()

    def toggle_hologram(self):
        self.view.toggle_hologram()
        self.btn_hologram.setChecked(self.view.hologram)
        self.view.setFocus()

    def toggle_cinematic(self):
        self.view.toggle_cinematic()
        self.btn_cinematic.setChecked(self.view.cinematic)
        self.view.setFocus()

    def on_speed(self, v):
        self.view.step_interval = max(0.05, v / 60.0)
        self.view.setFocus()

    def on_density(self, v):
        self.view.density = v
        self.view.setFocus()

    def on_slice(self, v):
        self.view.slice_z = v
        self.view.setFocus()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()