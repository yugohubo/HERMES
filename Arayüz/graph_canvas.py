import math
import random
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QMouseEvent, QWheelEvent
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF

class GraphCanvas(QWidget):
    nodeSelected = pyqtSignal(dict) # Emits node properties dictionary when clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

        # Graph data
        self.nodes = {} # id -> { "id", "label", "description", "x", "y", "vx", "vy", "radius", "is_doc" }
        self.edges = [] # list of { "source", "target", "label", "description" }
        
        # Interactive state
        self.zoom = 1.0
        self.pan_offset = QPointF(0, 0)
        self.dragged_node = None
        self.selected_node_id = None
        self.last_mouse_pos = None
        self.is_panning = False

        # Physics constants
        self.k_rep = 2500.0   # Repulsion constant
        self.k_att = 0.08     # Spring constant
        self.rest_len = 150.0 # Rest length of spring
        self.damping = 0.85   # Friction
        self.center_grav = 0.02 # Gravity pulling to center

        # Physics Timer (approx 30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_physics)
        self.timer.start(30)

    def set_graph_data(self, graph_data: dict):
        """Set the nodes and edges, preserving positions of existing nodes if possible."""
        new_nodes = {}
        cx = self.width() / 2
        cy = self.height() / 2
        
        for n in graph_data.get("nodes", []):
            nid = n["id"]
            node_label = n.get("node_label", "Concept")
            concept_type = n.get("concept_type", "Other")
            
            # Vary node size depending on type
            if node_label == "Company":
                radius = 28
            elif node_label == "Project":
                radius = 26
            elif node_label in ["User", "Document"]:
                radius = 24
            else:
                radius = 20
                
            if nid in self.nodes:
                new_nodes[nid] = self.nodes[nid]
                new_nodes[nid]["label"] = n["label"]
                new_nodes[nid]["description"] = n["description"]
                new_nodes[nid]["node_label"] = node_label
                new_nodes[nid]["concept_type"] = concept_type
                new_nodes[nid]["radius"] = radius
            else:
                rx = cx + random.uniform(-50, 50)
                ry = cy + random.uniform(-50, 50)
                new_nodes[nid] = {
                    "id": nid,
                    "label": n["label"],
                    "description": n["description"],
                    "node_label": node_label,
                    "concept_type": concept_type,
                    "x": rx,
                    "y": ry,
                    "vx": 0.0,
                    "vy": 0.0,
                    "radius": radius
                }
        
        self.nodes = new_nodes
        self.edges = graph_data.get("edges", [])
        
        if self.selected_node_id and self.selected_node_id not in self.nodes:
            self.selected_node_id = None
            
        self.update()

    def update_physics(self):
        """Single step of the spring-physics simulation."""
        if not self.nodes:
            return

        node_keys = list(self.nodes.keys())
        n_count = len(node_keys)
        cx = self.width() / 2
        cy = self.height() / 2

        # 1. Repulsion between all pairs of nodes
        for i in range(n_count):
            n1 = self.nodes[node_keys[i]]
            # Skip moving if it's the node currently dragged by mouse
            if n1 is self.dragged_node:
                continue
                
            fx, fy = 0.0, 0.0
            
            for j in range(n_count):
                if i == j:
                    continue
                n2 = self.nodes[node_keys[j]]
                
                dx = n1["x"] - n2["x"]
                dy = n1["y"] - n2["y"]
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq)
                
                if dist < 1.0:
                    # Avoid division by zero
                    dx = random.uniform(-1, 1)
                    dy = random.uniform(-1, 1)
                    dist = 1.0
                    dist_sq = 1.0
                
                # Force is inversely proportional to distance squared
                force = self.k_rep / dist_sq
                fx += (dx / dist) * force
                fy += (dy / dist) * force

            # Apply accumulated repulsion force
            n1["vx"] += fx
            n1["vy"] += fy

        # 2. Attraction along edges
        for edge in self.edges:
            src_id = edge["source"]
            tgt_id = edge["target"]
            
            if src_id not in self.nodes or tgt_id not in self.nodes:
                continue
                
            n_src = self.nodes[src_id]
            n_tgt = self.nodes[tgt_id]
            
            dx = n_tgt["x"] - n_src["x"]
            dy = n_tgt["y"] - n_src["y"]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 1.0:
                dist = 1.0
                
            # Hooke's Law: F = k * (x - L)
            force = self.k_att * (dist - self.rest_len)
            
            # Unit direction vectors
            ux = dx / dist
            uy = dy / dist
            
            # Pull source node towards target
            if n_src is not self.dragged_node:
                n_src["vx"] += ux * force
                n_src["vy"] += uy * force
                
            # Pull target node towards source
            if n_tgt is not self.dragged_node:
                n_tgt["vx"] -= ux * force
                n_tgt["vy"] -= uy * force

        # 3. Center Gravity (pulls nodes to center to prevent floating away)
        for key in node_keys:
            n = self.nodes[key]
            if n is self.dragged_node:
                continue
                
            dx = cx - n["x"]
            dy = cy - n["y"]
            n["vx"] += dx * self.center_grav
            n["vy"] += dy * self.center_grav

        # 4. Apply velocities, friction and update positions
        for key in node_keys:
            n = self.nodes[key]
            if n is self.dragged_node:
                continue
                
            # Apply velocity, capped to avoid explosive movements
            n["vx"] = max(-30, min(30, n["vx"]))
            n["vy"] = max(-30, min(30, n["vy"]))
            
            n["x"] += n["vx"]
            n["y"] += n["vy"]
            
            # Dampen velocity (friction)
            n["vx"] *= self.damping
            n["vy"] *= self.damping

        self.update()

    def paintEvent(self, event):
        """Draw the graph using QPainter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Fill Background
        painter.fillRect(self.rect(), QColor("#121214")) # Sleek dark grey/black

        # Save painter state before applying transforms
        painter.save()
        
        # 2. Apply Pan & Zoom transformations
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(self.zoom, self.zoom)
        painter.translate(-self.width() / 2 + self.pan_offset.x(), -self.height() / 2 + self.pan_offset.y())

        # 3. Draw Edges
        for edge in self.edges:
            src_id = edge["source"]
            tgt_id = edge["target"]
            
            if src_id not in self.nodes or tgt_id not in self.nodes:
                continue
                
            n_src = self.nodes[src_id]
            n_tgt = self.nodes[tgt_id]
            
            # Determine edge color based on selection status
            is_highlighted = (self.selected_node_id in (src_id, tgt_id))
            
            if self.selected_node_id is not None:
                if is_highlighted:
                    pen_color = QColor("#bb86fc") # Neon Purple highlight
                    pen_width = 2
                else:
                    pen_color = QColor("rgba(255, 255, 255, 0.05)") # Fade out others
                    pen_width = 1
            else:
                pen_color = QColor("rgba(255, 255, 255, 0.15)") # Standard translucent edge
                pen_width = 1
                
            pen = QPen(pen_color, pen_width)
            if is_highlighted:
                pen.setStyle(Qt.PenStyle.SolidLine)
            else:
                pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            
            # Draw line
            painter.drawLine(int(n_src["x"]), int(n_src["y"]), int(n_tgt["x"]), int(n_tgt["y"]))
            
            # Optional: draw small arrow in middle
            if is_highlighted:
                # Calculate middle point
                mx = (n_src["x"] + n_tgt["x"]) / 2
                my = (n_src["y"] + n_tgt["y"]) / 2
                
                # Angle of line
                angle = math.atan2(n_tgt["y"] - n_src["y"], n_tgt["x"] - n_src["x"])
                
                # Draw a tiny arrowhead pointing to target
                arrow_size = 6
                p1x = mx - arrow_size * math.cos(angle - math.pi/6)
                p1y = my - arrow_size * math.sin(angle - math.pi/6)
                p2x = mx - arrow_size * math.cos(angle + math.pi/6)
                p2y = my - arrow_size * math.sin(angle + math.pi/6)
                
                painter.setBrush(QBrush(QColor("#bb86fc")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon([QPointF(mx, my), QPointF(p1x, p1y), QPointF(p2x, p2y)])

        # 4. Draw Nodes
        font = QFont("Outfit", 9)
        font.setBold(True)
        painter.setFont(font)

        for nid, n in self.nodes.items():
            is_selected = (nid == self.selected_node_id)
            is_highlighted = False
            
            # Check if this node is connected to the selected node
            if self.selected_node_id is not None:
                if is_selected:
                    is_highlighted = True
                else:
                    # Look if there is an edge between this node and the selected one
                    for edge in self.edges:
                        if (edge["source"] == nid and edge["target"] == self.selected_node_id) or \
                           (edge["target"] == nid and edge["source"] == self.selected_node_id):
                            is_highlighted = True
                            break
            
            # Determine transparency/glow
            opacity = 255
            if self.selected_node_id is not None and not is_highlighted:
                opacity = 60 # Fade out unselected/unconnected nodes

            # Determine color by label & concept_type
            node_label = n.get("node_label", "Concept")
            concept_type = n.get("concept_type", "Other")
            
            if node_label == "Company":
                bg_color = QColor(220, 38, 38, opacity)
                border_color = QColor(248, 113, 113, opacity)
            elif node_label == "Project":
                bg_color = QColor(5, 150, 105, opacity)
                border_color = QColor(52, 211, 153, opacity)
            elif node_label == "User":
                bg_color = QColor(217, 119, 6, opacity)
                border_color = QColor(251, 191, 36, opacity)
            elif node_label == "Document":
                bg_color = QColor(79, 70, 229, opacity)
                border_color = QColor(129, 140, 248, opacity)
            else:
                if concept_type == "Technology":
                    bg_color = QColor(8, 145, 178, opacity)
                    border_color = QColor(34, 211, 238, opacity)
                elif concept_type == "Person":
                    bg_color = QColor(219, 39, 119, opacity)
                    border_color = QColor(244, 114, 182, opacity)
                elif concept_type == "Algorithm":
                    bg_color = QColor(234, 88, 12, opacity)
                    border_color = QColor(251, 146, 60, opacity)
                elif concept_type == "Parameter":
                    bg_color = QColor(101, 163, 13, opacity)
                    border_color = QColor(163, 230, 53, opacity)
                else:
                    bg_color = QColor(71, 85, 105, opacity)
                    border_color = QColor(148, 163, 184, opacity)

            if is_selected:
                # Highlight selected node with golden/orange ring
                border_color = QColor("#ffb74d")
                border_width = 3
            elif is_highlighted and self.selected_node_id is not None:
                border_width = 2
            else:
                border_width = 1.5

            # Draw Node Circle
            painter.setBrush(QBrush(bg_color))
            painter.setPen(QPen(border_color, border_width))
            
            radius = n["radius"]
            painter.drawEllipse(int(n["x"] - radius), int(n["y"] - radius), int(radius * 2), int(radius * 2))

            # Draw Node Text Label
            painter.setPen(QPen(QColor(255, 255, 255, opacity)))
            
            # Wrap text to fit node width, or truncate
            label_text = n["label"]
            if len(label_text) > 12:
                label_text = label_text[:10] + ".."
                
            rect_x = int(n["x"] - radius * 1.5)
            rect_y = int(n["y"] - radius)
            rect_w = int(radius * 3)
            rect_h = int(radius * 2)
            
            painter.drawText(rect_x, rect_y, rect_w, rect_h, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, label_text)

        # Restore painter state (resets translations/scaling)
        painter.restore()

        # Draw a small UI HUD overlay (watermark / controls)
        painter.setPen(QPen(QColor("rgba(255, 255, 255, 0.4)")))
        font_hud = QFont("Consolas", 8)
        painter.setFont(font_hud)
        painter.drawText(15, self.height() - 40, f"Zoom: {self.zoom:.2f}x | Düğüm: {len(self.nodes)} | İlişki: {len(self.edges)}")
        painter.drawText(15, self.height() - 25, "Mouse Sol Tık: Sürükle | Mouse Tekerlek: Zoom | Düğüm Seçmek için Tıklayın")

    def get_node_at(self, pos: QPointF) -> dict:
        """Finds a node at the coordinate, accounting for pan and zoom."""
        # Convert screen pos to graph coords
        cx = self.width() / 2
        cy = self.height() / 2
        
        # Reverse scaling and translation
        gx = (pos.x() - cx) / self.zoom + cx - self.pan_offset.x()
        gy = (pos.y() - cy) / self.zoom + cy - self.pan_offset.y()
        
        for nid, n in self.nodes.items():
            dx = n["x"] - gx
            dy = n["y"] - gy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist <= n["radius"]:
                return n
        return None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            node = self.get_node_at(QPointF(event.position()))
            if node:
                self.dragged_node = node
                self.selected_node_id = node["id"]
                self.nodeSelected.emit(node)
            else:
                self.is_panning = True
                self.last_mouse_pos = event.position()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragged_node:
            # Move the node to the graph coordinates corresponding to the mouse
            cx = self.width() / 2
            cy = self.height() / 2
            gx = (event.position().x() - cx) / self.zoom + cx - self.pan_offset.x()
            gy = (event.position().y() - cy) / self.zoom + cy - self.pan_offset.y()
            
            self.dragged_node["x"] = gx
            self.dragged_node["y"] = gy
            # Zero out velocity
            self.dragged_node["vx"] = 0.0
            self.dragged_node["vy"] = 0.0
            self.update()
        elif self.is_panning and self.last_mouse_pos:
            diff = event.position() - self.last_mouse_pos
            # Adjust panning based on zoom factor
            self.pan_offset += diff / self.zoom
            self.last_mouse_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragged_node = None
            self.is_panning = False
            self.last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent):
        # Zoom in or out
        angle = event.angleDelta().y()
        zoom_factor = 1.15 if angle > 0 else 0.85
        
        new_zoom = self.zoom * zoom_factor
        # Cap zoom levels
        self.zoom = max(0.2, min(5.0, new_zoom))
        self.update()
        
    def reset_view(self):
        """Resets panning and zooming to defaults."""
        self.zoom = 1.0
        self.pan_offset = QPointF(0, 0)
        self.selected_node_id = None
        self.update()
