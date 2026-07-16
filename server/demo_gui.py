import sys
import json
import math
import traceback
from typing import Optional, List, Dict, Set, Tuple

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QPointF
from PyQt6.QtGui import QFont, QPainter, QPolygonF, QColor, QPen, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, QGroupBox, QGridLayout,
    QGraphicsScene, QGraphicsView, QGraphicsPolygonItem, QGraphicsTextItem,
    QGraphicsEllipseItem, QScrollArea, QFrame, QMessageBox
)

from demo_client import Client, Hexudon
from demo_engine import PLAIN, ROAD, MOUNTAIN, POND, HexGrid


TERRAIN_COLORS = {
    PLAIN: QColor(200, 240, 200),
    ROAD: QColor(220, 220, 220),
    MOUNTAIN: QColor(160, 100, 50),
    POND: QColor(100, 150, 250)
}

class Worker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            res = self.func(*self.args, **self.kwargs)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(f"Error: {e}\n{traceback.format_exc()}")


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)
    def write(self, text):
        self.textWritten.emit(str(text))
    def flush(self):
        pass


class HexMapView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.hex_radius = 20
        self.hex_w = math.sqrt(3) * self.hex_radius
        self.hex_h = 2 * self.hex_radius

    def draw_map(self, grid: HexGrid, spots: List[dict], agents: List[dict], planned_actions: List[List[int]] = None):
        self.scene.clear()
        if not grid:
            return

        for r in range(grid.height):
            for c in range(grid.width):
                pos = r * grid.width + c
                terrain = grid.cells[pos]

                cx = c * self.hex_w
                if r % 2 != 0:
                    cx -= self.hex_w / 2
                cy = r * 3/4 * self.hex_h

                poly = QPolygonF()
                for i in range(6):
                    angle_deg = 60 * i - 30
                    angle_rad = math.pi / 180 * angle_deg
                    poly.append(QPointF(cx + self.hex_radius * math.cos(angle_rad),
                                        cy + self.hex_radius * math.sin(angle_rad)))

                item = self.scene.addPolygon(poly, QPen(Qt.GlobalColor.black), QBrush(TERRAIN_COLORS.get(terrain, Qt.GlobalColor.white)))

                spot = next((s for s in spots if s["pos"] == pos), None)
                if spot:
                    txt = self.scene.addText(f"B{spot['brand']}\nS{spot['stocks']}")
                    txt.setFont(QFont("Arial", 7, QFont.Weight.Bold))
                    txt.setDefaultTextColor(Qt.GlobalColor.darkRed)
                    txt.setPos(cx - 10, cy - 10)

        for ag in agents:
            pos = ag["pos"]
            aid = ag["agent_id"]
            atype = ag.get("type", "unknown")
            r, c = divmod(pos, grid.width)
            cx = c * self.hex_w
            if r % 2 != 0:
                cx -= self.hex_w / 2
            cy = r * 3/4 * self.hex_h

            color = Qt.GlobalColor.blue if atype == "patrol" else Qt.GlobalColor.magenta
            ell = self.scene.addEllipse(cx - 8, cy - 8, 16, 16, QPen(Qt.GlobalColor.black), QBrush(color))

            txt = self.scene.addText(f"{aid}")
            txt.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            txt.setDefaultTextColor(Qt.GlobalColor.white)
            txt.setPos(cx - 6, cy - 10)

        if planned_actions and len(planned_actions) == len(agents):
            for i, ag in enumerate(agents):
                path = planned_actions[i]
                cur_pos = ag["pos"]

                r, c = divmod(cur_pos, grid.width)
                cx = c * self.hex_w
                if r % 2 != 0: cx -= self.hex_w / 2
                cy = r * 3/4 * self.hex_h

                pen = QPen(Qt.GlobalColor.darkGreen if ag.get("type") == "patrol" else Qt.GlobalColor.magenta, 2, Qt.PenStyle.DashLine)

                for action in path:
                    if 0 <= action <= 5:
                        next_pos = None
                        for nd, nb in grid.neighbors(cur_pos):
                            if nd == action:
                                next_pos = nb
                                break
                        if next_pos is not None:
                            nr, nc = divmod(next_pos, grid.width)
                            nx = nc * self.hex_w
                            if nr % 2 != 0: nx -= self.hex_w / 2
                            ny = nr * 3/4 * self.hex_h

                            self.scene.addLine(cx, cy, nx, ny, pen)
                            cx, cy = nx, ny
                            cur_pos = next_pos


class DemoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HEXUDON Demo")
        self.resize(1000, 700)

        self.engine = None
        self.client = None
        self.worker = None

        self.grid = None
        self.spots = []
        self.day_steps_list = []
        self.fuel_limits = 0
        self.n_agents = 0

        self.current_day = 0
        self.state = None
        self.agents_ui_rows = []

        self.planned_actions = []

        self._setup_ui()
        self._load_config()

        sys.stdout = EmittingStream(textWritten=self.log)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left_panel = QVBoxLayout()

        cfg_group = QGroupBox("1. Configuration")
        grid = QGridLayout(cfg_group)
        self.input_url = QLineEdit("http://localhost:8000")
        self.input_game_id = QLineEdit("")
        self.input_team_id = QLineEdit("1")
        self.input_token = QLineEdit("")
        self.input_token.setEchoMode(QLineEdit.EchoMode.Password)
        grid.addWidget(QLabel("Server URL:"), 0, 0)
        grid.addWidget(self.input_url, 0, 1)
        grid.addWidget(QLabel("Game ID:"), 1, 0)
        grid.addWidget(self.input_game_id, 1, 1)
        grid.addWidget(QLabel("Team ID:"), 2, 0)
        grid.addWidget(self.input_team_id, 2, 1)
        grid.addWidget(QLabel("Token:"), 3, 0)
        grid.addWidget(self.input_token, 3, 1)

        self.btn_connect = QPushButton("Connect & Load Map")
        self.btn_connect.clicked.connect(self.cmd_connect)
        grid.addWidget(self.btn_connect, 4, 0, 1, 2)
        left_panel.addWidget(cfg_group)

        roles_group = QGroupBox("2. Agent Roles")
        self.roles_layout = QVBoxLayout(roles_group)

        roles_btns = QHBoxLayout()
        self.btn_auto_roles = QPushButton("Auto Assign")
        self.btn_auto_roles.clicked.connect(self.cmd_auto_roles)
        self.btn_submit_roles = QPushButton("Submit Roles")
        self.btn_submit_roles.clicked.connect(self.cmd_submit_roles)
        roles_btns.addWidget(self.btn_auto_roles)
        roles_btns.addWidget(self.btn_submit_roles)
        self.roles_layout.addLayout(roles_btns)

        self.agents_scroll = QScrollArea()
        self.agents_scroll.setWidgetResizable(True)
        self.agents_container = QWidget()
        self.agents_inner_layout = QVBoxLayout(self.agents_container)
        self.agents_scroll.setWidget(self.agents_container)
        self.roles_layout.addWidget(self.agents_scroll)

        left_panel.addWidget(roles_group)

        daily_group = QGroupBox("3. Daily Execution")
        daily_layout = QVBoxLayout(daily_group)

        self.lbl_day = QLabel("Day: -")
        daily_layout.addWidget(self.lbl_day)

        self.btn_fetch_day = QPushButton("Fetch Day State")
        self.btn_fetch_day.clicked.connect(self.cmd_fetch_day)

        self.btn_plan_day = QPushButton("Generate Plan (Lazy)")
        self.btn_plan_day.clicked.connect(self.cmd_plan_day)

        self.input_plan_json = QTextEdit()
        self.input_plan_json.setPlaceholderText("JSON Action Plan")
        self.input_plan_json.setMaximumHeight(100)
        self.input_plan_json.setFont(QFont("Courier", 8))

        self.btn_visualize_plan = QPushButton("Visualize Plan")
        self.btn_visualize_plan.clicked.connect(self.cmd_visualize_plan)

        self.btn_submit_day = QPushButton("Submit Plan")
        self.btn_submit_day.clicked.connect(self.cmd_submit_day)

        daily_layout.addWidget(self.btn_fetch_day)
        daily_layout.addWidget(self.btn_plan_day)
        daily_layout.addWidget(self.input_plan_json)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_visualize_plan)
        btn_layout.addWidget(self.btn_submit_day)
        daily_layout.addLayout(btn_layout)

        left_panel.addWidget(daily_group)

        metrics_group = QGroupBox("4. Live Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self.lbl_metric_day = QLabel("Day: 0")
        self.lbl_metric_brands = QLabel("Distinct Brands: 0")
        self.lbl_metric_agents = QLabel("Agent Status (Fuel):")
        self.lbl_metric_agents.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_metric_day)
        metrics_layout.addWidget(self.lbl_metric_brands)
        metrics_layout.addWidget(self.lbl_metric_agents)
        left_panel.addWidget(metrics_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 8))
        left_panel.addWidget(self.log_text)

        layout.addLayout(left_panel, stretch=1)

        self.map_view = HexMapView()
        layout.addWidget(self.map_view, stretch=3)

        self._toggle_ui(state="init")

    def _load_config(self):
        try:
            with open("config.json", "r") as f:
                cfg = json.load(f)
                self.input_url.setText(cfg.get("server_url", self.input_url.text()))
                self.input_team_id.setText(cfg.get("team_id", "1"))
                self.input_token.setText(cfg.get("token", ""))
        except FileNotFoundError:
            pass

    def log(self, text):
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def _toggle_ui(self, state):
        self.btn_connect.setEnabled(state == "init" or state == "connected")
        self.btn_auto_roles.setEnabled(state == "connected")
        self.btn_submit_roles.setEnabled(state == "connected")
        self.btn_fetch_day.setEnabled(state == "playing")
        self.btn_plan_day.setEnabled(state == "playing")
        self.btn_submit_day.setEnabled(state == "playing")


    def cmd_connect(self):
        self.client = Client(
            self.input_url.text(), self.input_game_id.text(),
            self.input_team_id.text(), self.input_token.text()
        )
        self.engine = Hexudon(self.client, practice=True)

        self.btn_connect.setEnabled(False)
        self.worker = Worker(self.engine.fetch_initial_state)
        self.worker.finished.connect(self.on_connected)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_connected(self, res):
        self.grid, self.spots, self.day_steps_list, self.fuel_limits, self.n_agents, self.agents_starts = res
        self.log(f"Connected! Map: {self.grid.width}x{self.grid.height}, Agents: {self.n_agents}\n")
        self.map_view.draw_map(self.grid, self.spots, [])
        self._build_agent_ui()
        self._toggle_ui("connected")

    def _build_agent_ui(self):
        for i in reversed(range(self.agents_inner_layout.count())):
            self.agents_inner_layout.itemAt(i).widget().setParent(None)

        self.agents_ui_rows = []
        for i in range(self.n_agents):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Agent {i}:"))
            cb = QComboBox()
            cb.addItems(["patrol", "refuel"])
            row.addWidget(cb)

            w = QWidget()
            w.setLayout(row)
            self.agents_inner_layout.addWidget(w)
            self.agents_ui_rows.append((str(i), cb))

    def cmd_auto_roles(self):
        total_steps = sum(self.day_steps_list) if self.day_steps_list else 50
        assignments = self.engine.auto_assign_agent_types(
            self.n_agents, self.fuel_limits, total_steps,
            self.agents_starts, self.grid
        )
        for aid, cb in self.agents_ui_rows:
            target = next(a["type"] for a in assignments if a["agent_id"] == aid)
            cb.setCurrentText(target)

    def cmd_submit_roles(self):
        assignments = [{"agent_id": aid, "type": cb.currentText()} for aid, cb in self.agents_ui_rows]
        self.type_assignments = {a["agent_id"]: a["type"] for a in assignments}

        self.worker = Worker(self.engine.submit_agent_types, assignments)
        self.worker.finished.connect(self.on_roles_submitted)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_roles_submitted(self, res):
        self.log("Roles submitted!\n")
        self.current_day = 0
        self._toggle_ui("playing")
        self.cmd_fetch_day()

    def cmd_fetch_day(self):
        self.lbl_day.setText(f"Day: {self.current_day} (Fetching...)")
        self.worker = Worker(self.engine.fetch_day_state, self.current_day)
        self.worker.finished.connect(self.on_day_fetched)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_day_fetched(self, state):
        self.state = state
        self.lbl_day.setText(f"Day: {self.current_day} (Ready to Plan)")

        team_data = state["teams"].get(str(self.client.team_id), {})
        raw_agents = team_data.get("agents", [])

        agents = []
        fuel_status = []
        for i, ag in enumerate(raw_agents):
            aid = str(ag.get("agent_id", i))
            pos = ag.get("pos", ag.get("cell"))
            fuel = ag.get("fuel", "-")
            t = self.type_assignments.get(aid, "patrol")
            agents.append({"agent_id": aid, "pos": pos, "type": t})
            fuel_status.append(f"Agent {aid} ({t[:1].upper()}): {fuel} fuel")

        self.map_view.draw_map(self.grid, self.spots, agents)

        self.lbl_metric_day.setText(f"Day: {state.get('day', self.current_day)}")
        distinct_brands = team_data.get("distinct_types", [])
        self.lbl_metric_brands.setText(f"Distinct Brands: {len(distinct_brands)}")
        self.lbl_metric_agents.setText("Agent Status:\n" + "\n".join(fuel_status))

    def cmd_plan_day(self):
        if not self.state: return
        self.worker = Worker(
            self.engine.generate_day_plan,
            self.current_day, self.state, self.grid, self.spots,
            self.day_steps_list, self.n_agents, self.type_assignments, "lazy"
        )
        self.worker.finished.connect(self.on_plan_generated)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_plan_generated(self, res):
        actions, patrol_agents, refuel_agents, brands_seen = res

        lines = ["[\n"]
        for i, a in enumerate(actions):
            lines.append("  " + json.dumps(a) + ("," if i < len(actions)-1 else "") + "\n")
        lines.append("]")
        json_str = "".join(lines)

        self.input_plan_json.setText(json_str)
        self.log("Plan generated (Lazy strategy). Ready to submit.\n")
        self.lbl_day.setText(f"Day: {self.current_day} (Plan Ready)")
        self.cmd_visualize_plan()

    def cmd_visualize_plan(self):
        try:
            actions = json.loads(self.input_plan_json.toPlainText())

            if not self.state: return
            team_data = self.state["teams"].get(str(self.client.team_id), {})
            raw_agents = team_data.get("agents", [])
            agents = []
            for i, ag in enumerate(raw_agents):
                aid = str(ag.get("agent_id", i))
                pos = ag.get("pos", ag.get("cell"))
                t = self.type_assignments.get(aid, "patrol")
                agents.append({"agent_id": aid, "pos": pos, "type": t})

            self.map_view.draw_map(self.grid, self.spots, agents, planned_actions=actions)
            self.log("Map paths updated from JSON.\n")
        except Exception as e:
            self.log(f"JSON Parse Error: {e}\n")

    def cmd_submit_day(self):
        try:
            actions = json.loads(self.input_plan_json.toPlainText())
        except Exception as e:
            self.log(f"JSON Parse Error: {e}\n")
            return

        self.worker = Worker(self.engine.submit_day_plan, self.current_day, actions)
        self.worker.finished.connect(self.on_day_submitted)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_day_submitted(self, res):
        self.log(f"Day {self.current_day} submitted: {res}\n")
        self.input_plan_json.clear()
        self.current_day += 1

        if self.current_day < len(self.day_steps_list):
            self.cmd_fetch_day()
        else:
            self.log("Game Finished!\n")
            self._toggle_ui("init")

    def on_error(self, err):
        self.log(f"\n{err}\n")
        self.btn_connect.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DemoGUI()
    win.show()
    sys.exit(app.exec())

