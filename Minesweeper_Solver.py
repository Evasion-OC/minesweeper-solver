import sys, random, logging
from collections import deque
from minesweeper.deductive import InconsistentObservation, Observation
from minesweeper.pipeline import STAGES, PlaySolver

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QSizePolicy, QScrollArea


"START logger"

logging.basicConfig(level=logging.INFO)

ai_logger = logging.getLogger('ai')

ai_logger.setLevel(logging.INFO)
ai_logger.propagate = False

"Design GUI logger QT"

class QTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)
    # logging.shutdown() at exit reads this attribute from every handler ever made; as a
    # class attribute it is found without touching the Qt object, which is gone by then.
    flushOnClose = False
    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        logging.Handler.__init__(self)
        self.text_edit = None
        self._alive = True
    def set_text_edit(self, text_edit):
        self.text_edit = text_edit
        self.log_signal.connect(self.append_log)
    def append_log(self, msg):
        if not self._alive or self.text_edit is None:
            return
        try:
            self.text_edit.append(msg)
            self.text_edit.verticalScrollBar().setValue(
                self.text_edit.verticalScrollBar().maximum()
            )
        except RuntimeError:
            # Underlying Qt object was destroyed (e.g. across a Spyder runfile reload)
            self._alive = False
    def emit(self, record):
        if not self._alive:
            return
        try:
            msg = self.format(record)
            self.log_signal.emit(msg)
        except RuntimeError:
            # The Qt object is gone, so ignore later records.
            self._alive = False
    def shutdown(self):
        self._alive = False

def _purge_stale_qtext_handlers(logger):
    """Remove QTextEditLogger handlers left by an earlier run in the same interpreter
    (Spyder keeps loggers between runs). Match by class name, since a reload rebinds
    the class."""
    for h in list(logger.handlers):
        if h.__class__.__name__ == 'QTextEditLogger':
            try:
                h.shutdown()
            except Exception:
                pass
            logger.removeHandler(h)

class LogWindow(QtWidgets.QMainWindow):
    closed_by_user = pyqtSignal()  # the title-bar close button: the game unticks its checkbox

    def __init__(self, logger, title="Log Window"):
        super().__init__()
        # Drop any handlers from a prior run before installing ours.
        _purge_stale_qtext_handlers(logger)
        self.setWindowTitle(title)
        self.setGeometry(150, 150, 600, 400)
        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        self.setCentralWidget(self.text_edit)
        self.qtext_handler = QTextEditLogger(self)
        self.qtext_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.qtext_handler.set_text_edit(self.text_edit)
        self.logger = logger
        self.start_logging()
        self.force_close = False
    def start_logging(self):
        if self.qtext_handler not in self.logger.handlers:
            self.logger.addHandler(self.qtext_handler)
            self.logger.info("Logging started.")
    def stop_logging(self):
        if self.qtext_handler in self.logger.handlers:
            self.logger.removeHandler(self.qtext_handler)
            self.logger.info("Logging stopped.")
    def closeEvent(self, event):
        if self.force_close:
            self.stop_logging()
            self.qtext_handler.shutdown()
            event.accept()
        else:
            self.stop_logging()
            self.hide()
            event.ignore()
            self.closed_by_user.emit()
    def force_close_window(self):
        self.force_close = True
        self.close()
        
"END Design GUI logger QT"

"END logger"

"DESIGNING BFS & RUNNING FLOODFILL ALGORITHM THROUGH IT"

def bfs_expand(board, r, c, adj, logger):
    """Flood-fill from a 0-clue cell. Caller is expected to have already revealed (r, c)."""
    if board[r][c]['clue'] != 0:
        return 0

    revealed_count = 0
    queue = deque([(r, c)])
    visited = set()

    while queue:
        curr_r, curr_c = queue.popleft()
        if (curr_r, curr_c) in visited:
            continue
        visited.add((curr_r, curr_c))

        cell = board[curr_r][curr_c]
        # Reveal if currently covered (and not flagged); count it
        if cell['covered'] and not cell['flagged']:
            cell['covered'] = False
            revealed_count += 1

        # Expand only through 0-clue cells
        if cell['clue'] == 0 and not cell['flagged']:
            for nr, nc in adj[curr_r][curr_c]:
                if (nr, nc) not in visited:
                    queue.append((nr, nc))

    logger.info(f"BFS expanded from ({r}, {c}), revealed {revealed_count} cells")
    return revealed_count


"INTEGRATION"

def generate_random_board(n_rows, n_cols, num_mines, first_move=None):
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    if first_move:
        dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        exset = set([first_move])
        fr, fc = first_move
        for dr, dc in dirs:
            rr, cc = fr + dr, fc + dc
            if 0 <= rr < n_rows and 0 <= cc < n_cols:
                exset.add((rr, cc))
        available = list(set(cells) - exset)
    else:
        available = cells[:]
    if num_mines > len(available):
        raise ValueError("Too many mines for the available cells.")
    mines = set(random.sample(available, num_mines))
    grid = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            row.append('*' if (r, c) in mines else '.')
        grid.append(row)
    return grid

def compute_clues(grid):
    dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    n_rows, n_cols = len(grid), len(grid[0])
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] == '*':
                continue
            count = 0
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols and grid[rr][cc] == '*':
                    count += 1
            grid[r][c] = str(count)

def create_board(grid):
    board = []
    for row in grid:
        rowdata = []
        for ch in row:
            if ch == '*':
                rowdata.append({'isMine': True, 'clue': -1, 'covered': True, 'flagged': False})
            else:
                rowdata.append({'isMine': False, 'clue': int(ch), 'covered': True, 'flagged': False})
        board.append(rowdata)
    return board

def build_adjacency(board):
    n_rows, n_cols = len(board), len(board[0])
    dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    adj = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            neighbors = []
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    neighbors.append((rr, cc))
            row.append(neighbors)
        adj.append(row)
    return adj


"UI DESIGN"

class DifficultyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Difficulty")
        layout = QtWidgets.QVBoxLayout(self)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["Beginner", "Intermediate", "Expert", "Extreme"])
        layout.addWidget(self.combo)
        btn_ok = QtWidgets.QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok)
        self.selected_difficulty = None
    def accept(self):
        self.selected_difficulty = self.combo.currentText().lower()
        super().accept()

class CellButton(QtWidgets.QPushButton):
    def __init__(self, row, col, parent=None):
        super().__init__(parent)
        self.row = row
        self.col = col
        self.setFixedSize(35,35)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QPushButton {
                background-color: #BDBDBD;
                border: 1px solid #999;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: white;
                border: 1px solid #CCC;
            }
        """)
        self.setFont(QtGui.QFont("Arial", 14))

def get_difficulty_params(difficulty):
    params = {
        'beginner': (8,8,10),
        'intermediate': (16,16,40),
        'expert': (16,30,99),
        'extreme': (25,50,375)
    }
    return params.get(difficulty, (8,8,10))

class MinesweeperWindow(QtWidgets.QMainWindow):
    
    def __init__(self, difficulty):
        super().__init__()
        self.difficulty = difficulty
        self.solver = PlaySolver()  # solved regions per game, moves per stage across games
        self.n_rows, self.n_cols, self.num_mines = get_difficulty_params(difficulty)
        self.init_game()
        self.ai_timer = QtCore.QTimer()
        self.ai_timer.timeout.connect(self.ai_step)
        self.fail_count = 0
        self.solved = False
        self.first_move_made = False

        self.init_ui()
        self.resize(min(self.n_cols * 40 + 50, 1280), min(self.n_rows * 40 + 150, 720))

    def init_game(self):
        grid = generate_random_board(self.n_rows, self.n_cols, self.num_mines)
        compute_clues(grid)
        self.board = create_board(grid)
        self.adj = build_adjacency(self.board)
        self.game_over = False
        self.first_move_made = False

    def init_ui(self):
        self.setWindowTitle(f"Minesweeper AI - {self.difficulty.capitalize()}")
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10,10,10,10)
        main_layout.setSpacing(10)
        
        control_panel = QtWidgets.QHBoxLayout()
        
        self.ai_button = QtWidgets.QPushButton("Start AI")
        self.ai_button.clicked.connect(self.toggle_ai)
        control_panel.addWidget(self.ai_button)
        reset_btn = QtWidgets.QPushButton("New Game")
        reset_btn.clicked.connect(lambda: self.reset_game(manual=True))
        control_panel.addWidget(reset_btn)

        self.ai_log_checkbox = QtWidgets.QCheckBox("Show AI Logs")
        self.ai_log_checkbox.stateChanged.connect(self.toggle_ai_logging)
        control_panel.addWidget(self.ai_log_checkbox)

        control_panel.addStretch()
        main_layout.addLayout(control_panel)
        
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_widget = QtWidgets.QWidget()
        scroll.setWidget(grid_widget)
        self.grid_layout = QtWidgets.QGridLayout(grid_widget)
        self.grid_layout.setSpacing(1)
        self.grid_layout.setContentsMargins(2,2,2,2)
        self.buttons = []
        for r in range(self.n_rows):
            row_buttons = []
            for c in range(self.n_cols):
                btn = CellButton(r, c, grid_widget)
                btn.clicked.connect(lambda _, rr=r, cc=c: self.reveal(rr, cc))
                row_buttons.append(btn)
                self.grid_layout.addWidget(btn, r, c)
            self.buttons.append(row_buttons)
        main_layout.addWidget(scroll)
        
        self.ai_log_window = LogWindow(ai_logger, "AI Logs")
        self.ai_log_window.closed_by_user.connect(lambda: self.ai_log_checkbox.setChecked(False))
        self.moves_made = 0
        self.flags_placed = 0
        self.time_elapsed = 0
        self.moves_label = QtWidgets.QLabel("Moves Made: 0")
        self.flags_label = QtWidgets.QLabel("Flags Placed: 0")
        self.mines_left_label = QtWidgets.QLabel(f"Mines Left: {self.num_mines}")
        self.time_label = QtWidgets.QLabel("Time Elapsed: 0s")
        control_panel.addWidget(self.moves_label)
        control_panel.addWidget(self.flags_label)
        control_panel.addWidget(self.mines_left_label)
        control_panel.addWidget(self.time_label)
        self.game_timer = QtCore.QTimer()
        self.game_timer.timeout.connect(self.update_time)
        self.game_timer.start(1000)
        self.update_ui()

    def toggle_ai_logging(self, state):
        if state == QtCore.Qt.Checked:
            self.ai_log_window.show()
            self.ai_log_window.start_logging()
        else:
            self.ai_log_window.stop_logging()
            self.ai_log_window.hide()

    def closeEvent(self, event):
        """Closing the game stops its timers, closes its log window and ends the application."""
        self.game_timer.stop()
        self.ai_timer.stop()
        self.ai_log_window.force_close_window()
        self.ai_log_window.close()
        event.accept()
        QtWidgets.QApplication.instance().quit()

    def update_counters(self):
        self.flags_placed = sum(cell['flagged'] for row in self.board for cell in row)
        self.flags_label.setText(f"Flags Placed: {self.flags_placed}")
        mines_left = max(self.num_mines - self.flags_placed, 0)
        self.mines_left_label.setText(f"Mines Left: {mines_left}")

    def update_ui(self):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                btn = self.buttons[r][c]
                cell = self.board[r][c]
                btn.setDisabled(cell['flagged'] or not cell['covered'])
                if cell['flagged']:
                    btn.setText('🚩')
                    btn.setStyleSheet("background-color: #BDBDBD; color: black;")
                elif not cell['covered']:
                    if cell['isMine']:
                        btn.setText('💣')
                        btn.setStyleSheet("background-color: red; color: white;")
                    else:
                        text = str(cell['clue']) if cell['clue'] > 0 else ''
                        btn.setText(text)
                        btn.setStyleSheet("background-color: white; color: black;")
                else:
                    btn.setText('')
                    btn.setStyleSheet("background-color: #BDBDBD; color: black;")
                btn.repaint()
        self.update_counters()

    def reveal(self, r, c):
        if self.game_over or self.board[r][c]['flagged']:
            return
        if not self.first_move_made:
            # The first cell revealed in a game, by a person or by the solver, deals the
            # board: no mine on it or next to it.
            self.first_move_made = True
            grid = generate_random_board(self.n_rows, self.n_cols, self.num_mines, first_move=(r, c))
            compute_clues(grid)
            self.board = create_board(grid)
            self.adj = build_adjacency(self.board)
        self.board[r][c]['covered'] = False
        self.moves_made += 1
        self.moves_label.setText(f"Moves Made: {self.moves_made}")
        self.update_counters()
        is_mine = self.board[r][c]['isMine']
        clue = self.board[r][c]['clue']
        if (not is_mine) and clue == 0:
            bfs_expand(self.board, r, c, self.adj, ai_logger)
        if self.check_loss():
            self.game_over = True
            self.reveal_all()
            QtWidgets.QMessageBox.critical(self, "Game Over", "You hit a mine!")
        elif self.check_win():
            self.game_over = True
            self._mark_remaining_mines_as_flagged()
            QtWidgets.QMessageBox.information(self, "Victory", "All mines cleared!")
        self.update_ui()

    def flag(self, r, c):
        """Flag a covered cell. Return False if it cannot be flagged."""
        if self.game_over or not self.board[r][c]['covered'] or self.board[r][c]['flagged']:
            return False
        self.board[r][c]['flagged'] = True
        self.update_counters()
        self.update_ui()
        return True

    def check_loss(self):
        return any((not cell['covered'] and cell['isMine']) for row in self.board for cell in row)

    def check_win(self):
        return all((cell['covered'] == cell['isMine']) for row in self.board for cell in row)

    def reveal_all(self):
        for row in self.board:
            for cell in row:
                cell['covered'] = False
        self.update_ui()

    def _mark_remaining_mines_as_flagged(self):
        """On a win, flag every mine not flagged yet, so the finished board shows where
        the mines were."""
        for row in self.board:
            for cell in row:
                if cell['isMine'] and not cell['flagged']:
                    cell['flagged'] = True

    def toggle_ai(self):
        if self.ai_timer.isActive():
            self.ai_timer.stop()
            self.ai_button.setText("Start AI")
        else:
            ai_logger.info("Starting the solver.")
            self.ai_timer.start(100)
            self.ai_button.setText("Stop AI")
        self.update_ui()

    def reset_game(self, manual=True):
        if manual:
            self.fail_count = 0
            self.solved = False
            self.ai_timer.stop()
            self.ai_button.setText("Start AI")
        self.init_game()
        self.solver.new_game()
        self.game_over = False
        self.first_move_made = False
        self.moves_made = 0
        self.time_elapsed = 0
        self.time_label.setText("Time Elapsed: 0s")
        self.moves_label.setText("Moves Made: 0")
        self.flags_label.setText("Flags Placed: 0")
        self.mines_left_label.setText(f"Mines Left: {self.num_mines}")
        self.update_counters()
        self.update_ui()

    def update_time(self):
        if not self.game_over:
            self.time_elapsed += 1
            self.time_label.setText(f"Time Elapsed: {self.time_elapsed}s")

    def _log_stage_counts(self):
        counts = self.solver.stage_counts
        ai_logger.info("Solver moves by stage so far: " + ", ".join(f"{stage} {counts[stage]}" for stage in STAGES))

    def ai_step(self):
        if self.game_over:
            # Nothing to play: the solver has won, or a person's click ended the game.
            self.ai_timer.stop()
            self.ai_button.setText("Start AI")
            ai_logger.info("The game is over. Press New Game to play again.")
            return
        try:
            if not self.first_move_made:
                # The solver opens on a random cell; reveal() deals the board around it.
                self.reveal(random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1))
            else:
                try:
                    observation = Observation.from_board(self.board, self.num_mines)
                    step = self.solver.step(observation)
                except InconsistentObservation as exc:
                    ai_logger.error(f"Deductive solver stopped: {exc}")
                    self.ai_timer.stop()
                    self.ai_button.setText("Start AI")
                    return
                if step is None:
                    ai_logger.info("Deductive solver has no covered cells to act on.")
                    self.ai_timer.stop()
                    self.ai_button.setText("Start AI")
                    return
                action = step.action
                row, column = action.cell
                ai_logger.info(f"Deductive solver ({step.stage}): {action.kind} at ({row}, {column}).")
                if action.kind == "flag":
                    if not self.flag(row, column):
                        ai_logger.error(
                            f"Deductive solver could not flag cell at ({row}, {column})."
                        )
                        self.ai_timer.stop()
                        self.ai_button.setText("Start AI")
                    return
                self.reveal(row, column)
            if self.game_over:
                self._log_stage_counts()
                if self.check_loss():
                    self.fail_count += 1
                    self.reset_game(manual=False)
                    ai_logger.info(f"Game lost. Attempt {self.fail_count}. Retrying...")
                else:
                    self.solved = True
                    self.ai_timer.stop()
                    self.ai_button.setText("Start AI")
                    ai_logger.info(f"Game won. Games lost before this one: {self.fail_count}.")
        except Exception as e:
            import traceback

            ai_logger.error(f"AI error: {str(e)}")
            ai_logger.error(traceback.format_exc())
            self.reset_game(manual=False)
            self.update_ui()

if __name__ == "__main__":
    def main_gui():
        app = QtWidgets.QApplication(sys.argv)
        diff_dialog = DifficultyDialog()
        if diff_dialog.exec_() == QtWidgets.QDialog.Accepted:
            difficulty = diff_dialog.selected_difficulty
            window = MinesweeperWindow(difficulty)
            window.show()
            sys.exit(app.exec_())
        else:
            app.quit()
    main_gui()
