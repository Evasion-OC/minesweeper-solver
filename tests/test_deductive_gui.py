import logging
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

import Minesweeper_Solver as gui
from minesweeper.board import build_adjacency, compute_clues, create_board


class StubSignal:
    def connect(self, slot):
        self.slot = slot


class StubLogWindow:
    """Stands in for the GUI's log window, which most tests do not show."""

    def __init__(self, *args, **kwargs):
        self.closed = False
        self.closed_by_user = StubSignal()

    def force_close_window(self):
        pass

    def close(self):
        self.closed = True


class Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_gui_deductive_step_uses_verified_core(monkeypatch):
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    monkeypatch.setattr(gui, "LogWindow", StubLogWindow)
    window = gui.MinesweeperWindow("beginner")
    window.game_timer.stop()
    mine_cells = {(0, 1), (7, 7)} | {(6, column) for column in range(8)}
    grid = [
        ["*" if (row, column) in mine_cells else "." for column in range(8)]
        for row in range(8)
    ]
    compute_clues(grid)
    window.board = create_board(grid)
    window.adj = build_adjacency(window.board)
    window.first_move_made = True
    for row, column in ((0, 0), (1, 0), (1, 1)):
        window.board[row][column]["covered"] = False

    window.ai_step()
    assert window.board[0][1]["flagged"]
    assert window.solver.stage_counts == {"single clue": 1}  # the 1 at (0, 0) sees one covered cell
    assert not window.game_over
    window.ai_step()
    assert window.moves_made == 1
    assert not window.game_over
    window.ai_timer.stop()
    quits = []
    monkeypatch.setattr(QtWidgets.QApplication, "quit", lambda self: quits.append(True))
    window.close()
    assert window.ai_log_window.closed and quits == [True]
    assert not window.game_timer.isActive() and not window.ai_timer.isActive()
    assert application is not None


def test_gui_logs_each_move_by_stage_and_the_counts_per_game(monkeypatch):
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    monkeypatch.setattr(gui, "LogWindow", StubLogWindow)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)  # a lost game ends the loop too
    collect = Collect()
    gui.ai_logger.addHandler(collect)
    try:
        random.seed(3)
        window = gui.MinesweeperWindow("beginner")
        window.game_timer.stop()
        solver = window.solver
        for _ in range(3000):
            window.ai_step()
            window.ai_timer.stop()
            if window.solved or window.fail_count:
                break
        assert window.solved or window.fail_count  # one game played to the end
        moves = [m for m in collect.messages if m.startswith("Deductive solver (")]
        assert moves and all(m.split("(")[1].split(")")[0] in gui.STAGES for m in moves)
        assert len(moves) == sum(solver.stage_counts.values())
        assert sum(m.startswith("Solver moves by stage so far") for m in collect.messages) == 1
        window.reset_game(manual=True)
        assert window.solver is solver and len(solver.cache) == 0  # same solver, regions forgotten
        assert sum(solver.stage_counts.values()) == len(moves)  # counts kept
        window.close()
    finally:
        gui.ai_logger.removeHandler(collect)
    assert application is not None


def test_closing_the_game_closes_its_log_windows_and_ends_the_application(tmp_path):
    # Run as the GUI's __main__ does (the application lives in a function that ends with
    # sys.exit), with its log window open. Closing the game window must end the event
    # loop by itself, and nothing may be reported at interpreter exit.
    script = tmp_path / "close_game.py"
    script.write_text(textwrap.dedent(f"""
        import os, random, sys
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from PyQt5 import QtCore, QtWidgets
        import Minesweeper_Solver as gui

        def main():
            app = QtWidgets.QApplication(sys.argv)
            random.seed(1)
            window = gui.MinesweeperWindow("beginner")
            window.show()
            window.ai_log_checkbox.setChecked(True)
            for _ in range(5):
                window.ai_step()
            QtCore.QTimer.singleShot(100, window.close)
            QtCore.QTimer.singleShot(5000, lambda: app.exit(3))  # still running after the close
            sys.exit(app.exec_())

        main()
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Exception ignored" not in result.stderr


def quiet_window(monkeypatch, level):
    """A game window with its log window stubbed and its message boxes recorded, not shown."""
    boxes = []
    monkeypatch.setattr(gui, "LogWindow", StubLogWindow)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: boxes.append(args[2]))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: boxes.append(args[2]))
    window = gui.MinesweeperWindow(level)
    window.game_timer.stop()
    return window, boxes


def test_a_persons_first_click_is_never_a_mine_nor_next_to_one(monkeypatch):
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    random.seed(5)
    window, boxes = quiet_window(monkeypatch, "expert")
    for _ in range(100):  # without the protection about one first click in five on Expert is a mine
        window.reset_game(manual=True)
        r, c = random.randrange(window.n_rows), random.randrange(window.n_cols)
        window.buttons[r][c].click()
        assert not window.game_over and boxes == []
        assert window.board[r][c]["clue"] == 0
        assert all(not window.board[rr][cc]["covered"] for rr, cc in window.adj[r][c])
    window.close()
    assert application is not None


def test_start_ai_after_a_click_plays_on_from_the_same_board(monkeypatch):
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    random.seed(6)
    window, _ = quiet_window(monkeypatch, "beginner")
    window.buttons[3][3].click()
    layout = [[cell["isMine"] for cell in row] for row in window.board]
    uncovered = {(r, c) for r in range(8) for c in range(8) if not window.board[r][c]["covered"]}
    window.toggle_ai()
    window.ai_step()
    window.ai_timer.stop()
    assert [[cell["isMine"] for cell in row] for row in window.board] == layout
    assert all(not window.board[r][c]["covered"] for r, c in uncovered)
    assert window.moves_made + window.flags_placed == 2  # the person's click and the solver's move (reveal or flag)
    window.close()
    assert application is not None


def test_start_ai_after_a_lost_game_stops_and_says_so(monkeypatch):
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    random.seed(7)
    window, boxes = quiet_window(monkeypatch, "beginner")
    window.buttons[0][0].click()
    r, c = next((r, c) for r in range(8) for c in range(8) if window.board[r][c]["isMine"])
    window.reveal(r, c)
    assert window.game_over and boxes == ["You hit a mine!"]
    collect = Collect()
    gui.ai_logger.addHandler(collect)
    try:
        window.toggle_ai()
        window.ai_step()
    finally:
        gui.ai_logger.removeHandler(collect)
    assert not window.ai_timer.isActive() and window.ai_button.text() == "Start AI"
    assert "The game is over. Press New Game to play again." in collect.messages
    assert window.fail_count == 0 and boxes == ["You hit a mine!"]
    window.close()
    assert application is not None


def test_after_the_solver_wins_the_button_reads_start_ai_and_there_is_one_victory_box(monkeypatch):
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    random.seed(3)  # a game the solver wins without losing first
    window, boxes = quiet_window(monkeypatch, "beginner")
    window.toggle_ai()
    for _ in range(3000):
        window.ai_step()
        if window.solved:
            break
    assert window.solved and window.fail_count == 0
    assert not window.ai_timer.isActive() and window.ai_button.text() == "Start AI"
    collect = Collect()
    gui.ai_logger.addHandler(collect)
    try:
        window.toggle_ai()  # pressing Start AI on a finished game plays nothing, shows nothing and says why
        window.ai_step()
    finally:
        gui.ai_logger.removeHandler(collect)
    assert not window.ai_timer.isActive() and window.ai_button.text() == "Start AI"
    assert collect.messages[-1] == "The game is over. Press New Game to play again."
    assert boxes == ["All mines cleared!"]
    window.close()
    assert application is not None


def test_new_game_resets_the_time_shown(monkeypatch):
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window, _ = quiet_window(monkeypatch, "beginner")
    for _ in range(5):
        window.update_time()
    assert window.time_label.text() == "Time Elapsed: 5s"
    window.reset_game(manual=True)
    assert window.time_label.text() == "Time Elapsed: 0s"
    window.close()
    assert application is not None


def test_closing_the_log_window_unticks_its_box(monkeypatch):
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(QtWidgets.QApplication, "quit", lambda self: None)
    window = gui.MinesweeperWindow("beginner")
    window.game_timer.stop()
    window.ai_log_checkbox.setChecked(True)
    assert window.ai_log_window.isVisible()
    window.ai_log_window.close()  # the title-bar close button
    assert not window.ai_log_window.isVisible() and not window.ai_log_checkbox.isChecked()
    window.ai_log_checkbox.setChecked(True)  # one tick shows it again
    assert window.ai_log_window.isVisible()
    window.close()
    assert application is not None
