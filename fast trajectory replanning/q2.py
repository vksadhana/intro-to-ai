"""
q2.py — Repeated Forward A* (Forward Replanning) with tie-breaking variants + Pygame visualization

Renders TWO views side-by-side:
- LEFT  : full (ground-truth) maze used for the run
- RIGHT : agent knowledge + search visualization

Controls:
- R : generate a new random maze and run again (max-g by default)
- 1 : run MAX-G on the current maze
- 2 : run MIN-G on the current maze
- L : load maze from text file (see readFile format) and run MAX-G
- ESC or close window : quit

Legend (colors):
GREY   = expanded / frontier / unknown (unseen)
PATH   = executed path
YELLOW = start + agent position
BLUE   = goal
WHITE  = known free
BLACK  = known blocked
"""

from __future__ import annotations

import heapq
import argparse
import json
import time
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import pygame
from constants import ROWS, START_NODE, END_NODE, BLACK, WHITE, GREY, YELLOW, BLUE, PATH, NODE_LENGTH, GRID_LENGTH, WINDOW_W, WINDOW_H, GAP
from custom_pq import CustomPQ_maxG, CustomPQ_minG


def readMazes(fname: str) -> List[List[List[int]]]:
    """
    Reads a JSON file containing a list of mazes.
    Each maze is a list of ROWS lists, each with ROWS int values (0=free, 1=blocked).
    Returns a list of maze[r][c] grids.
    """
    with open(fname, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    mazes = []
    for idx in range(len(data)):
        grid = data[idx]
        if len(grid) != ROWS or any(len(row) != ROWS for row in grid):
            raise ValueError(f"Maze {idx}: expected {ROWS}x{ROWS}, got {len(grid)}x{len(grid[0]) if grid else 0}")
        maze = []
        for row in grid:
            new_row = []
            for val in row:
                new_row.append(int(val))
            maze.append(new_row)
        maze[START_NODE[0]][START_NODE[1]] = 0
        maze[END_NODE[0]][END_NODE[1]] = 0
        mazes.append(maze)

    return mazes


def get_neighbors(r, c):
    neighbors = []
    if r - 1 >= 0:
        neighbors.append((r - 1, c))
    if r + 1 < ROWS:
        neighbors.append((r + 1, c))
    if c - 1 >= 0:
        neighbors.append((r, c - 1))
    if c + 1 < ROWS:
        neighbors.append((r, c + 1))
    return neighbors


def manhattan(a, b):
    row_diff = abs(a[0] - b[0])
    col_diff = abs(a[1] - b[1])
    return row_diff + col_diff


def observe(pos, actual_maze, known_blocked):
    r = pos[0]
    c = pos[1]
    all_neighbors = get_neighbors(r, c)
    for neighbor in all_neighbors:
        nr = neighbor[0]
        nc = neighbor[1]
        if actual_maze[nr][nc] == 1:
            known_blocked.add((nr, nc))


def single_astar(start, goal, known_blocked, tie_breaking):
    if tie_breaking == "max_g":
        open_list = CustomPQ_maxG()
    else:
        open_list = CustomPQ_minG()

    g_values = {}
    g_values[start] = 0

    came_from = {}
    came_from[start] = None

    closed_set = set()
    num_expanded = 0

    h_start = manhattan(start, goal)
    f_start = g_values[start] + h_start
    open_list.push(f_start, g_values[start], start)

    while open_list.is_empty() == False:
        popped = open_list.pop()
        f_val = popped[0]
        tiebreak_val = popped[1]
        current_cell = popped[2]

        if current_cell in closed_set:
            continue

        if current_cell == goal:
            path = []
            node = current_cell
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, num_expanded

        closed_set.add(current_cell)
        num_expanded = num_expanded + 1

        all_neighbors = get_neighbors(current_cell[0], current_cell[1])
        for neighbor in all_neighbors:
            if neighbor in known_blocked:
                continue
            if neighbor in closed_set:
                continue
            new_g = g_values[current_cell] + 1
            if neighbor not in g_values or new_g < g_values[neighbor]:
                g_values[neighbor] = new_g
                came_from[neighbor] = current_cell
                h_val = manhattan(neighbor, goal)
                f_val = new_g + h_val
                open_list.push(f_val, new_g, neighbor)

    return None, num_expanded


def repeated_forward_astar(
    actual_maze: List[List[int]],
    start: Tuple[int, int] = START_NODE,
    goal: Tuple[int, int] = END_NODE,
    tie_breaking: str = "max_g",
    visualize_callbacks: Optional[Dict[str, Callable[[Tuple[int, int]], None]]] = None,
) -> Tuple[bool, List[Tuple[int, int]], int, int]:

    current_pos = start
    executed_path = [start]
    known_blocked = set()
    total_expanded = 0
    total_replans = 0

    observe(current_pos, actual_maze, known_blocked)

    while current_pos != goal:
        path, num_exp = single_astar(current_pos, goal, known_blocked, tie_breaking)
        total_expanded = total_expanded + num_exp
        total_replans = total_replans + 1

        if path is None:
            return False, executed_path, total_expanded, total_replans

        found_block = False
        for i in range(1, len(path)):
            next_cell = path[i]
            observe(current_pos, actual_maze, known_blocked)

            if next_cell in known_blocked:
                found_block = True
                break

            current_pos = next_cell
            executed_path.append(current_pos)

            if visualize_callbacks is not None:
                if "on_move" in visualize_callbacks:
                    visualize_callbacks["on_move"](current_pos)

            if current_pos == goal:
                return True, executed_path, total_expanded, total_replans

        if found_block == False:
            break

    return True, executed_path, total_expanded, total_replans


def draw_grid(surface, actual_maze, known_blocked, known_free, executed_path, current_pos, goal, start, planned_path):
    surface.fill(BLACK)

    right_side_start = GRID_LENGTH + GAP

    for r in range(ROWS):
        for c in range(ROWS):
            y_pos = r * NODE_LENGTH
            x_left = c * NODE_LENGTH
            x_right = right_side_start + c * NODE_LENGTH

            if actual_maze[r][c] == 1:
                pygame.draw.rect(surface, BLACK, (x_left, y_pos, NODE_LENGTH, NODE_LENGTH))
            else:
                pygame.draw.rect(surface, WHITE, (x_left, y_pos, NODE_LENGTH, NODE_LENGTH))

            this_cell = (r, c)
            if this_cell in known_blocked:
                pygame.draw.rect(surface, BLACK, (x_right, y_pos, NODE_LENGTH, NODE_LENGTH))
            elif this_cell in known_free:
                pygame.draw.rect(surface, WHITE, (x_right, y_pos, NODE_LENGTH, NODE_LENGTH))
            else:
                pygame.draw.rect(surface, GREY, (x_right, y_pos, NODE_LENGTH, NODE_LENGTH))

    if planned_path is not None:
        for cell in planned_path:
            r = cell[0]
            c = cell[1]
            pygame.draw.rect(surface, PATH, (right_side_start + c * NODE_LENGTH, r * NODE_LENGTH, NODE_LENGTH, NODE_LENGTH))

    for cell in executed_path:
        r = cell[0]
        c = cell[1]
        pygame.draw.rect(surface, PATH, (c * NODE_LENGTH, r * NODE_LENGTH, NODE_LENGTH, NODE_LENGTH))
        pygame.draw.rect(surface, PATH, (right_side_start + c * NODE_LENGTH, r * NODE_LENGTH, NODE_LENGTH, NODE_LENGTH))

    sr = start[0]
    sc = start[1]
    pygame.draw.rect(surface, YELLOW, (sc * NODE_LENGTH, sr * NODE_LENGTH, NODE_LENGTH, NODE_LENGTH))
    pygame.draw.rect(surface, YELLOW, (right_side_start + sc * NODE_LENGTH, sr * NODE_LENGTH, NODE_LENGTH, NODE_LENGTH))

    gr = goal[0]
    gc = goal[1]
    pygame.draw.rect(surface, BLUE, (gc * NODE_LENGTH, gr * NODE_LENGTH, NODE_LENGTH, NODE_LENGTH))
    pygame.draw.rect(surface, BLUE, (right_side_start + gc * NODE_LENGTH, gr * NODE_LENGTH, NODE_LENGTH, NODE_LENGTH))

    cr = current_pos[0]
    cc = current_pos[1]
    pygame.draw.rect(surface, YELLOW, (cc * NODE_LENGTH, cr * NODE_LENGTH, NODE_LENGTH, NODE_LENGTH))
    pygame.draw.rect(surface, YELLOW, (right_side_start + cc * NODE_LENGTH, cr * NODE_LENGTH, NODE_LENGTH, NODE_LENGTH))

    pygame.display.flip()


def show_astar_search(win: pygame.Surface, actual_maze: List[List[int]], algo: str, fps: int = 240, step_delay_ms: int = 0, save_path: Optional[str] = None) -> None:
    if save_path is None:
        save_path = f"vis_{algo}.png"

    start = START_NODE
    goal = END_NODE
    current_pos = start
    executed_path = [start]
    known_blocked = set()
    known_free = set()
    clock = pygame.time.Clock()

    def observe_with_free(pos):
        r = pos[0]
        c = pos[1]
        all_neighbors = get_neighbors(r, c)
        for neighbor in all_neighbors:
            nr = neighbor[0]
            nc = neighbor[1]
            if actual_maze[nr][nc] == 1:
                known_blocked.add((nr, nc))
            else:
                known_free.add((nr, nc))

    observe_with_free(current_pos)

    keep_going = True
    while current_pos != goal and keep_going == True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keep_going = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    keep_going = False
                    break

        if keep_going == False:
            break

        path, exp = single_astar(current_pos, goal, known_blocked, algo)
        if path is None:
            break

        draw_grid(win, actual_maze, known_blocked, known_free, executed_path, current_pos, goal, start, path)
        clock.tick(fps)

        for i in range(1, len(path)):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    keep_going = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        keep_going = False
                        break

            if keep_going == False:
                break

            next_cell = path[i]
            observe_with_free(current_pos)

            if next_cell in known_blocked:
                break

            current_pos = next_cell
            executed_path.append(current_pos)

            draw_grid(win, actual_maze, known_blocked, known_free, executed_path, current_pos, goal, start, path)

            if step_delay_ms > 0:
                pygame.time.delay(step_delay_ms)

            clock.tick(fps)

            if current_pos == goal:
                break

    draw_grid(win, actual_maze, known_blocked, known_free, executed_path, current_pos, goal, start, None)
    pygame.image.save(win, save_path)
    print(f"Saved the visualization -> {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2: Repeated Forward A*")
    parser.add_argument("--maze_file", type=str, required=True,
                        help="Path to input JSON file containing a list of mazes")
    parser.add_argument("--output", type=str, default="results_q2.json",
                        help="Path to output JSON results file")
    parser.add_argument("--tie_braking", type=str, choices=["max_g", "min_g", "both"], default="both",
                        help="Tie-breaking variant to run (default: both)")
    parser.add_argument("--show_vis", action="store_true",
                        help="[Bonus] If set, show Pygame visualization for the selected maze")
    parser.add_argument("--maze_vis_id", type=int, default=0,
                        help="[Bonus] maze_id (index) 0 ... 49 among 50 grid worlds")
    parser.add_argument("--save_vis_path", type=str, default="q2-vis-max-g.png",
                        help="[Bonus] If set, save visualization to this PNG file")
    args = parser.parse_args()

    mazes = readMazes(args.maze_file)
    results = []

    for maze_id in tqdm(range(len(mazes)), desc="Processing mazes"):
        entry = {}
        entry["maze_id"] = maze_id

        if args.tie_braking == "max_g" or args.tie_braking == "both":
            t0 = time.perf_counter()
            found, executed, expanded, replans = repeated_forward_astar(
                actual_maze=mazes[maze_id],
                start=START_NODE,
                goal=END_NODE,
                tie_breaking="max_g"
            )
            t1 = time.perf_counter()
            entry["max_g"] = {
                "found": found,
                "path_length": len(executed) - 1 if found else -1,
                "expanded": expanded,
                "replans": replans,
                "runtime_ms": (t1 - t0) * 1000,
            }

        if args.tie_braking == "min_g" or args.tie_braking == "both":
            t0 = time.perf_counter()
            found, executed, expanded, replans = repeated_forward_astar(
                actual_maze=mazes[maze_id],
                start=START_NODE,
                goal=END_NODE,
                tie_breaking="min_g"
            )
            t1 = time.perf_counter()
            entry["min_g"] = {
                "found": found,
                "path_length": len(executed) - 1 if found else -1,
                "expanded": expanded,
                "replans": replans,
                "runtime_ms": (t1 - t0) * 1000,
            }

        results.append(entry)

    if args.show_vis:
        pygame.init()
        win = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Repeated Forward A* Visualization")
        clock = pygame.time.Clock()
        selected_maze = mazes[args.maze_vis_id]
        current_algo = "max_g"
        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
        running = True
        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_1:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_2:
                        current_algo = "min_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
            pygame.display.flip()
        pygame.quit()

    with open(args.output, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results for {len(results)} mazes written to {args.output}")


if __name__ == "__main__":
    main()