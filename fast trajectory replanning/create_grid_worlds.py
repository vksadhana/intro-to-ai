"""
create_grid_worlds.py — Generate N random 101x101 mazes and save as mazes.json.

Usage:
    python create_grid_worlds.py [--num_mazes N] [--seed S] [--output FILE]
"""
import json
import random
import argparse
from constants import ROWS
from tqdm import tqdm

# set random seed for reproducibility
random.seed(42)


def create_maze():
    grid = []
    for r in range(ROWS):
        one_row = []
        for c in range(ROWS):
            one_row.append(None)
        grid.append(one_row)

    visited = []
    for r in range(ROWS):
        one_row = []
        for c in range(ROWS):
            one_row.append(False)
        visited.append(one_row)

    start_r = random.randint(0, ROWS - 1)
    start_c = random.randint(0, ROWS - 1)

    visited[start_r][start_c] = True
    grid[start_r][start_c] = 0

    stack = []
    stack.append((start_r, start_c))

    while True:
        if len(stack) == 0:
            found_unvisited = None
            for r in range(ROWS):
                for c in range(ROWS):
                    if visited[r][c] == False:
                        found_unvisited = (r, c)
                        break
                if found_unvisited is not None:
                    break

            if found_unvisited is None:
                break

            r = found_unvisited[0]
            c = found_unvisited[1]
            visited[r][c] = True
            grid[r][c] = 0
            stack.append((r, c))

        current_r = stack[-1][0]
        current_c = stack[-1][1]

        unvisited_neighbors = []
        if current_r - 1 >= 0:
            if visited[current_r - 1][current_c] == False:
                unvisited_neighbors.append((current_r - 1, current_c))
        if current_r + 1 < ROWS:
            if visited[current_r + 1][current_c] == False:
                unvisited_neighbors.append((current_r + 1, current_c))
        if current_c - 1 >= 0:
            if visited[current_r][current_c - 1] == False:
                unvisited_neighbors.append((current_r, current_c - 1))
        if current_c + 1 < ROWS:
            if visited[current_r][current_c + 1] == False:
                unvisited_neighbors.append((current_r, current_c + 1))

        if len(unvisited_neighbors) == 0:
            stack.pop()
            continue

        chosen = random.choice(unvisited_neighbors)
        next_r = chosen[0]
        next_c = chosen[1]

        visited[next_r][next_c] = True

        random_number = random.random()
        if random_number < 0.3:
            grid[next_r][next_c] = 1
        else:
            grid[next_r][next_c] = 0
            stack.append((next_r, next_c))

    grid[0][0] = 0
    grid[ROWS - 1][ROWS - 1] = 0

    return grid


def main():
    parser = argparse.ArgumentParser(description="Generate random mazes as JSON")
    parser.add_argument("--num_mazes", type=int, default=50,
                        help="Number of mazes to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="mazes.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    random.seed(args.seed)

    mazes = []
    for _ in tqdm(range(args.num_mazes), desc="Generating mazes"):
        one_maze = create_maze()
        mazes.append(one_maze)

    with open(args.output, "w") as fp:
        json.dump(mazes, fp)

    print(f"Generated {args.num_mazes} mazes (seed={args.seed}) -> {args.output}")


if __name__ == "__main__":
    main()