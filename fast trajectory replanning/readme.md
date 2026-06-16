# assignment 1 — fast trajectory replanning

implements repeated A\* search for an agent navigating a partially observable gridworld, where blocked cells are discovered dynamically during traversal.

## problem

an agent must reach a target cell in a gridworld it doesn't fully know. it operates under a "freespace assumption" — treating unobserved cells as unblocked — and replans whenever its path is obstructed by newly discovered blocked cells.

## what's implemented

- **repeated forward A\*** — replans from current cell to target each time the path is blocked
- **repeated backward A\*** — searches from target to agent; reuses heuristic information across replanning steps
- **adaptive A\*** — updates h-values after each search using information from previous runs to improve future heuristic estimates
- gridworld generation with configurable obstacle density
- tie-breaking strategies (larger g-values preferred) and their effect on runtime

## key findings

backward A\* generally expands fewer nodes than forward A\* on these gridworlds. adaptive A\* further reduces expansions by learning better heuristics across replanning episodes. tie-breaking in favor of larger g-values consistently outperforms smaller g-value tie-breaking.

## stack

- Python
- report typeset in LaTeX
