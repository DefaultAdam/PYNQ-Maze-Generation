import pygame
import sys
import os
import numpy as np
import time
import random
from multiprocessing import Pool, cpu_count

# --- SCREEN CONFIGURATION ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 30

# --- MAZE & CPU CONFIGURATION ---
MAZE_DIMENSION = 16
TOTAL_MAZES_TO_GENERATE = 10240

# --- COLOURS - DARK THEME ---
DARK_BG = (25, 25, 25)
WALL_COLOR = (200, 200, 200)

# =================================================================
#                     CPU MAZE GENERATION
# =================================================================

def generate_single_maze(seed):
    """
    Generates a single maze using a randomised backtracker algorithm.
    This function is written in pure Python/NumPy to run on the CPU.
    """
    # all walls are initially present
    # wall bits: [left, bottom, right, top]
    maze_walls = np.full((MAZE_DIMENSION, MAZE_DIMENSION), 0b1111, dtype=np.uint8)
    
    # boolean array for the visited map for clarity and efficiency
    visited = np.zeros((MAZE_DIMENSION, MAZE_DIMENSION), dtype=bool)
    
    stack = []
    
    # random number generator (Linear Feedback Shift Register)
    lfsr = seed
    if lfsr == 0: lfsr = 1

    # start at (0,0)
    x, y = 0, 0
    visited[y, x] = True
    stack.append((x, y))

    while stack:
        x, y = stack[-1]
        
        # find unvisited neighbors
        neighbors = []
        if y > 0 and not visited[y - 1, x]: neighbors.append((x, y - 1)) # top
        if x < MAZE_DIMENSION - 1 and not visited[y, x + 1]: neighbors.append((x + 1, y)) # right
        if y < MAZE_DIMENSION - 1 and not visited[y + 1, x]: neighbors.append((x, y + 1)) # bottom
        if x > 0 and not visited[y, x - 1]: neighbors.append((x - 1, y)) # left

        if neighbors:
            # choose a random neighbor
            lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & 0xD0000001)
            next_x, next_y = neighbors[lfsr % len(neighbors)]

            # remove walls between current and next cell
            if next_x > x: # right
                maze_walls[y, x] &= ~np.uint8(0b0100)
                maze_walls[next_y, next_x] &= ~np.uint8(0b0001)
            elif next_x < x: # left
                maze_walls[y, x] &= ~np.uint8(0b0001)
                maze_walls[next_y, next_x] &= ~np.uint8(0b0100)
            elif next_y > y: # bottom
                maze_walls[y, x] &= ~np.uint8(0b0010)
                maze_walls[next_y, next_x] &= ~np.uint8(0b1000)
            else: # top
                maze_walls[y, x] &= ~np.uint8(0b1000)
                maze_walls[next_y, next_x] &= ~np.uint8(0b0010)

            visited[next_y, next_x] = True
            stack.append((next_x, next_y))
        else:
            stack.pop()
            
    return seed, maze_walls

# =================================================================
#                    PYGAME VISUALISATION
# =================================================================

def draw_maze(screen, maze_walls_np):
    """Draws a single maze from a numpy array of wall data."""
    rows, cols = maze_walls_np.shape
    cell_size = (min(SCREEN_WIDTH, SCREEN_HEIGHT) * 0.9) / rows
    x_offset = (SCREEN_WIDTH - cell_size * cols) / 2
    y_offset = (SCREEN_HEIGHT - cell_size * rows) / 2

    for r in range(rows):
        for c in range(cols):
            wall_data = int(maze_walls_np[r, c])
            x_pixel, y_pixel = c * cell_size + x_offset, r * cell_size + y_offset
            
            if (wall_data >> 3) & 1: pygame.draw.line(screen, WALL_COLOR, (x_pixel, y_pixel), (x_pixel + cell_size, y_pixel))
            if (wall_data >> 2) & 1: pygame.draw.line(screen, WALL_COLOR, (x_pixel + cell_size, y_pixel), (x_pixel + cell_size, y_pixel + cell_size))
            if (wall_data >> 1) & 1: pygame.draw.line(screen, WALL_COLOR, (x_pixel + cell_size, y_pixel + cell_size), (x_pixel, y_pixel + cell_size))
            if (wall_data >> 0) & 1: pygame.draw.line(screen, WALL_COLOR, (x_pixel, y_pixel + cell_size), (x_pixel, y_pixel))

# =================================================================
#                       MAIN APPLICATION
# =================================================================

def main():
    save_dir = "saved_mazes_cpu"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"CREATED DIRECTORY: {save_dir}")

    print("--- STARTING MAZE GENERATION ---")
    print(f"GENERATING {TOTAL_MAZES_TO_GENERATE} MAZES FOR THE AI TRAINING DATASET.")

    # --- unique random seeds for each maze ---
    seeds = [random.randint(1, 2**32 - 1) for _ in range(TOTAL_MAZES_TO_GENERATE)]
    
    start_time = time.time()
    
    # --- use multiprocessing to speed up ---
    num_processes = cpu_count()
    
    with Pool(processes=num_processes) as pool:
        # generate mazes in parallel
        results = pool.map(generate_single_maze, seeds)

    end_time = time.time()
    print("\n--- COMPLETED ---")
    print(f"GENERATED {len(results)} MAZES IN {end_time - start_time:.4f} SECONDS.")

    # --- save results ---
    print("SAVING MAZES AS .npy FILES")
    for seed, maze_data in results:
        filename = os.path.join(save_dir, f"maze_{seed}.npy")
        np.save(filename, maze_data)
    print(f"SUCCESSFULLY SAVED TO '{save_dir}' DIRECTORY.")

    # --- pygame visualisation ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("CPU-GENERATED MAZE")
    clock = pygame.time.Clock()

    maze_to_display = results[0][1]

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        screen.fill(DARK_BG)
        draw_maze(screen, maze_to_display)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
