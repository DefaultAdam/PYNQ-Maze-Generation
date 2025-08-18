import pygame
import numpy as np
import os
import sys
import time
import threading
import random
import heapq

# =================================================================
#                       HARDWARE IMPLEMENTATION
try:
    from pynq import Overlay, allocate
    PYNQ_AVAILABLE = True
except ImportError:
    print("WARNING: PYNQ libraries not found. Running in software-only simulation mode.")
    PYNQ_AVAILABLE = False
# =================================================================

# =================================================================
#                GUI SETTINGS FOR JUPYTER NOTEBOOKS
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
# =================================================================

# =================================================================
#                       MAZE SETTINGS
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 650
FPS = 60
MAZE_DIMENSION = 16
Q_TABLE_FILENAME = "q_table2.npy"
BITSTREAM_FILENAME = "14.bit"

REALTIME_LEARNING_RATE = 0.1
REALTIME_DISCOUNT_FACTOR = 0.9
REWARD_STEP_SUCCESS = 1.0
PENALTY_BACKTRACK = -2.0

DARK_BG = (25, 25, 25)
WALL_COLOR = (200, 200, 200)
START_COLOR = (200, 0, 0)
END_COLOR = (128, 0, 128)
PATH_COLOR = (0, 150, 255)
TEXT_COLOR = (220, 220, 220)
INFO_COLOR = (50, 200, 50)
ERROR_COLOR = (255, 50, 50)
SAVE_MSG_COLOR = (50, 200, 50)
HW_MODE_COLOR = (50, 150, 255)
SW_MODE_COLOR = (255, 150, 50)
CURRENT_CELL_COLOR = (0, 180, 0)
# =================================================================

# =================================================================
#                      PYNQ HARDWARE SETUP
pynq_overlay = None
maze_gens = []
dmas = []

def initialize_pynq_hardware():
    global pynq_overlay
    global maze_gens 
    global dmas

    if not PYNQ_AVAILABLE:
        return False
    try:
        print(f"Loading overlay: {BITSTREAM_FILENAME}...")
        pynq_overlay = Overlay(BITSTREAM_FILENAME)
        maze_gens = [pynq_overlay.maze_generator_0, pynq_overlay.maze_generator_1, pynq_overlay.maze_generator_2, pynq_overlay.maze_generator_3]
        dmas = [pynq_overlay.axi_maze_gen_0_output, pynq_overlay.axi_maze_gen_0_output1, pynq_overlay.axi_maze_gen_0_output2, pynq_overlay.axi_maze_gen_0_output3]
        print("PYNQ Overlay and all 4 hardware IPs loaded successfully.")
        return True
    except Exception as e:
        print(f"FATAL ERROR: Could not load PYNQ. {e}")
        return False

def run_single_hw_generator(index, dma_buffer, seed):
    generator_ip = maze_gens[index]
    dma_ip = dmas[index]
    generator_ip.write(0x10, seed)
    dma_ip.recvchannel.transfer(dma_buffer)
    generator_ip.write(0x00, 1)
    dma_ip.recvchannel.wait()

def generate_parallel_mazes_hw():
    if not PYNQ_AVAILABLE or not maze_gens:
        print("ERROR: PYNQ hardware not available.")
        return None, None, 0
    
    start_time = time.perf_counter()
    dma_buffers = [allocate(shape = (MAZE_DIMENSION, MAZE_DIMENSION), dtype = np.uint8) for _ in range(4)]
    threads = []
    seeds = [(int(time.time() * 1000) + i) % (2**32) or 1 for i in range(4)]
    print(f"Generating 4 new hardware mazes with seeds: {seeds}")
    
    for i in range(4):
        thread = threading.Thread(target = run_single_hw_generator, args = (i, dma_buffers[i], seeds[i]))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    end_time = time.perf_counter()
    duration = (end_time - start_time) * 1000
    print(f"All 4 hardware generators finished in {duration:.2f} ms.")
    results = []
    for buf in dma_buffers:
        results.append(np.copy(buf))
    return results, seeds, duration
# =================================================================

# =================================================================
#                     CELL FOR MAZE DRAWING
class Cell:
    def __init__(self, x_col, y_row):
        self.x_col, self.y_row = x_col, y_row
        self.walls = {'top': True, 'right': True, 'bottom': True, 'left': True}
        self.visited = False

    def get_neighbors(self, grid):
        neighbors = []
        if self.y_row > 0 and not grid[self.y_row - 1][self.x_col].visited: neighbors.append(grid[self.y_row - 1][self.x_col])
        if self.x_col < MAZE_DIMENSION - 1 and not grid[self.y_row][self.x_col + 1].visited: neighbors.append(grid[self.y_row][self.x_col + 1])
        if self.y_row < MAZE_DIMENSION - 1 and not grid[self.y_row + 1][self.x_col].visited: neighbors.append(grid[self.y_row + 1][self.x_col])
        if self.x_col > 0 and not grid[self.y_row][self.x_col - 1].visited: neighbors.append(grid[self.y_row][self.x_col - 1])

        if neighbors:
            return random.choice(neighbors)
        else:
            return None

def remove_walls(current, next_cell):
    dx = current.x_col - next_cell.x_col
    if dx == 1:
        current.walls['left'] = False
        next_cell.walls['right'] = False
    elif dx == -1:
        current.walls['right'] = False
        next_cell.walls['left'] = False

    dy = current.y_row - next_cell.y_row
    if dy == 1:
        current.walls['top'] = False
        next_cell.walls['bottom'] = False
    elif dy == -1:
        current.walls['bottom'] = False
        next_cell.walls['top'] = False

def convert_cell_grid_to_numpy(grid):
    numpy_array = np.zeros((MAZE_DIMENSION, MAZE_DIMENSION), dtype=np.uint8)
    for r in range(MAZE_DIMENSION):
        for c in range(MAZE_DIMENSION):
            cell = grid[r][c]
            val = 0
            if cell.walls['left']: val |= 0b0001
            if cell.walls['bottom']: val |= 0b0010
            if cell.walls['right']: val |= 0b0100
            if cell.walls['top']: val |= 0b1000
            numpy_array[r, c] = val
    return numpy_array

def get_next_pos(pos, action):
    r, c = pos
    if action == 0:
        return (r - 1, c)
    if action == 1:
        return (r, c + 1)
    if action == 2:
        return (r + 1, c)
    if action == 3:
        return (r, c - 1)
    return pos

def solve_and_learn(q_table, maze_np):
    start_time = time.perf_counter()
    start_pos = (0, 0)
    goal_pos = (MAZE_DIMENSION - 1, MAZE_DIMENSION - 1)
    
    path = [start_pos]
    visited = {start_pos}
    decision_points = {}

    while path[-1] != goal_pos:
        current_pos = path[-1]
        current_state = current_pos[0] * MAZE_DIMENSION + current_pos[1]
        
        r, c = current_pos
        wall_data = maze_np[r, c]
        valid_actions = []
        # Check each direction for valid moves based on wall bits
        # Up (0): wall_data >> 3 & 1 (top wall)
        if r > 0:
            if not ((wall_data >> 3) & 1):
                valid_actions.append(0)
        # Right (1): wall_data >> 2 & 1 (right wall)
        if c < MAZE_DIMENSION - 1:
            if not ((wall_data >> 2) & 1):
                valid_actions.append(1)
        # Down (2): wall_data >> 1 & 1 (bottom wall)
        if r < MAZE_DIMENSION - 1:
            if not ((wall_data >> 1) & 1):
                valid_actions.append(2)
        # Left (3): wall_data >> 0 & 1 (left wall)
        if c > 0:
            if not ((wall_data >> 0) & 1):
                valid_actions.append(3)

        q_values = {}
        for action in valid_actions:
            next_position = get_next_pos(current_pos, action)
            if next_position not in visited:
                q_values[action] = q_table[current_state, action]
        sorted_moves = sorted(q_values.keys(), key = lambda action: q_values[action], reverse = True)

        if sorted_moves:
            best_move = sorted_moves[0]
            decision_points[current_pos] = sorted_moves[1:]
            
            next_pos = get_next_pos(current_pos, best_move)
            path.append(next_pos)
            visited.add(next_pos)
        else:
            if not path:
                print("Solver Error: No path found.")
                return [], 0
            
            bad_move_pos = path.pop()
            last_decision_pos = path[-1]
            bad_action = [a for a, p in enumerate([get_next_pos(last_decision_pos, a) for a in range(4)]) if p == bad_move_pos][0]
            bad_state = last_decision_pos[0] * MAZE_DIMENSION + last_decision_pos[1]
            
            old_q = q_table[bad_state, bad_action]
            q_table[bad_state, bad_action] = old_q + REALTIME_LEARNING_RATE * (PENALTY_BACKTRACK - old_q)
            
            while not path or not decision_points.get(path[-1]):
                if not path: return [], 0
                path.pop()

            last_decision_pos = path[-1]
            next_best_move = decision_points[last_decision_pos].pop(0)
            next_pos = get_next_pos(last_decision_pos, next_best_move)
            path.append(next_pos)
            visited.add(next_pos)

        if len(path) > (MAZE_DIMENSION * MAZE_DIMENSION * 2):
            print("Solver Error: Path is too long, likely an issue.")
            return [], 0

    duration = (time.perf_counter() - start_time) * 1000
    print(f"Path found. Length: {len(path)}. Solve time: {duration:.2f} ms. Reinforcing path...")
    for i in range(len(path) - 1):
        state = path[i][0] * MAZE_DIMENSION + path[i][1]
        next_state_pos = path[i+1]
        action = [a for a, p in enumerate([get_next_pos(path[i], a) for a in range(4)]) if p == next_state_pos][0]
        
        old_q = q_table[state, action]
        next_max_q = np.max(q_table[next_state_pos[0] * MAZE_DIMENSION + next_state_pos[1]])
        new_q = old_q + REALTIME_LEARNING_RATE * (REWARD_STEP_SUCCESS + REALTIME_DISCOUNT_FACTOR * next_max_q - old_q)
        q_table[state, action] = new_q

    return path, duration

def draw_maze(screen, maze_np, path, x_offset, y_offset, maze_size, highlight_cell=None):
    cell_size = int((maze_size * 0.9) / MAZE_DIMENSION)
    maze_x_offset = x_offset + (maze_size - cell_size * MAZE_DIMENSION) / 2
    maze_y_offset = y_offset + (maze_size - cell_size * MAZE_DIMENSION) / 2

    for r, c in path:
        pygame.draw.rect(screen, PATH_COLOR, (c * cell_size + maze_x_offset, r * cell_size + maze_y_offset, cell_size, cell_size))
    
    if highlight_cell:
        r, c = highlight_cell.y_row, highlight_cell.x_col
        pygame.draw.rect(screen, CURRENT_CELL_COLOR, (c * cell_size + maze_x_offset, r * cell_size + maze_y_offset, cell_size, cell_size))

    pygame.draw.rect(screen, START_COLOR, (maze_x_offset, maze_y_offset, cell_size, cell_size))
    end_r, end_c = MAZE_DIMENSION - 1, MAZE_DIMENSION - 1
    pygame.draw.rect(screen, END_COLOR, (end_c * cell_size + maze_x_offset, end_r * cell_size + maze_y_offset, cell_size, cell_size))

    for r in range(MAZE_DIMENSION):
        for c in range(MAZE_DIMENSION):
            wall_data = int(maze_np[r, c])
            x_pixel, y_pixel = c * cell_size + maze_x_offset, r * cell_size + maze_y_offset
            if (wall_data >> 3) & 1: pygame.draw.line(screen, WALL_COLOR, (x_pixel, y_pixel), (x_pixel + cell_size, y_pixel))
            if (wall_data >> 2) & 1: pygame.draw.line(screen, WALL_COLOR, (x_pixel + cell_size, y_pixel), (x_pixel + cell_size, y_pixel + cell_size))
            if (wall_data >> 1) & 1: pygame.draw.line(screen, WALL_COLOR, (x_pixel + cell_size, y_pixel + cell_size), (x_pixel, y_pixel + cell_size))
            if (wall_data >> 0) & 1: pygame.draw.line(screen, WALL_COLOR, (x_pixel, y_pixel + cell_size), (x_pixel, y_pixel))

def draw_text(screen, text, position, font, color=TEXT_COLOR, center_x=False):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center_x: text_rect.centerx = position[0]
    else: text_rect.x = position[0]
    text_rect.y = position[1]
    screen.blit(text_surface, text_rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("FPGA/CPU AI Maze Co-Design")
    clock = pygame.time.Clock()
    title_font = pygame.font.Font(None, 48)
    info_font = pygame.font.Font(None, 28)
    
    if not os.path.exists(Q_TABLE_FILENAME):
        screen.fill(DARK_BG); draw_text(screen, f"ERROR: Q-table file not found: {Q_TABLE_FILENAME}", (50, 50), info_font, color=ERROR_COLOR); pygame.display.flip(); time.sleep(5); pygame.quit(); sys.exit()
    
    print(f"Loading trained Q-table from {Q_TABLE_FILENAME}..."); q_table = np.load(Q_TABLE_FILENAME); print("Q-table loaded successfully.")

    if not initialize_pynq_hardware() and PYNQ_AVAILABLE:
        screen.fill(DARK_BG); draw_text(screen, "FATAL: Could not initialize PYNQ Overlay.", (50, 100), info_font, color=ERROR_COLOR); pygame.display.flip(); time.sleep(5); pygame.quit(); sys.exit()

    game_mode = 'HARDWARE'
    hw_mazes, hw_seeds, hw_paths, hw_solve_times = [], [], [], []
    hw_gen_time = 0
    sw_grid, sw_stack, sw_current_cell, sw_path = None, None, None, []
    sw_gen_time, sw_solve_time = 0, 0
    sw_gen_start_time = 0
    message, save_message, save_message_timer = "", "", 0
    total_solved_count = 0

    def reset_hw_mode():
        nonlocal hw_mazes, hw_seeds, hw_paths, hw_gen_time, hw_solve_times, total_solved_count
        print("\n--- Generating and Solving 4 Hardware Mazes ---")
        hw_mazes, hw_seeds, hw_gen_time = generate_parallel_mazes_hw()
        hw_paths, hw_solve_times = [], []
        if hw_mazes:
            for maze in hw_mazes:
                path, solve_time = solve_and_learn(q_table, maze)
                hw_paths.append(path)
                hw_solve_times.append(solve_time)
                if path:
                    total_solved_count += 1
        
    def start_sw_mode():
        nonlocal sw_grid, sw_stack, sw_current_cell, sw_path, sw_gen_start_time, sw_gen_time, sw_solve_time
        print("\n--- Starting Software Maze Generation ---")
        sw_grid = [[Cell(c, r) for c in range(MAZE_DIMENSION)] for r in range(MAZE_DIMENSION)]
        sw_current_cell = sw_grid[0][0]
        sw_current_cell.visited = True
        sw_stack = [sw_current_cell]
        sw_path = []
        sw_gen_time, sw_solve_time = 0, 0
        sw_gen_start_time = time.perf_counter()

    reset_hw_mode()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h: game_mode = 'HARDWARE'; reset_hw_mode()
                elif event.key == pygame.K_s: game_mode = 'SOFTWARE'; start_sw_mode()
                elif event.key == pygame.K_p and game_mode == 'HARDWARE' and hw_mazes:
                    save_dir = "solved_mazes_pynq"
                    if not os.path.exists(save_dir): os.makedirs(save_dir)
                    print("\n--- SAVING MAZES ---")
                    for i, maze_np in enumerate(hw_mazes):
                        seed_val = hw_seeds[i]
                        filename = os.path.join(save_dir, f"maze_{seed_val}.npy")
                        print(f"Saving maze generated with seed {seed_val} as '{filename}'")
                        np.save(filename, maze_np)
                    save_message = f"Saved 4 mazes to '{save_dir}/'"; save_message_timer = time.time()
                elif event.key == pygame.K_u:
                    print(f"\n--- SAVING UPDATED Q-TABLE ---")
                    np.save(Q_TABLE_FILENAME, q_table)
                    save_message = f"AI knowledge updated and saved to {Q_TABLE_FILENAME}"; save_message_timer = time.time(); print(save_message)

        screen.fill(DARK_BG)
        draw_text(screen, "FPGA/CPU AI Maze Co-Design", (SCREEN_WIDTH // 2, 20), title_font, center_x=True)
        draw_text(screen, "[H] HW Mode | [S] SW Mode | [P] Save Mazes | [U] Update AI | [ESC] Quit", (SCREEN_WIDTH // 2, 65), info_font, center_x=True)

        if save_message and time.time() - save_message_timer < 3:
            draw_text(screen, save_message, (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30), info_font, color=SAVE_MSG_COLOR, center_x=True)
        
        maze_area_y_start = 110
        padding_ratio = 0.95

        if game_mode == 'HARDWARE':
            success_count = sum(1 for path in hw_paths if path)
            message = f"AI solved {success_count}/4. Total Solved: {total_solved_count}."
            draw_text(screen, "Hardware Mode: 4 Mazes Generated & Solved by AI", (SCREEN_WIDTH // 2, 95), info_font, color=HW_MODE_COLOR, center_x=True)
            
            available_width = SCREEN_WIDTH
            available_height = SCREEN_HEIGHT - maze_area_y_start
            
            single_maze_size = min(available_width / 2, available_height / 2) * padding_ratio

            total_grid_width = 2 * single_maze_size
            total_grid_height = 2 * single_maze_size

            grid_offset_x = (available_width - total_grid_width) / 2
            grid_offset_y = maze_area_y_start + (available_height - total_grid_height) / 2

            if hw_mazes:
                for i, maze_np in enumerate(hw_mazes):
                    row, col = i // 2, i % 2
                    x_off = grid_offset_x + col * single_maze_size
                    y_off = grid_offset_y + row * single_maze_size
                    path = hw_paths[i] if i < len(hw_paths) else []
                    draw_maze(screen, maze_np, path, x_off, y_off, single_maze_size)

        elif game_mode == 'SOFTWARE':
            draw_text(screen, "Software Mode: Maze Animated on CPU, Solved by AI", (SCREEN_WIDTH // 2, 95), info_font, color=SW_MODE_COLOR, center_x=True)
            
            if sw_stack:
                active_cell = sw_stack[-1]
                next_cell = active_cell.get_neighbors(sw_grid)
                if next_cell:
                    next_cell.visited = True
                    remove_walls(active_cell, next_cell)
                    sw_current_cell = next_cell
                    sw_stack.append(next_cell)
                else:
                    sw_current_cell = sw_stack.pop()
            elif not sw_path:
                if sw_gen_time == 0:
                    sw_gen_time = (time.perf_counter() - sw_gen_start_time) * 1000
                    print(f"Software maze generated in {sw_gen_time:.2f} ms. Solving with AI...")
                sw_maze_np = convert_cell_grid_to_numpy(sw_grid)
                sw_path, sw_solve_time = solve_and_learn(q_table, sw_maze_np)
                if sw_path: total_solved_count += 1
            
            available_width = SCREEN_WIDTH
            available_height = SCREEN_HEIGHT - maze_area_y_start
            
            display_size = min(available_width, available_height) * padding_ratio
            
            x_offset = (available_width - display_size) / 2
            y_offset = maze_area_y_start + (available_height - display_size) / 2

            sw_maze_to_draw = convert_cell_grid_to_numpy(sw_grid)
            draw_maze(screen, sw_maze_to_draw, sw_path, x_offset, y_offset, display_size, sw_current_cell if sw_stack else None)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
