import numpy as np
import os
import random
import glob
import time
import pygame

# =================================================================
#                       PYGAME CONFIGURATION
PYGAME_CELL_SIZE = 40
PYGAME_WINDOW_DIMENSION = 16 * PYGAME_CELL_SIZE
PYGAME_BACKGROUND_COLOR = (20, 20, 20)
PYGAME_WALL_COLOR = (200, 200, 200)
PYGAME_AGENT_COLOR = (0, 150, 255)
PYGAME_GOAL_COLOR = (0, 255, 0)
PYGAME_PATH_COLOR = (50, 50, 70)
PYGAME_VISITED_COLOR = (100, 30, 30)
PYGAME_FPS = 60
# =================================================================


# =================================================================
#                       MAZE SOLVER CONFIGURATION
def manhattan_distance(pos1, pos2):
    """
    Calculates the Manhattan distance between two points.

    The Manhattan distance is the sum of the absolute differences of their Cartesian coordinates.

    Args:
        pos1 (tuple): The (x, y) coordinates of the first point.
        pos2 (tuple): The (x, y) coordinates of the second point.

    Returns:
        int: The Manhattan distance between pos1 and pos2.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

MAZE_DIMENSION = 16
MAZE_DIR = "saved_mazes_cpu" # directory to load mazes 
Q_TABLE_FILENAME = "q_table2.1.npy" # filename for the Q-table (Long-Term Memory)
MASTERED_LOG_FILENAME = "mastered_mazes1.2.log" # log file for mastered mazes

START_POS = (0, 0)
GOAL_POS = (MAZE_DIMENSION - 1, MAZE_DIMENSION - 1)
MAX_DIST = manhattan_distance(START_POS, GOAL_POS)

# --- SARSA parameters ---
""" state, action, reward, next_state, next_action """
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
MAX_EPISODES_PER_MAZE = 2000
SUCCESS_RATE_THRESHOLD = 0.95
STEPS_PER_EPISODE = MAZE_DIMENSION * MAZE_DIMENSION * 2

# --- steps per phase ---
EXPLOIT_PHASE_EPISODES = 1000

# --- UCB1 exploration strategy ---
""" exploration algorithm, the higher the value the more exploration """
EXPLORATION_CONSTANT_C = 2.5

# --- movement values ---
ACTIONS = [0, 1, 2, 3] # 0: up, 1: right, 2: down, 3: left

# --- reward/penalty values ---
REWARD_GOAL = 100
REWARD_CLOSER = 1
PENALTY_AWAY = -2
PENALTY_VISITED = -2.5
PENALTY_WALL = -10
# =================================================================

# =================================================================
#                       PYGAME FUNCTIONS
def draw_maze(screen, maze):
    """Draws the maze walls on the Pygame screen."""
    screen.fill(PYGAME_BACKGROUND_COLOR)
    for r in range(MAZE_DIMENSION):
        for c in range(MAZE_DIMENSION):
            wall_data = maze[r, c]
            x, y = c * PYGAME_CELL_SIZE, r * PYGAME_CELL_SIZE
            if (wall_data >> 3) & 1: # up
                pygame.draw.line(screen, PYGAME_WALL_COLOR, (x, y), (x + PYGAME_CELL_SIZE, y), 2)
            if (wall_data >> 2) & 1: # right
                pygame.draw.line(screen, PYGAME_WALL_COLOR, (x + PYGAME_CELL_SIZE, y), (x + PYGAME_CELL_SIZE, y + PYGAME_CELL_SIZE), 2)
            if (wall_data >> 1) & 1: # down
                pygame.draw.line(screen, PYGAME_WALL_COLOR, (x, y + PYGAME_CELL_SIZE), (x + PYGAME_CELL_SIZE, y + PYGAME_CELL_SIZE), 2)
            if (wall_data >> 0) & 1: # left
                pygame.draw.line(screen, PYGAME_WALL_COLOR, (x, y), (x, y + PYGAME_CELL_SIZE), 2)

def draw_agent_and_goal(screen, pos, path_history):
    """Draws the agent, goal, and the path taken."""
    # path
    for p in path_history:
        pygame.draw.rect(screen, PYGAME_PATH_COLOR, (p[1] * PYGAME_CELL_SIZE + 5, p[0] * PYGAME_CELL_SIZE + 5, PYGAME_CELL_SIZE - 10, PYGAME_CELL_SIZE - 10))

    # goal
    goal_x, goal_y = GOAL_POS[1] * PYGAME_CELL_SIZE, GOAL_POS[0] * PYGAME_CELL_SIZE
    pygame.draw.rect(screen, PYGAME_GOAL_COLOR, (goal_x + 10, goal_y + 10, PYGAME_CELL_SIZE - 20, PYGAME_CELL_SIZE - 20))

    # agent
    agent_x, agent_y = pos[1] * PYGAME_CELL_SIZE, pos[0] * PYGAME_CELL_SIZE
    pygame.draw.circle(screen, PYGAME_AGENT_COLOR, (agent_x + PYGAME_CELL_SIZE // 2, agent_y + PYGAME_CELL_SIZE // 2), PYGAME_CELL_SIZE // 3)


def load_mazes(maze_dir):
    """Loads all maze .npy files from the specified directory and returns them with their filenames."""
    if not os.path.exists(maze_dir):
        print(f"DIRECTORY '{maze_dir}' NOT FOUND. CREATING IT.")
        os.makedirs(maze_dir)
        # dummy maze to avoid errors
        print("CREATE TEMP FILE: 'maze_0.npy' AS NON EXIST")
        dummy_maze = np.zeros((MAZE_DIMENSION, MAZE_DIMENSION), dtype=int)
        np.save(os.path.join(maze_dir, "maze_0.npy"), dummy_maze)


    maze_files = sorted(glob.glob(os.path.join(maze_dir, "*.npy")))
    if not maze_files:
        raise FileNotFoundError(f"NO .npy FILES FOUND IN DIRECTORY: {maze_dir}")
    mazes = [(np.load(f), os.path.basename(f)) for f in maze_files]
    print(f"Loaded {len(mazes)} mazes.")
    return mazes
# =================================================================

# =================================================================
#                       MAZE SOLVER FUNCTIONS
def get_state_from_pos(pos):
    """Converts a (row, col) position to a single integer state."""
    return pos[0] * MAZE_DIMENSION + pos[1]

def get_valid_actions(maze, pos):
    """Returns a list of valid actions from a given position."""
    r, c = pos
    valid = []
    wall_data = maze[r, c]
    # bitwise checks for walls: 3: up, 2: right, 1: down, 0: left
    if r > 0 and not ((wall_data >> 3) & 1): valid.append(0) # up
    if c < MAZE_DIMENSION - 1 and not ((wall_data >> 2) & 1): valid.append(1) # right
    if r < MAZE_DIMENSION - 1 and not ((wall_data >> 1) & 1): valid.append(2) # down
    if c > 0 and not ((wall_data >> 0) & 1): valid.append(3) # left
    return valid

def get_next_pos(pos, action):
    """Calculates the next position given an action."""
    r, c = pos
    if action == 0: return (r - 1, c)
    if action == 1: return (r, c + 1)
    if action == 2: return (r + 1, c)
    if action == 3: return (r, c - 1)
    return pos

def choose_action_ucb(q_table, state, valid_actions, state_counts, state_action_counts, C):
    """Chooses an action using the UCB1 algorithm."""
    for action in valid_actions:
        if state_action_counts[state, action] == 0:
            return action

    ucb_values = {}
    total_state_visits = state_counts[state]
    
    for action in valid_actions:
        q_value = q_table[state, action]
        action_visits = state_action_counts[state, action]
        # add epsilon to avoid division by zero if action_visits is 0
        exploration_bonus = C * np.sqrt(np.log(total_state_visits + 1e-5) / (action_visits + 1e-5))
        ucb_values[action] = q_value + exploration_bonus
        
    return max(ucb_values, key = ucb_values.get)

def choose_action_greedy(q_table, state, valid_actions):
    """Chooses the best known action (pure exploitation)."""
    q_valid = {action: q_table[state, action] for action in valid_actions}
    return max(q_valid, key = q_valid.get)


def run_episode_sarsa(q_table, maze, mode, screen, clock, state_counts = None, state_action_counts = None):
    """Runs a single episode using the SARSA update rule with Pygame."""
    current_pos = START_POS
    done = False
    path_history = {current_pos}
    
    current_state = get_state_from_pos(current_pos)
    valid_actions = get_valid_actions(maze, current_pos)
    if not valid_actions: return False, 0, MAX_DIST
    
    if mode == 'exploit':
        current_action = choose_action_greedy(q_table, current_state, valid_actions)
    else:
        current_action = choose_action_ucb(q_table, current_state, valid_actions, state_counts, state_action_counts, EXPLORATION_CONSTANT_C)

    for step in range(1, int(STEPS_PER_EPISODE) + 1):
        # --- Pygame Visualisation ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        draw_maze(screen, maze)
        draw_agent_and_goal(screen, current_pos, path_history)
        pygame.display.flip()
        clock.tick(PYGAME_FPS)
        # --- End Visualisation ---

        if mode == 'explore':
            state_counts[current_state] += 1
            state_action_counts[current_state, current_action] += 1

        next_pos = get_next_pos(current_pos, current_action)
        
        if next_pos == GOAL_POS:
            reward = REWARD_GOAL
            done = True
        elif next_pos in path_history:
            reward = PENALTY_VISITED
        else:
            if manhattan_distance(next_pos, GOAL_POS) < manhattan_distance(current_pos, GOAL_POS):
                reward = REWARD_CLOSER
            else:
                reward = PENALTY_AWAY

        next_state = get_state_from_pos(next_pos)
        path_history.add(next_pos)

        next_valid_actions = get_valid_actions(maze, next_pos)
        if not next_valid_actions:
            q_table[current_state, current_action] += LEARNING_RATE * (reward - q_table[current_state, current_action])
            break

        if mode == 'exploit':
            next_action = choose_action_greedy(q_table, next_state, next_valid_actions)
        else:
            next_action = choose_action_ucb(q_table, next_state, next_valid_actions, state_counts, state_action_counts, EXPLORATION_CONSTANT_C)

        old_value = q_table[current_state, current_action]
        next_value = q_table[next_state, next_action]
        new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_value - old_value)
        q_table[current_state, current_action] = new_value

        current_state = next_state
        current_pos = next_pos
        current_action = next_action

        if done:
            break
            
    final_dist = manhattan_distance(current_pos, GOAL_POS)
    return done, step, final_dist
# =================================================================

# =================================================================
#                       MAIN TRAINING LOOP
def train():
    """Main training loop with Pygame visualisation."""
    pygame.init()
    screen = pygame.display.set_mode((PYGAME_WINDOW_DIMENSION, PYGAME_WINDOW_DIMENSION))
    pygame.display.set_caption("REINFORCED LEARNING ENVIRONMENT - MAZE SOLVER")
    clock = pygame.time.Clock()

    mazes_with_filenames = load_mazes(MAZE_DIR)
    num_states = MAZE_DIMENSION * MAZE_DIMENSION

    if os.path.exists(Q_TABLE_FILENAME):
        print(f"FOUND EXISTING Q-TABLE. LOADING FOR TRAINING.")
        q_table_long_term = np.load(Q_TABLE_FILENAME)
    else:
        print("NO Q-TABLE FOUND. CREATING A NEW ONE.")
        q_table_long_term = np.zeros((num_states, len(ACTIONS)))

    mastered_mazes = set()
    if os.path.exists(MASTERED_LOG_FILENAME):
        with open(MASTERED_LOG_FILENAME, 'r') as f:
            mastered_mazes = set(line.strip() for line in f)
        print(f"LOADED {len(mastered_mazes)} MASTERED MAZES FROM LOG FILE.")

    start_time = time.time()
    print(f"\n--- STARTING TRAINING ---")

    for maze_index, (maze, filename) in enumerate(mazes_with_filenames):
        
        if filename in mastered_mazes:
            continue

        print(f"\n======================================================================")
        print(f"ATTEMPTING MAZE {maze_index + 1}/{len(mazes_with_filenames)} (FILE: {filename})")
        print(f"======================================================================")
        
        q_table_working_copy = np.copy(q_table_long_term)
        is_mastered = False
        
        # --- phase 1: exploitation (already known knowledge in q-table) ---
        print("--- PHASE 1: EXPLOITATION ---")
        batch_outcomes, batch_steps, batch_final_dist = [], [], []
        for episode in range(1, EXPLOIT_PHASE_EPISODES + 1):
            done, steps, final_dist = run_episode_sarsa(q_table_working_copy, maze, 'exploit', screen, clock)
            batch_outcomes.append(1 if done else 0)
            if done: 
                batch_steps.append(steps)
            else: 
                batch_final_dist.append(final_dist)

            if episode % 100 == 0:
                success_rate = np.mean(batch_outcomes)
                print(f"EPISODES {episode-99}-{episode}: SUCCESS RATE: {success_rate:.0%}")
                if success_rate >= SUCCESS_RATE_THRESHOLD:
                    is_mastered = True
                    break

        # --- phase 2: exploration (if needed) ---
        if not is_mastered:
            print("--- USING UCB1 ---")
            state_action_counts = np.zeros((num_states, len(ACTIONS)))
            state_counts = np.zeros(num_states)
            
            batch_outcomes, batch_steps, batch_final_dist = [], [], []
            
            for episode in range(1, MAX_EPISODES_PER_MAZE - EXPLOIT_PHASE_EPISODES + 1):
                done, steps, final_dist = run_episode_sarsa(q_table_working_copy, maze, 'explore', screen, clock, state_counts = state_counts, state_action_counts = state_action_counts)
                batch_outcomes.append(1 if done else 0)
                if done: 
                    batch_steps.append(steps)
                else: 
                    batch_final_dist.append(final_dist)

                if episode % 100 == 0:
                    current_batch_outcomes = batch_outcomes[-100:]
                    success_rate = np.mean(current_batch_outcomes)
                    
                    current_batch_steps = [
                        s for i, s in enumerate(batch_steps) if batch_outcomes[i] == 1
                    ]
                    current_batch_fails = [
                        d for i, d in enumerate(batch_final_dist) if batch_outcomes[i] == 0
                    ]

                    avg_steps = (
                        np.mean(current_batch_steps) if current_batch_steps else 0
                    )
                    min_steps = (
                        min(current_batch_steps) if current_batch_steps else 0
                    )
                    avg_dist = (
                        np.mean(current_batch_fails) if current_batch_fails else 0
                    )
                    percent_to_goal = (
                        (1 - (avg_dist / MAX_DIST)) * 100 if avg_dist > 0 else 0
                    )

                    print(f"EPISODES {EXPLOIT_PHASE_EPISODES + episode-99}-{EXPLOIT_PHASE_EPISODES + episode}:  "
                          f"SUCCESS RATE: {success_rate:.0%} | "
                          f"AVG/MIN STEPS: {avg_steps:.0f}/{min_steps:.0f} | "
                          f"AVG% TO GOAL (FAILS): {percent_to_goal:.0f}%")
                    
                    if success_rate >= SUCCESS_RATE_THRESHOLD:
                        is_mastered = True
                        break
        
        # --- results ---
        if is_mastered:
            print(f"--- MAZE MASTERED. ADD TO LONG TERM KNOWLEDGE. ---")
            mastered_mazes.add(filename)
            q_table_long_term = np.copy(q_table_working_copy)
            np.save(Q_TABLE_FILENAME, q_table_long_term)
            with open(MASTERED_LOG_FILENAME, 'a') as f:
                f.write(f"{filename}\n")
            print(f" SAVED ")
        else:
            print(f"--- FAILED TO MASTER MAZE. DISCARDING KNOWLEDGE. ---")

    end_time = time.time()
    print("\n======================================================================")
    print("--- ALL MAZES PROCESSED. TRAINING COMPLETE. ---")
    print(f"Total training time: {end_time - start_time:.2f} seconds.")
    print(f"Final Q-table saved.")
    
    pygame.quit()
# =================================================================

if __name__ == "__main__":
    train()
