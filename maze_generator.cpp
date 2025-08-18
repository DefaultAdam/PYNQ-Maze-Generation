#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>

// using 16x16 for speed
const int MAZE_WIDTH = 16;
const int MAZE_HEIGHT = 16;

// cell wall represented by 4 bits: [left, bottom, right, top]
typedef ap_uint<4> WallState;

struct Cell {
    ap_uint<8> x;
    ap_uint<8> y;
};

void maze_generator(
    ap_uint<32> seed,
    hls::stream<ap_axiu<8, 0, 0, 0>>& stream_out
) {
    #pragma HLS INTERFACE s_axilite port=seed bundle=control
    #pragma HLS INTERFACE axis port=stream_out
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // stores the wall state for each cell
    static WallState maze_walls[MAZE_HEIGHT][MAZE_WIDTH];
    static bool visited[MAZE_HEIGHT][MAZE_WIDTH];

    #pragma HLS ARRAY_PARTITION variable=maze_walls complete dim=2
    #pragma HLS ARRAY_PARTITION variable=visited complete dim=2

    static Cell stack[MAZE_HEIGHT * MAZE_WIDTH];
    ap_uint<16> stack_top = 0;
    ap_uint<32> lfsr = (seed == 0) ? ap_uint<32>(1) : seed;

    // initalises the maze with all walls up and unvisited
    // every cell starts with all 4 walls up (binary 1111) and unvisited
    INIT_LOOP_Y: for (int y = 0; y < MAZE_HEIGHT; y++) {
        INIT_LOOP_X: for (int x = 0; x < MAZE_WIDTH; x++) {
            #pragma HLS PIPELINE
            maze_walls[y][x] = 0b1111; // all walls present
            visited[y][x] = false;
        }
    }

    // --- MAZE GENERATION LOGIC ---
    Cell current_cell = {0, 0};
    visited[0][0] = true;
    stack[stack_top++] = current_cell;

    // --- LOOP LOGIC ---
    MAIN_GENERATION_LOOP: while (stack_top > 0) {
        #pragma HLS PIPELINE

        // pop a cell from the stack (current cell)
        current_cell = stack[--stack_top];

        Cell neighbors[4];
        ap_uint<3> neighbor_count = 0;

        // find unvisited neighbors
        if (current_cell.y > 0 && !visited[current_cell.y - 1][current_cell.x]) neighbors[neighbor_count++] = {current_cell.x, (ap_uint<8>)(current_cell.y - 1)}; // Top
        if (current_cell.x < MAZE_WIDTH - 1 && !visited[current_cell.y][current_cell.x + 1]) neighbors[neighbor_count++] = {(ap_uint<8>)(current_cell.x + 1), current_cell.y}; // Right
        if (current_cell.y < MAZE_HEIGHT - 1 && !visited[current_cell.y + 1][current_cell.x]) neighbors[neighbor_count++] = {current_cell.x, (ap_uint<8>)(current_cell.y + 1)}; // Bottom
        if (current_cell.x > 0 && !visited[current_cell.y][current_cell.x - 1]) neighbors[neighbor_count++] = {(ap_uint<8>)(current_cell.x - 1), current_cell.y}; // Left

        // if the current cell has any unvisited neighbors
        if (neighbor_count > 0) {
            // Push the current cell back onto the stack. We'll come back to it
            // if the new path leads to a dead end.
            stack[stack_top++] = current_cell;

            // choose a random neighbour
            lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xD0000001u);
            Cell next_cell = neighbors[lfsr % neighbor_count];
            
            // --- remove walls ---
            if (next_cell.x == current_cell.x + 1) { // right neighbour
                maze_walls[current_cell.y][current_cell.x][2] = 0; // remove current right wall
                maze_walls[next_cell.y][next_cell.x][0] = 0;   // remove next left wall
            } else if (next_cell.x == current_cell.x - 1) { // left neighbour
                maze_walls[current_cell.y][current_cell.x][0] = 0; // remove current left wall
                maze_walls[next_cell.y][next_cell.x][2] = 0;   // remove next right wall
            } else if (next_cell.y == current_cell.y + 1) { // bottom neighbour
                maze_walls[current_cell.y][current_cell.x][1] = 0; // remove current bottom wall
                maze_walls[next_cell.y][next_cell.x][3] = 0;   // remove next top wall
            } else { // top neighbour
                maze_walls[current_cell.y][current_cell.x][3] = 0; // remove current top wall
                maze_walls[next_cell.y][next_cell.x][1] = 0;   // remove next bottom wall
            }

            // mark the new cell as visited and push it to the stack.
            // it becomes the new current cell for the next iteration.
            visited[next_cell.y][next_cell.x] = true;
            stack[stack_top++] = next_cell;
        }
    }

    // --- STREAM WALL DATA OUT ---
    STREAM_OUT_LOOP_Y: for (int y = 0; y < MAZE_HEIGHT; y++) {
        STREAM_OUT_LOOP_X: for (int x = 0; x < MAZE_WIDTH; x++) {
            #pragma HLS PIPELINE
            ap_axiu<8, 0, 0, 0> val;
            val.data = maze_walls[y][x]; // send the 4-bit wall state
            val.keep = -1;
            val.last = ((y == MAZE_HEIGHT - 1) && (x == MAZE_WIDTH - 1));
            stream_out.write(val);
        }
    }
}
