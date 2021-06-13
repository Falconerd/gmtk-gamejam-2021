#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "./deps/lib/linmath.h"

/* TODO:
 * [X] setup window
 * [X] setup shaders
 * [X] ortho project a square
 * [X] render board
 * [X] render characters
 * [X] A pushes B
 * [X] A pulls B when > 2
 * tiles
 * [X] normal (no special properties)
 * [X] wall (cannot walk through, string must go around)
 * [X] water (A dies when walking in, B can be pushed in)
 * [ ] block (can be pushed into water to create a path)
 * [ ] ice (entity slides across)
 */

#define u8 uint8_t
#define u16 uint16_t
#define u32 uint32_t
#define f32 float
#define f64 double
#define i32 int32_t

#define LEFT 0
#define RIGHT 1
#define UP 2
#define DOWN 3

#define SCALE 5
#define WIDTH 384
#define HEIGHT 216

#define BOARD_TILE_SIZE 16
#define BOARD_OFFSET_X WIDTH / 2 - 4 * BOARD_TILE_SIZE
#define BOARD_OFFSET_Y HEIGHT / 2 - 4 * BOARD_TILE_SIZE

typedef enum entity_type {
	ENTITY_TYPE_NONE,
	ENTITY_TYPE_PLAYER_A,
	ENTITY_TYPE_PLAYER_B,
	ENTITY_TYPE_PLAYER_BOTH,
	ENTITY_TYPE_COLLECTABLE
} Entity_Type;

typedef enum tile_type {
	TILE_TYPE_NORMAL,
	TILE_TYPE_WALL,
	TILE_TYPE_WATER,
	TILE_TYPE_ICE,
	TILE_TYPE_GOAL
} Tile_Type;

typedef struct tile {
	Tile_Type type;
	u32 flags;
	Entity_Type entity;
} Tile;

typedef struct queue_item Queue_Item;
struct queue_item {
	Queue_Item *next;
	int data;
};

typedef struct {
	Queue_Item *head;
} Queue;

typedef struct bfs_result {
	int start;
	int found;
	int distance;
	int came_from[64];
	int path[4];
} BFS_Result;

typedef struct state {
	Tile tiles[64];
	int player_a_index;
	int player_b_index;
	char level_name[80];
	int chain_indices[2];
	int chain_visible[2];
	BFS_Result last_bfs;
} State;

static GLFWwindow *window;
static u32 shader;
static u32 square_vao;
static u32 square_vbo;
static u32 square_ebo;
static u32 line_vao;
static u32 line_vbo;
static mat4x4 projection;

static vec4 color_white = { 1.0f, 1.0f, 1.0f, 1.0f };
static vec4 color_black = { 0.0f, 0.0f, 0.0f, 1.0f };
static vec4 color_grey100 = { 0.1f, 0.1f, 0.1f, 1.0f };
static vec4 color_grey200 = { 0.2f, 0.2f, 0.2f, 1.0f };
static vec4 color_grey400 = { 0.4f, 0.4f, 0.4f, 1.0f };
static vec4 color_water = {0.1f, 0.4f, 0.4f, 1.0f};
static vec4 color_ice = {0.1f, 0.9f, 0.9f, 1.0f};
static vec4 color_orange = {1.0f, 0.55f, 0.1f, 1.0f};
static vec4 color_teal = {0.0f, 0.8f, 0.9f, 1.0f};
static vec4 color_salmon = {1.0f, 0.24f, 0.24f, 1.0f};
static vec4 color_green  = {0.0f, 1.0f, 0.0f, 1.0f};
static vec4 color_goal = {0.9f, 0.9f, 0.0f, 1.0f};

static State state = {0};

static void error_and_exit(int error, const char *message) {
	fprintf(stderr, "Error: %s\n", message);
	exit(-1);
}

static Queue_Item *enqueue(Queue *queue) {
	Queue_Item *item = malloc(sizeof(Queue_Item));
	item->next = NULL;
	if (queue->head == NULL) {
		queue->head = item;
	} else {
		// expensive add cause i'm tired and stupid
		Queue_Item *curr = queue->head;
		while (curr->next != NULL) {
			curr = curr->next;
		}
		curr->next = item;
	}
	return item;
}


static Queue_Item *dequeue(Queue *queue) {
	if (queue->head == NULL) {
		return NULL;
	}
	Queue_Item *temp = queue->head;
	queue->head = queue->head->next;
	return temp;
}

static char *read_file_into_buffer(const char *path) {
	FILE *fp = fopen(path, "rb");
	if (!fp)
		error_and_exit(-1, "Can't read file");
	fseek(fp, 0, SEEK_END);
	int length = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char *buffer = malloc((length+1) * sizeof(char));
	if (!buffer)
		error_and_exit(-1, "Can't allocate file buffer");
	fread(buffer, sizeof(char), length, fp);
	buffer[length] = 0;
	return buffer;
}

static void get_neighbours(int *n, int index) {
	n[0] = index % 8 ? index - 1 : -1;
	n[1] = index % 8 != 7 ? index + 1 : -1;
	n[2] = index <= 56 ? index + 8 : -1;
	n[3] = index >= 8 ? index - 8 : -1;
}

static BFS_Result bfs(int start, int goal) {
	BFS_Result result = {-1};
	result.start = start;
	Queue q = {0};
	Queue_Item *q_item = enqueue(&q);
	q_item->data = start;
	memset(result.came_from, -1, 64 * sizeof(int));

	while (q.head != NULL) {
		Queue_Item *item = dequeue(&q);
		int neighbours[4];
		get_neighbours(neighbours, item->data);
		for (int i = 0; i < 4; ++i) {
			int index = neighbours[i];
			if (index == -1)
				continue;
			if (state.tiles[index].type == TILE_TYPE_WALL)
				continue;
			if (result.came_from[index] == -1) {
				Queue_Item *new_item = enqueue(&q);
				new_item->data = index;
				result.came_from[index] = item->data;
			}
			if (goal == index) {
				result.found = index;
				break;
			}
		}
	}

	int i = 0;
	result.distance = -1;
	if (result.found >= 0) {
		int current = goal;
		while (current != start) {
			++result.distance;
			result.path[i++] = current;
			current = result.came_from[current];
		}
	}

	memcpy(&state.last_bfs, &result, sizeof(BFS_Result));

	return result;
}

static void load_level(const char *path) {
	char *level_data = read_file_into_buffer(path);

	int name_length = strlen(path);
	memcpy(state.level_name, path, name_length * sizeof(char));
	state.level_name[name_length] = 0;

	for (int row = 0; row < 8; ++row) {
		char *start = &level_data[row * 9];
		for (int col = 0; col < 8; ++col) {
			int index = (7 - row) * 8 + col;
			Tile *tile = &state.tiles[index];
			tile->type = TILE_TYPE_NORMAL;
			tile->entity = ENTITY_TYPE_NONE;
			switch (start[col]) {
			case '.': tile->type = TILE_TYPE_NORMAL; break;
			case '#': tile->type = TILE_TYPE_WALL; break;
			case ' ': tile->type = TILE_TYPE_WATER; break;
			case 'A': {
				tile->entity = ENTITY_TYPE_PLAYER_A;
				state.player_a_index = index;
			} break;
			case 'B': {
				tile->entity = ENTITY_TYPE_PLAYER_B;
				state.player_b_index = index;
			} break;
			case 'c': {
				tile->entity = ENTITY_TYPE_COLLECTABLE;
			} break;
			case 'X': tile->type = TILE_TYPE_GOAL; break;
			case '+': tile->type = TILE_TYPE_ICE; break;
			}
		}
	}

	// bfs to create chain
	BFS_Result result = bfs(state.player_a_index, state.player_b_index);

	// somehow could not find B...
	if (result.found == -1) {
		error_and_exit(-1, "Could not trace a path from A to B");
	} 

	int current = result.found;
	int i = 2;

	memset(state.chain_indices, -1, 2 * sizeof(int));
	memset(state.chain_visible, 1, 2 * sizeof(int));

	while (current != state.player_a_index) {
		if (i <= 1 && i >= 0) {
			state.chain_indices[i] = current;
		}
		current = result.came_from[current];
		--i;
	}
}

static void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
	glViewport(0, 0, width, height);
}

static int can_move(int direction, int index) {
	switch (direction) {
	case LEFT: {
		if (index % 8 == 0)
			break;
		Tile *left_tile = &state.tiles[index-1];
		if (left_tile->type == TILE_TYPE_WALL)
			break;
		return index - 1;
	} break;
	case RIGHT: {
		if (index % 8 == 7)
			break;
		Tile *right_tile = &state.tiles[index+1];
		if (right_tile->type == TILE_TYPE_WALL)
			break;
		return index + 1;
	} break;
	case UP: {
		if (index >= 56)
			break;
		Tile *up_tile = &state.tiles[index+8];
		if (up_tile->type == TILE_TYPE_WALL)
			break;
		return index + 8;
	} break;
	case DOWN: {
		if (index <= 7)
			break;
		Tile *down_tile = &state.tiles[index-8];
		if (down_tile->type == TILE_TYPE_WALL)
			break;
		return index - 8;
	} break;
	}

	return -1;
}

static void try_move(int direction, int index) {
	int last_a = state.player_a_index;
	int last_b = state.player_b_index;
	int new_index = can_move(direction, index);
	if (new_index >= 0) {
		switch (state.tiles[index].entity) {
		case ENTITY_TYPE_PLAYER_A: {
			// if pushing B
			if (state.tiles[new_index].entity == ENTITY_TYPE_PLAYER_B) {
				// if riding B
				if (state.tiles[new_index].type == TILE_TYPE_WATER) {
					state.tiles[new_index].entity = ENTITY_TYPE_PLAYER_BOTH;
					state.tiles[index].entity = ENTITY_TYPE_NONE;
					state.player_a_index = new_index;
					state.player_b_index = new_index;
				} else {
					int new_b_index = can_move(direction, new_index);
					if (new_b_index >= 0) {
						state.tiles[new_b_index].entity = ENTITY_TYPE_PLAYER_B;
						state.player_b_index = new_b_index;
						state.tiles[new_index].entity = ENTITY_TYPE_PLAYER_A;
						state.tiles[index].entity = ENTITY_TYPE_NONE;
						state.player_a_index = new_index;
					}
				}
			} else {
				state.tiles[index].entity = ENTITY_TYPE_NONE;
				state.tiles[new_index].entity = ENTITY_TYPE_PLAYER_A;
				state.player_a_index = new_index;
			}

		} break;
		case ENTITY_TYPE_PLAYER_BOTH: {
			state.player_a_index = new_index;
			state.tiles[new_index].entity = ENTITY_TYPE_PLAYER_A;
			state.tiles[index].entity = ENTITY_TYPE_PLAYER_B;
		} break;
		default: break;
		}
	}

	// pull chain
	BFS_Result r = bfs(state.player_a_index, state.player_b_index);
	if (r.distance > 2) {
		state.chain_indices[0] = r.path[3];
		state.chain_indices[1] = r.path[2];
		state.tiles[state.player_b_index].entity = ENTITY_TYPE_NONE;
		state.tiles[r.path[1]].entity = ENTITY_TYPE_PLAYER_B;
		state.player_b_index = r.path[1];
		state.chain_visible[0] = 1;
		state.chain_visible[1] = 1;
	} else {
		if (r.distance == 0) {
			state.chain_visible[0] = 0;
			state.chain_visible[1] = 0;
		} else if (r.distance == 1) {
			state.chain_indices[1] = r.path[1];
			state.chain_visible[0] = 0;
			state.chain_visible[1] = 1;
		} else if (r.distance == 2) {
			state.chain_indices[0] = r.path[2];
			state.chain_indices[1] = r.path[1];
			state.chain_visible[0] = 1;
			state.chain_visible[1] = 1;
		}
	}

	// game over
	if (state.tiles[state.player_a_index].type == TILE_TYPE_WATER && state.player_a_index != state.player_b_index) {
		load_level(state.level_name);
	}
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);

	if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
		try_move(LEFT, state.player_a_index);
	} else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
		try_move(RIGHT, state.player_a_index);
	} else if (key == GLFW_KEY_UP && action == GLFW_PRESS) {
		try_move(UP, state.player_a_index);
	} else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
		try_move(DOWN, state.player_a_index);
	}
}

static void setup_window() {
	glfwSetErrorCallback(error_and_exit);
	if (!glfwInit()) {
		error_and_exit(-1, "Failed to init GLFW");
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

	window = glfwCreateWindow(WIDTH * SCALE, HEIGHT * SCALE, "Puzzle game", NULL, NULL);
	if (!window) {
		error_and_exit(-1, "Failed to create window");
	}

	glfwSetKeyCallback(window, key_callback);

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		error_and_exit(-1, "Failed to init GLAD");
	}

	glViewport(0, 0, WIDTH * SCALE, HEIGHT * SCALE);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
}

static void setup_rendering() {
	f32 square_vertices[] = {
		 0.5f,  0.5f, 0.0f,
		 0.5f, -0.5f, 0.0f,
		-0.5f, -0.5f, 0.0f,
		-0.5f,  0.5f, 0.0f
	};
	u32 square_indices[] = {
		0, 1, 3,
		1, 2, 3
	};
	glGenVertexArrays(1, &square_vao);
	glGenBuffers(1, &square_vbo);
	glGenBuffers(1, &square_ebo);

	glBindVertexArray(square_vao);
	glBindBuffer(GL_ARRAY_BUFFER, square_vao);
	glBufferData(GL_ARRAY_BUFFER, sizeof(square_vertices), square_vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, square_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(square_indices), square_indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(f32), NULL);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	f32 line_vertices[] = { 0, 0, 0, 1, 1, 1 };
	glGenVertexArrays(1, &line_vao);
	glGenBuffers(1, &line_vbo);

	glBindVertexArray(line_vao);
	glBindBuffer(GL_ARRAY_BUFFER, line_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(line_vertices), line_vertices, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(f32), NULL);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	mat4x4_ortho(projection, 0, WIDTH, 0, HEIGHT, -2.0f, 2.0f);
}

static void setup_shaders() {
	int success;
	char log[512];
	char *vertex_source = read_file_into_buffer("shader.vert");
	uint32_t vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, (const char *const *)&vertex_source, NULL);
	glCompileShader(vertex_shader);
	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vertex_shader, 512, NULL, log);
		error_and_exit(-1, log);
	}

	char *fragment_source = read_file_into_buffer("shader.frag");
	uint32_t fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, (const char *const *)&fragment_source, NULL);
	glCompileShader(fragment_shader);
	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragment_shader, 512, NULL, log);
		error_and_exit(-1, log);
	}

	shader = glCreateProgram();
	glAttachShader(shader, vertex_shader);
	glAttachShader(shader, fragment_shader);
	glLinkProgram(shader);
	glGetProgramiv(shader, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shader, 512, NULL, log);
		error_and_exit(-1, log);
	}
}

static void render_square(f32 x, f32 y, f32 width, f32 height, vec4 color) {
	mat4x4 model;
	mat4x4_identity(model);

	mat4x4_translate(model, x + width * 0.5f, y + height * 0.5f, 0.0f);
	mat4x4_scale_aniso(model, model, width, height, 1.0f);

	glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &model[0][0]);
	glUniform4fv(glGetUniformLocation(shader, "color"), 1, color);

	glBindVertexArray(square_vao);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

static void render_line(f32 x0, f32 y0, f32 x1, f32 y1, vec4 color) {
	mat4x4 model;
	mat4x4_identity(model);

	f32 vertices[] = {x0, y0, 0, x1, y1, 0};

	glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, &model[0][0]);
	glUniform4fv(glGetUniformLocation(shader, "color"), 1, color);

	glBindVertexArray(line_vao);
	glBindBuffer(GL_ARRAY_BUFFER, line_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
	glDrawArrays(GL_LINES, 0, 2);
}

static void render_entity(f32 x, f32 y, Entity_Type type) {
	switch (type) {
	case ENTITY_TYPE_NONE: break;
	case ENTITY_TYPE_PLAYER_A: render_square(x + 4, y + 4, BOARD_TILE_SIZE - 8, BOARD_TILE_SIZE - 8, color_orange); break;
	case ENTITY_TYPE_PLAYER_B: render_square(x + 2, y + 2, BOARD_TILE_SIZE - 4, BOARD_TILE_SIZE - 4, color_salmon); break;
	case ENTITY_TYPE_PLAYER_BOTH: {
		render_square(x + 2, y + 2, BOARD_TILE_SIZE - 4, BOARD_TILE_SIZE - 4, color_salmon);
		render_square(x + 4, y + 4, BOARD_TILE_SIZE - 8, BOARD_TILE_SIZE - 8, color_orange);
	} break;
	case ENTITY_TYPE_COLLECTABLE: {
		render_square(x + 6, y + 6, BOARD_TILE_SIZE / 4, BOARD_TILE_SIZE / 4, color_green);
	} break;
	}
}

static void render_chain() {
	for (int i = 0; i < 2; ++i) {
		if (state.chain_indices[i] != -1 && state.chain_visible[i]) {
			int index = state.chain_indices[i];
			int col = index % 8;
			int row = index / 8;
			render_square(
				BOARD_OFFSET_X + col * BOARD_TILE_SIZE + 6,
				BOARD_OFFSET_Y + row * BOARD_TILE_SIZE + 6,
				BOARD_TILE_SIZE / 4,
				BOARD_TILE_SIZE / 4,
				color_white
			);
		}
	}
}

static void render_bfs() {
	int index = state.last_bfs.found;
	fprintf(stdout, "%d -> %d [%d]\n", state.last_bfs.start, state.last_bfs.found, state.last_bfs.distance);
	if (index == -1)
		return;
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	for (;;) {
		int col = index % 8;
		int row = index / 8;
		render_square(
			BOARD_OFFSET_X + col * BOARD_TILE_SIZE + 6,
			BOARD_OFFSET_Y + row * BOARD_TILE_SIZE + 6,
			BOARD_TILE_SIZE / 4,
			BOARD_TILE_SIZE / 4,
			color_green
		);
		if (index == state.last_bfs.start) {
			break;
		}
		index = state.last_bfs.came_from[index];
	}
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

static void render_tile(int col, int row, Tile tile) {
	f32 x = BOARD_OFFSET_X + col * BOARD_TILE_SIZE;
	f32 y = BOARD_OFFSET_Y + row * BOARD_TILE_SIZE;
	switch (tile.type) {
	case TILE_TYPE_NORMAL: {
	} break;
	case TILE_TYPE_WATER: {
		render_square(x, y, BOARD_TILE_SIZE, BOARD_TILE_SIZE, color_water);
	} break;
	case TILE_TYPE_WALL: {
		render_square(x, y, BOARD_TILE_SIZE, BOARD_TILE_SIZE, color_white);
	} break;
	case TILE_TYPE_GOAL: {
		render_square(x, y, BOARD_TILE_SIZE, BOARD_TILE_SIZE, color_goal);
	} break;
	case TILE_TYPE_ICE: {
		render_square(x, y, BOARD_TILE_SIZE, BOARD_TILE_SIZE, color_ice);
	} break;
	}

	render_entity(x, y, tile.entity);
}

static void render_board() {
	for (int x = 0; x < 8; ++x) {
		for (int y = 0; y < 8; ++y) {
			render_square(
				BOARD_OFFSET_X + x * BOARD_TILE_SIZE,
				BOARD_OFFSET_Y + y * BOARD_TILE_SIZE,
				BOARD_TILE_SIZE,
				BOARD_TILE_SIZE,
				(y + x) % 2 == 0 ? color_grey100 : color_grey200
			);
			render_tile(x, y, state.tiles[y * 8 + x]);
		}
	}
}

static void render() {
	glfwPollEvents();
	glClearColor(0.2f, 0.0f, 0.2f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(shader);
	glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, &projection[0][0]);

	render_board();
	render_chain();
	render_score();

	glfwSwapBuffers(window);
}

int main(void) {
	setup_window();
	setup_rendering();
	setup_shaders();

	load_level("level1.dat");

	while (!glfwWindowShouldClose(window))
		render();
	return 0;
}
