#The function creates an initial graph
#In this initial graph, nodes are set and randomly put anywhere for the game
#Edges are then calculated
#Adjacency list is then calculated off those edges
#the nodes, edges, and adj list are all returned then

import time
import pygame
import random
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import pandas as pd
import influence_maximization_algorithms as im
from plyer import notification



# Initialize Pygame
pygame.init()

# Constants
SCREEN_SIZE = (800, 600)
NUM_NODES = 50
NODE_RADIUS = 15
CONTROL_MARK_RADIUS = 5
LINE_THICKNESS = 2
FONT_SIZE = 24

NAVY = (0, 46, 93)
WHITE = (255, 255, 255)
ROYAL = (0, 61, 165)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0,0,0)

NODE_COLORS = [RED, BLUE]
CONTROL_MARK_COLORS = [WHITE, BLACK]

# Initialize screen and clock
screen = pygame.display.set_mode(SCREEN_SIZE, pygame.RESIZABLE)
pygame.display.set_caption("State of Drinking Nodes Modeled Per Night")
clock = pygame.time.Clock()
font = pygame.font.Font(None, FONT_SIZE)
highlighted_node = None

corner_radius = 10

def draw_rounded_rect(screen, color, rect, corner_radius):
    """Draw a rounded rectangle"""
    x, y, width, height = rect

    # Draw the main body of the rectangle
    pygame.draw.rect(screen, color, (x, y + corner_radius, width, height - 2 * corner_radius))
    pygame.draw.rect(screen, color, (x + corner_radius, y, width - 2 * corner_radius, height))

    # Draw the four rounded corners
    pygame.draw.circle(screen, color, (x + corner_radius, y + corner_radius), corner_radius)
    pygame.draw.circle(screen, color, (x + width - corner_radius, y + corner_radius), corner_radius)
    pygame.draw.circle(screen, color, (x + corner_radius, y + height - corner_radius), corner_radius)
    pygame.draw.circle(screen, color, (x + width - corner_radius, y + height - corner_radius), corner_radius)

def initialize_graph():
    # Initialize Nodes and Edges
    nodes = []

    # Scaled placement margins
    x_margin = int(SCREEN_SIZE[0] * 0.10)
    y_margin = int(SCREEN_SIZE[1] * 0.12)

    # Calculate the diagonal of the screen
    screen_diagonal = np.linalg.norm(np.array(SCREEN_SIZE))
    # Set edge threshold as a fraction of the diagonal
    edge_threshold = screen_diagonal * 0.095

    while len(nodes) < NUM_NODES:
        new_node = (random.randint(x_margin, SCREEN_SIZE[0] - x_margin), random.randint(y_margin, SCREEN_SIZE[1] - y_margin))
        if all(np.linalg.norm(np.array(new_node) - np.array(existing_node)) > 2 * NODE_RADIUS for existing_node in nodes):
            nodes.append(new_node)

    edges = [(i, j) for i in range(NUM_NODES) for j in range(i+1, NUM_NODES) if np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j])) < edge_threshold]

    # Create adjacency list
    adj_list = [[] for _ in range(NUM_NODES)]
    for i, j in edges:
        adj_list[i].append(j)
        adj_list[j].append(i)

    return nodes, edges, adj_list


def get_graph_laplacian(edges):
    # Create adjacency matrix
    adj_mat = np.zeros((NUM_NODES, NUM_NODES))
    for i, j in edges:
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1

    # Get degree matrix
    deg_mat = np.diag(np.sum(adj_mat, axis=1))

    return deg_mat - adj_mat  # Laplacian matrix

nodes, edges, adj_list = initialize_graph()

# Get Laplacian matrix
laplacian = get_graph_laplacian(edges)

opinions = [0 for _ in range(NUM_NODES)]
controls = [None for _ in range(NUM_NODES)]


# For buffering VIDEORESIZE events
latest_resize_event = None


def draw_run_one_night(screen, font):
    button_width, button_height = 150, 40
    button_x = 10
    button_y = 50

    # Draw the button
    draw_rounded_rect(screen, (200, 200, 200), (button_x, button_y, button_width, button_height), corner_radius)
    label = font.render("One night?", True, (0, 0, 0))
    screen.blit(label, (button_x + 30, button_y + 10))

    # Return the button's rectangle for click detection
    return (button_x, button_y, button_width, button_height)


# Game loop
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Check for button clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos

            # Check for clicks on the Reset button
            if reset_button_x <= x <= reset_button_x + reset_button_w and reset_button_y <= y <= reset_button_y + reset_button_h:
                nodes, edges, adj_list = initialize_graph()
                controls = [None for _ in range(NUM_NODES)]
                opinions = [0 for _ in range(NUM_NODES)]  # Reset opinions to neutral
                config = np.array([])

                nodes, edges, adj_list = initialize_graph()

                # Get Laplacian matrix 
                laplacian = get_graph_laplacian(edges)

                # Check to see if one night button is clicked (probably working) if i do this does it instantly update
            elif run_one_night_x <= event.pos[0] <= run_one_night_x + run_one_night_width and run_one_night_y <= event.pos[1] <= run_one_night_y + run_one_night_height:
                pass



        elif event.type == pygame.VIDEORESIZE:
            latest_resize_event = event
    if latest_resize_event and pygame.time.get_ticks() - resize_cooldown > 500:  # 500ms cool-down
        # Get new dimensions
        new_w, new_h = latest_resize_event.w, latest_resize_event.h

        # Calculate scaling factors
        x_scale = new_w / SCREEN_SIZE[0]
        y_scale = new_h / SCREEN_SIZE[1]

        # Update screen size constants
        SCREEN_SIZE = (new_w, new_h)

        # Scale node positions and other elements
        nodes = [(int(x * x_scale), int(y * y_scale)) for (x, y) in nodes]
        NODE_RADIUS = int(NODE_RADIUS * (x_scale + y_scale) / 2)  # average scaling factor for radius

        # Resize screen
        screen = pygame.display.set_mode((latest_resize_event.w, latest_resize_event.h), pygame.RESIZABLE)

        # Reset the cooldown timer
        resize_cooldown = pygame.time.get_ticks()
        latest_resize_event = None

        # Explicitly fill screen with a background color
        screen.fill(WHITE)
        pygame.display.flip()

    # Update Opinions
    new_opinions = []
    for i in range(NUM_NODES):
        neighbor_opinions = [opinions[j] for j in adj_list[i]]
        if controls[i] is not None:
            new_opinions.append(controls[i] * 2 - 1)
        elif neighbor_opinions:
            new_opinions.append(sum(neighbor_opinions) / len(neighbor_opinions))
        else:
            new_opinions.append(opinions[i])
    opinions = new_opinions

    # Draw UI
    screen.fill(WHITE)

    # Draw edges
    for edge in edges:
        pygame.draw.line(screen, (128, 128, 128), nodes[edge[0]], nodes[edge[1]], LINE_THICKNESS)

    # Draw nodes
    for i, (x, y) in enumerate(nodes):
        color = tuple(
            int((NODE_COLORS[0][j] * (1 - opinions[i]) + NODE_COLORS[1][j] * (1 + opinions[i])) / 2) for j in range(3))
        pygame.draw.circle(screen, color, (x, y), NODE_RADIUS)
        if controls[i] is not None:
            pygame.draw.circle(screen, CONTROL_MARK_COLORS[controls[i]], (x, y), CONTROL_MARK_RADIUS)

    # Drawing the Reset button
    reset_button_color = (200, 200, 200)  # Gray color
    reset_button_x, reset_button_y, reset_button_w, reset_button_h = 10, SCREEN_SIZE[1] - 60, 100, 40
    draw_rounded_rect(screen, reset_button_color, (reset_button_x, reset_button_y, reset_button_w, reset_button_h), 10)
    reset_text = font.render("Reset", True, (0, 0, 0))
    screen.blit(reset_text, (reset_button_x + 25, reset_button_y + 10))

    # Draw colorbar for the score
    score = sum(opinions) / len(opinions)
    bar_x_start, bar_y_start, bar_width, bar_height = 120, SCREEN_SIZE[1] - 60, SCREEN_SIZE[0] - 140, 20
    pygame.draw.rect(screen, NODE_COLORS[0], (bar_x_start, bar_y_start, bar_width // 2, bar_height))
    pygame.draw.rect(screen, NODE_COLORS[1], (bar_x_start + bar_width // 2, bar_y_start, bar_width // 2, bar_height))

    # Draw a vertical indicator based on the score
    indicator_x = bar_x_start + int((score + 1) / 2 * bar_width)
    pygame.draw.line(screen, (0, 0, 0), (indicator_x, bar_y_start), (indicator_x, bar_y_start + bar_height), 3)

    # Label the bar not using right now but maybe label DRINKING LIKELYHOOD
    #label_surface = font.render("Score", True, (0, 0, 0))
    #screen.blit(label_surface, (bar_x_start + bar_width // 2 - 20, bar_y_start - 30))

    # Display run one night button
    run_one_night_x, run_one_night_y, run_one_night_width, run_one_night_height = run_one_night_res = draw_run_one_night(screen, font)

    pygame.display.flip()
    clock.tick(10)

pygame.quit()