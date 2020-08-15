import cv2
import csv
import json
from PIL import Image

class Node():

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar(maze, start, end):

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append([x, y])

def detect_corners(filename):
    global corners
    corners = []

    while True:
        cv2.namedWindow('aerial', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('aerial', filename)
        cv2.setMouseCallback('aerial', on_click)

        k = cv2.waitKey(16) & 0xFF
        if k == 13: # enter
            break

    cv2.destroyAllWindows()
    return corners

def get_grid_centers(filename):
    coords = []
    im = Image.open(filename.split('.')[0] + '.jpg')
    width, height = im.size

    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=' ')
        for row in csv_reader:
            xmin = round(float(row[1]) * width, 6)
            ymin = round(float(row[2]) * height, 6)
            w = float(row[3]) * width
            h = float(row[4]) * height
            coords.append( (xmin, ymin, w, h) )

    count = 0
    grid_centers = {}

    for row in range(5):
        for col in range(5):
            if row % 2 == 0 and col % 2 != 0:
                continue
            grid_centers[coords[count]] = (row, col)
            count += 1
                
    return grid_centers

def get_objects(detection):
    objects = []

    # with open(filename, 'r') as f:
    #     result = json.load(f)[0]

    im = Image.open(detection["filename"])
    width, height = im.size

    for o in detection["objects"]:
        cat_id = o["class_id"] + 1
        xmin = round((o["relative_coordinates"]["center_x"]) * width, 6)
        ymin = round((o["relative_coordinates"]["center_y"]) * height, 6)
        w = o["relative_coordinates"]["width"] * width
        h = o["relative_coordinates"]["height"] * height
        objects.append({"category_id":cat_id, "bbox": [xmin, ymin, w, h]})

    return objects

def iou(r1, r2):
    r1_xmin = r1[1] - r1[3] / 2
    r1_xmax = r1[1] + r1[3] / 2
    r1_ymin = r1[0] - r1[2] / 2
    r1_ymax = r1[0] + r1[2] / 2

    r2_xmin = r2[1] - r2[3] / 2
    r2_xmax = r2[1] + r2[3] / 2
    r2_ymin = r2[0] - r2[2] / 2
    r2_ymax = r2[0] + r2[2] / 2

    dx = min(r1_xmax, r2_xmax) - max(r1_xmin, r2_xmin)
    dy = min(r1_ymax, r2_ymax) - max(r1_ymin, r2_ymin)
    intersection = dx * dy

    r1_area = (r1_xmax - r1_xmin) * (r1_ymax - r1_ymin)
    r2_area = (r2_xmax - r2_xmin) * (r2_ymax - r2_ymin)
    union = r1_area + r2_area - intersection
    return max(0, intersection/union)

def get_start_end(grid_centers, objects):
    top_grids = list(grid_centers)[:3]
    bottom_grids = list(grid_centers)[-3:]
    start = None
    end = None

    for top_grid in top_grids:
        for o in objects:
            if o["category_id"] == 3 and iou(top_grid, o["bbox"]) > 0.3:
                print('end found')
                end = grid_centers[top_grid]

    for bottom_grid in bottom_grids:
        for o in objects:
            if o["category_id"] == 2 and iou(bottom_grid, o["bbox"]) > 0.3:
                print('start found')
                start = grid_centers[bottom_grid]

    return start, end

def add_obstacles(grid_centers, objects, maze):
    grids = list(grid_centers)[3:-3]

    for coord in grids:
        for o in objects:
            if o["category_id"] == 1 and iou(coord, o["bbox"]) > 0.3:
                print('obstacle found')
                (row, col) = grid_centers[coord]
                maze[row][col] = 1
                objects.remove(o)
                
                # tempx = coord[0] / 720
                # tempy = coord[1] / 960
                # tempw = coord[2] / 720
                # temph = coord[3] / 960

                # o_x = o["bbox"][0] / 720
                # o_y = o["bbox"][1] / 960
                # o_w = o["bbox"][2] / 720
                # o_h = o["bbox"][3] / 960

                # print([tempx, tempy, tempw, temph], '|', [o_x, o_y, o_w, o_h])
    return maze
