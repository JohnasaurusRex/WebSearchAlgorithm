from flask import Flask, render_template, request, jsonify
import random
import math
from collections import deque
import heapq
from queue import PriorityQueue

app = Flask(__name__)

graph_data = {
    'nodes': [
        {'index': 0, 'x': 448.3075371591334, 'y': 108.85981125642901, 'label': 'A', 'isStartNode': False, 'isGoalNode': False, 'links': 3},
        {'index': 1, 'x': 300, 'y': 200, 'label': 'B', 'isStartNode': False, 'isGoalNode': False, 'links': 2},
        {'index': 2, 'x': 600, 'y': 200, 'label': 'C', 'isStartNode': False, 'isGoalNode': False, 'links': 2},
        {'index': 3, 'x': 50, 'y': 300, 'label': 'D', 'isStartNode': False, 'isGoalNode': False, 'links': 2},
        {'index': 4, 'x': 750, 'y': 300, 'label': 'E', 'isStartNode': False, 'isGoalNode': False, 'links': 2},
        {'index': 5, 'x': 300, 'y': 400, 'label': 'F', 'isStartNode': False, 'isGoalNode': False, 'links': 2},
        {'index': 6, 'x': 600, 'y': 400, 'label': 'G', 'isStartNode': False, 'isGoalNode': False, 'links': 3},
        {'index': 7, 'x': 448.3075371591334, 'y': 500, 'label': 'H', 'isStartNode': False, 'isGoalNode': False, 'links': 3}
    ],
    'links': [
         {'source': 0, 'target': 1, 'distance': 20},
         {'source': 1, 'target': 0, 'distance': 20},  # Vice-versa link
         {'source': 0, 'target': 2, 'distance': 23.58884702259604},
         {'source': 2, 'target': 0, 'distance': 23.58884702259604},  # Vice-versa link
         {'source': 0, 'target': 3, 'distance': 32},
         {'source': 3, 'target': 0, 'distance': 32},  # Vice-versa link
         {'source': 1, 'target': 2, 'distance': 15},
         {'source': 2, 'target': 1, 'distance': 15},  # Vice-versa link
         {'source': 2, 'target': 5, 'distance': 18},
         {'source': 5, 'target': 2, 'distance': 18},  # Vice-versa link
         {'source': 2, 'target': 4, 'distance': 13.892122440081096},
         {'source': 4, 'target': 2, 'distance': 13.892122440081096},  # Vice-versa link
         {'source': 3, 'target': 7, 'distance': 42.461598428328124},
         {'source': 7, 'target': 3, 'distance': 42.461598428328124},  # Vice-versa link
         {'source': 5, 'target': 6, 'distance': 13.892122440081096},
         {'source': 6, 'target': 5, 'distance': 13.892122440081096},  # Vice-versa link
         {'source': 6, 'target': 7, 'distance': 9},
         {'source': 7, 'target': 6, 'distance': 9}  # Vice-versa link
    ],
    'startNodeLabel': '',
    'goalNodeLabel': ''
}




@app.route('/graph_data', methods=['GET'])
def get_graph_data():
    print(graph_data)
    return jsonify(graph_data)

@app.route('/')
def index():
    global graph_data  # Use the global graph_data variable
    return render_template('index.html')


@app.route('/generate_nodes', methods=['POST'])
def generate_nodes():
    global graph_data
    node_count = int(request.form['node_count'])
    nodes = [
        {
            'index': i,
            'x': random.random() * 850,
            'y': random.random() * 650,
            'label': chr(65 + i),
            'isStartNode': False,
            'isGoalNode': False,
            'links': 0
        }
        for i in range(node_count)
    ]
    links = []
    for i in range(node_count):
        for j in range(i+1, node_count):  # Connect each node to every other node
            source = i
            target = j

            nodes[source]['links'] += 1
            nodes[target]['links'] += 1

            x1, y1 = nodes[source]['x'], nodes[source]['y']
            x2, y2 = nodes[target]['x'], nodes[target]['y']
            distance = max(10, math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 10)

            links.append({'source': source, 'target': target, 'distance': distance})
            links.append({'source': target, 'target': source, 'distance': distance})

    graph_data = {'nodes': nodes, 'links': links, 'startNodeLabel': '', 'goalNodeLabel': ''}
    print(graph_data)
    return jsonify(graph_data)




def dfs(graph_data, start_label, goal_label):
    visited_labels = []
    paths = []
    stack = [(start_label, None, 0)]  # Add initial distance as 0

    iteration_counts = {
        'enqueues': 0,
        'extensions': 0,
        'queue_size': 0,
        'path_nodes': 0,
        'path_cost': 0
    }

    while stack:
        current_label, parent_label, distance = stack.pop()

        if current_label not in visited_labels:
            visited_labels.append(current_label)

            if parent_label is not None:
                paths.append((parent_label, current_label))
                iteration_counts['extensions'] += 1

            if current_label == goal_label:
                break

            current_node = next(node for node in graph_data['nodes'] if node['label'] == current_label)

            unvisited_neighbors = [(graph_data['nodes'][link['target']]['label'], link['distance'])
                                   for link in graph_data['links']
                                   if link['source'] == current_node['index'] and graph_data['nodes'][link['target']][
                                       'label'] not in visited_labels]

            # Sort neighbors alphabetically and add to the stack
            unvisited_neighbors.sort(reverse=True)  # Reverse the order for lexical ordering
            stack.extend((neighbor, current_label, distance + link_distance) for neighbor, link_distance in
                         unvisited_neighbors)

            # Update counts
            iteration_counts['enqueues'] += len(unvisited_neighbors)
            iteration_counts['queue_size'] = len(stack)
            iteration_counts['path_nodes'] = len(paths)
            iteration_counts['path_cost'] = distance  # Update with the last distance

            # Display counts for the current iteration
            print(f"Iteration Counts: {iteration_counts}")

    print(f"DFS Visited Labels: {visited_labels}")
    print(f"DFS Paths: {paths}")
    return visited_labels, paths, iteration_counts



def bfs(graph_data, start_label, goal_label):
    visited_labels = []
    paths = []
    queue = deque([(start_label, None, 0)])  # Initialize distance as 0

    iteration_counts = {
        'enqueues': 0,
        'extensions': 0,
        'queue_size': 0,
        'path_nodes': 0,
        'path_cost': 0
    }

    while queue:
        current_label, parent_label, distance = queue.popleft()

        if current_label not in visited_labels:
            visited_labels.append(current_label)

            if parent_label is not None:
                paths.append((parent_label, current_label))
                iteration_counts['extensions'] += 1

            if current_label == goal_label:
                break

            current_node = next(node for node in graph_data['nodes'] if node['label'] == current_label)

            unvisited_neighbors = [(graph_data['nodes'][link['target']]['label'], link['distance'])
                                   for link in graph_data['links']
                                   if link['source'] == current_node['index'] and graph_data['nodes'][link['target']][
                                       'label'] not in visited_labels]

            # Sort neighbors alphabetically and add to the queue
            unvisited_neighbors.sort()  # Reverse the order for lexical ordering
            queue.extend((neighbor, current_label, distance + link_distance) for neighbor, link_distance in
                         unvisited_neighbors)

            # Update counts
            iteration_counts['enqueues'] += len(unvisited_neighbors)
            iteration_counts['queue_size'] = len(queue)
            iteration_counts['path_nodes'] = len(paths)
            iteration_counts['path_cost'] = distance  # Update with the last distance

            # Display counts for the current iteration
            print(f"Iteration Counts: {iteration_counts}")

    print(f"BFS Visited Labels: {visited_labels}")
    print(f"BFS Paths: {paths}")
    return visited_labels, paths, iteration_counts



def hill_climb(graph_data, start_label, goal_label):
    visited_labels = []
    paths = []
    current_label = start_label

    iteration_counts = {
        'enqueues': 0,
        'extensions': 0,
        'path_nodes': 0,
        'path_cost': 0
    }

    while current_label != goal_label:
        visited_labels.append(current_label)

        current_node = next(node for node in graph_data['nodes'] if node['label'] == current_label)

        unvisited_neighbors = [(graph_data['nodes'][link['target']]['label'], link['distance'])
                               for link in graph_data['links']
                               if link['source'] == current_node['index'] and graph_data['nodes'][link['target']][
                                   'label'] not in visited_labels]

        if not unvisited_neighbors:
            # Stuck in a local minimum, break out of the loop
            break

        # Choose the neighbor with the smallest distance as the next step
        next_label, distance = min(unvisited_neighbors, key=lambda x: x[1])

        paths.append((current_label, next_label))
        iteration_counts['extensions'] += 1
        iteration_counts['path_nodes'] = len(paths)
        iteration_counts['path_cost'] += distance

        # Display counts for the current iteration
        print(f"Iteration Counts: {iteration_counts}")

        current_label = next_label

    print(f"Hill Climbing Visited Labels: {visited_labels}")
    print(f"Hill Climbing Paths: {paths}")
    return visited_labels, paths, iteration_counts

# def beam_search(graph_data, start_label, goal_label, beam_width=2):
#     visited_labels = []
#     paths = [[start_label]]
#     iteration_counts = {
#         'enqueues': 0,
#         'extensions': 0,
#         'path_nodes': 0,
#         'path_cost': 0
#     }
#
#     while paths:
#         # Generate all successor paths
#         successor_paths = [path + [neighbor[0]] for path in paths for neighbor in get_neighbors(path[-1], graph_data, visited_labels)]
#         iteration_counts['enqueues'] += len(successor_paths)
#
#         # Select the beam_width best paths according to their cost
#         paths = sorted(successor_paths, key=lambda path: path_cost(path, graph_data))[:beam_width]
#         visited_labels.extend(path[-1] for path in paths)
#
#         # Check if goal is reached
#         for path in paths:
#             if path[-1] == goal_label:
#                 print(f"Beam Search Visited Labels: {visited_labels}")
#                 print(f"Beam Search Paths: {paths}")
#                 return visited_labels, paths, iteration_counts
#
#         iteration_counts['extensions'] += len(paths)
#         iteration_counts['path_nodes'] += sum(len(path) for path in paths)
#         iteration_counts['path_cost'] += sum(path_cost(path, graph_data) for path in paths)
#
#     # If no paths found, return the start node as the only path
#     if not paths:
#         paths = [[start_label]]
#
#     print(f"Beam Search Visited Labels: {visited_labels}")
#     print(f"Beam Search Paths: {paths}")
#     return visited_labels, paths, iteration_counts
#
#
#
# def get_neighbors(label, graph_data, visited_labels):
#     node = next(node for node in graph_data['nodes'] if node['label'] == label)
#     return [(graph_data['nodes'][link['target']]['label'], link['distance'])
#             for link in graph_data['links']
#             if link['source'] == node['index'] and graph_data['nodes'][link['target']]['label'] not in visited_labels]
#
# def path_cost(path, graph_data):
#     total_cost = 0
#     for i in range(len(path) - 1):
#         source_label = path[i]
#         target_label = path[i + 1]
#         link = next(link for link in graph_data['links'] if (graph_data['nodes'][link['source']]['label'] == source_label and graph_data['nodes'][link['target']]['label'] == target_label))
#         total_cost += link['distance']
#     # Add heuristic cost for the last node in the path
#     last_node = next(node for node in graph_data['nodes'] if node['label'] == path[-1])
#     goal_node = next(node for node in graph_data['nodes'] if node['label'] == graph_data['goalNodeLabel'])
#     heuristic_cost = calculate_manhattan_distance(last_node, goal_node)
#     total_cost += heuristic_cost
#     return total_cost
#
# def calculate_manhattan_distance(node1, node2):
#     return abs(node1['x'] - node2['x']) + abs(node1['y'] - node2['y'])


def beam_search(graph_data, start_label, goal_label):
    visited_labels = []
    paths = []
    beam_width = 2  # You can adjust this value as needed
    queue = deque([(start_label, None, 0)])  # Initialize distance as 0

    iteration_counts = {
        'enqueues': 0,
        'extensions': 0,
        'queue_size': 0,
        'path_nodes': 0,
        'path_cost': 0
    }

    while queue:
        beam = []
        for _ in range(min(beam_width, len(queue))):
            current_label, parent_label, total_cost = queue.popleft()
            if current_label not in visited_labels:
                visited_labels.append(current_label)
                if parent_label is not None:
                    paths.append((parent_label, current_label))
                    iteration_counts['extensions'] += 1

                if current_label == goal_label:
                    iteration_counts['path_cost'] = total_cost
                    print(f"Beam Search Visited Labels: {visited_labels}")
                    return paths, visited_labels, iteration_counts

                current_node = next(node for node in graph_data['nodes'] if node['label'] == current_label)
                neighbors = [(graph_data['nodes'][link['target']]['label'], link['distance'])
                             for link in graph_data['links']
                             if link['source'] == current_node['index'] and graph_data['nodes'][link['target']][
                                 'label'] not in visited_labels]

                # Add neighbors to the beam
                beam.extend((neighbor, current_label, total_cost + link_distance) for neighbor, link_distance in neighbors)
                iteration_counts['enqueues'] += len(neighbors)

        # Select the top-k items from the beam
        beam.sort(key=lambda x: x[2])  # Sort by total cost
        queue.extend(beam[:beam_width])

        # Update counts
        iteration_counts['queue_size'] = len(queue)
        iteration_counts['path_nodes'] = len(paths)

        # Display counts for the current iteration
        print(f"Iteration Counts: {iteration_counts}")

    print(f"Beam search stopped. Node {goal_label} haven't found Total Cost: {total_cost}")
    print(paths)
    print(visited_labels)
    return paths, visited_labels, iteration_counts

def b_bound(graph_data, start_label, goal_label):
    visited_labels = []
    paths = []
    current_label = start_label

    iteration_counts = {
        'enqueues': 0,
        'extensions': 0,
        'path_nodes': 0,
        'path_cost': 0
    }

    priority_queue = []  # Priority queue for partial solutions (min-heap)

    heapq.heappush(priority_queue, (0, [start_label]))  # Initial state with priority 0

    while priority_queue:
        current_priority, current_path = heapq.heappop(priority_queue)
        current_label = current_path[-1]

        if current_label == goal_label:
            # Found a complete path to the goal
            paths.append(tuple(current_path))
            continue

        if current_label in visited_labels:
            # Skip already visited nodes to avoid cycles
            continue

        visited_labels.append(current_label)

        current_node = next(node for node in graph_data['nodes'] if node['label'] == current_label)

        unvisited_neighbors = [(graph_data['nodes'][link['target']]['label'], link['distance'])
                               for link in graph_data['links']
                               if link['source'] == current_node['index'] and
                               graph_data['nodes'][link['target']]['label'] not in visited_labels]

        if not unvisited_neighbors:
            # Stuck in a local minimum, backtrack
            continue

        # Enqueue the unvisited neighbors with their estimated priorities
        for neighbor_label, distance in unvisited_neighbors:
            new_priority = current_priority + distance
            new_path = current_path + [neighbor_label]
            heapq.heappush(priority_queue, (new_priority, new_path))

        iteration_counts['enqueues'] += len(unvisited_neighbors)
        iteration_counts['extensions'] += 1
        iteration_counts['path_nodes'] = len(paths)
        iteration_counts['path_cost'] += current_priority

        # Display counts for the current iteration
        print(f"Iteration Counts: {iteration_counts}")

    print(f"Branch and Bound Visited Labels: {visited_labels}")
    print(f"Branch and Bound Paths: {paths}")
    return visited_labels, paths, iteration_counts

def heuristic(node, goal_node):
    # Implement your heuristic function here
    # This function should estimate the cost from the current node to the goal node
    # The heuristic should be admissible, meaning it should never overestimate the true cost
    return 0  # Replace with your actual heuristic calculation

def a_star(graph_data, start_label, goal_label):
    visited_labels = []
    paths = []
    current_label = start_label

    iteration_counts = {
        'enqueues': 0,
        'extensions': 0,
        'path_nodes': 0,
        'path_cost': 0
    }

    priority_queue = []  # Priority queue for nodes (min-heap)

    heapq.heappush(priority_queue, (0, [start_label], 0))  # Initial state with priority 0 and g(n) 0

    while priority_queue:
        current_priority, current_path, current_g = heapq.heappop(priority_queue)
        current_label = current_path[-1]

        if current_label == goal_label:
            # Found a complete path to the goal
            paths.append(tuple(current_path))
            continue

        if current_label in visited_labels:
            # Skip already visited nodes to avoid cycles
            continue

        visited_labels.append(current_label)

        current_node = next(node for node in graph_data['nodes'] if node['label'] == current_label)

        unvisited_neighbors = [(graph_data['nodes'][link['target']]['label'], link['distance'])
                               for link in graph_data['links']
                               if link['source'] == current_node['index'] and
                               graph_data['nodes'][link['target']]['label'] not in visited_labels]

        if not unvisited_neighbors:
            # Stuck in a local minimum, backtrack
            continue

        # Enqueue the unvisited neighbors with their estimated priorities
        for neighbor_label, distance in unvisited_neighbors:
            new_g = current_g + distance
            new_priority = new_g + heuristic(graph_data['nodes'][neighbor_label], graph_data['nodes'][goal_label])
            new_path = current_path + [neighbor_label]
            heapq.heappush(priority_queue, (new_priority, new_path, new_g))

        iteration_counts['enqueues'] += len(unvisited_neighbors)
        iteration_counts['extensions'] += 1
        iteration_counts['path_nodes'] = len(paths)
        iteration_counts['path_cost'] += current_g

        # Display counts for the current iteration
        print(f"Iteration Counts: {iteration_counts}")

    print(f"A* Visited Labels: {visited_labels}")
    print(f"A* Paths: {paths}")
    return visited_labels, paths, iteration_counts







@app.route('/dfs', methods=['POST'])
def run_dfs():
    try:
        start_label = request.form['start_label']
        goal_label = request.form['goal_label']
        visited_labels, paths, iteration_counts = dfs(graph_data, start_label, goal_label)
        return jsonify({'visited_labels': visited_labels, 'paths': paths, 'iteration_counts': iteration_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/bfs', methods=['POST'])
def run_bfs():
    try:
        start_label = request.form['start_label']
        goal_label = request.form['goal_label']
        visited_labels, paths, iteration_counts = bfs(graph_data, start_label, goal_label)
        return jsonify({'visited_labels': visited_labels, 'paths': paths, 'iteration_counts': iteration_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/hill_climb', methods=['POST'])
def run_hillclimb():
    try:
        start_label = request.form['start_label']
        goal_label = request.form['goal_label']
        visited_labels, paths, iteration_counts = hill_climb(graph_data, start_label, goal_label)
        return jsonify({'visited_labels': visited_labels, 'paths': paths, 'iteration_counts': iteration_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/beam_search', methods=['POST'])
def run_beam():
    try:
        start_label = request.form['start_label']
        goal_label = request.form['goal_label']
        visited_labels, paths, iteration_counts = beam_search(graph_data, start_label, goal_label)
        return jsonify({'visited_labels': visited_labels, 'paths': paths, 'iteration_counts': iteration_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/b_bound', methods=['POST'])
def run_bound():
    try:
        start_label = request.form['start_label']
        goal_label = request.form['goal_label']
        visited_labels, paths, iteration_counts = b_bound(graph_data, start_label, goal_label)
        return jsonify({'visited_labels': visited_labels, 'paths': paths, 'iteration_counts': iteration_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/a_star', methods=['POST'])
def run_astar():
    try:
        start_label = request.form['start_label']
        goal_label = request.form['goal_label']
        visited_labels, paths, iteration_counts = a_star(graph_data, start_label, goal_label)
        return jsonify({'visited_labels': visited_labels, 'paths': paths, 'iteration_counts': iteration_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
