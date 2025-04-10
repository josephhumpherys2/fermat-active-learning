"""
Note:
        Original and adapted implementation from @jifanz
        https://github.com/jifanz/GALAXY

"""


import numpy as np
from queue import PriorityQueue, Queue


class Node:
    def __init__(self, idx, label, loc):
        self.idx = idx
        self.label = label
        self.queried = False
        self.loc = loc
        self.neighbors = set()

    def add_neighbors(self, neighbors):
        for n in neighbors:
            self.neighbors.add(n)

    def set_neighbors(self, neighbors):
        self.neighbors = set(neighbors)


class Graph:
    def __init__(self, nodes, name):
        self.name = name
        self.nodes = set(nodes)
        self.node_list = list(nodes)
        self.num_nodes = len(nodes)
        self.node_dict = {node.idx: node for node in self.nodes}
        self.edges = []
        for node in nodes:
            for neighbor in node.neighbors:
                if node.idx < neighbor.idx:
                    self.edges.append((node, neighbor))
        self.edges = set(self.edges)
        self.queried = []
        self.not_queried = list(nodes)
        self.hamming_error = 0
        self.cut_error = 0
        self.preds = None
        self.labels = np.array([0 for _ in nodes])
        for node in nodes:
            self.labels[node.idx] = node.label

    def label(self, idx):
        node = self.node_dict[idx]
        node.queried = True
        self.queried.append(node)
        self.not_queried.remove(node)

        new_neighbors = []
        for neighbor in node.neighbors:
            if neighbor.queried and neighbor.label != node.label:
                self.cut_error += 1
                if node.idx < neighbor.idx:
                    self.edges.remove((node, neighbor))
                else:
                    self.edges.remove((neighbor, node))
                neighbor.neighbors.remove(node)
            else:
                new_neighbors.append(neighbor)
        node.set_neighbors(new_neighbors)

        if self.preds is not None and self.preds[idx] != node.label:
            self.hamming_error += 1

    def nn_pred(self):
        self.preds = np.random.choice([-1, 1], len(self.nodes))
        queue = Queue()
        predicted = set(self.queried)
        for n in self.queried:
            self.preds[n.idx] = n.label
            queue.put(n)
        while not queue.empty():
            n = queue.get()
            for neighbor in n.neighbors:
                if neighbor not in predicted:
                    queue.put(neighbor)
                    self.preds[neighbor.idx] = self.preds[n.idx]
                    predicted.add(neighbor)
        return self.preds

    def gt_error(self):
        return np.sum((np.array(self.preds) != self.labels).astype(int))

    def pred_cut(self):
        self.pred_cut_error = self.cut_error
        for (n1, n2) in self.edges:
            if self.preds[n1.idx] != self.preds[n2.idx]:
                self.pred_cut_error += 1
        self.pred_cut_error += (self.pred_cut_error - self.cut_error) / float(len(self.nodes))

    def shortest_shortest_path(self, weight_dictionary=None, weighted=False):
        queue = PriorityQueue()
        count = 0
        dist = {}
        path_prev = {}
        positive_queried = set()
        negative_queried = set()
        for node in self.queried:
            if node.label == 1:
                positive_queried.add(node)
        
            else:
                negative_queried.add(node)

        for node in self.nodes:
            if node in positive_queried:
                dist[node] = 0
                queue.put((0, count, node))
                count += 1
            else:
                dist[node] = np.inf
            path_prev[node] = None

        while not queue.empty():
            _, _, node = queue.get()

            if weighted:
                for neighbor in node.neighbors:
                    if node.idx < neighbor.idx:
                        new_dist = dist[node] + weight_dictionary[node.idx][neighbor.idx]
                    else:
                        new_dist = dist[node] + weight_dictionary[neighbor.idx][node.idx]
                    
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        path_prev[neighbor] = node
                        queue.put((new_dist, count, neighbor))
                        count += 1

                    if neighbor in negative_queried:
                        current = neighbor
                        path = [neighbor]
                        while path_prev[current] is not None:
                            current = path_prev[current]
                            path.append(current)
                        return new_dist, path
            
            # Not weighted
            else:
                for neighbor in node.neighbors:
                    new_dist = dist[node] + 1
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        path_prev[neighbor] = node
                        queue.put((new_dist, count, neighbor))
                        count += 1

                    if neighbor in negative_queried:
                        current = neighbor
                        path = [neighbor]
                        while path_prev[current] is not None:
                            current = path_prev[current]
                            path.append(current)
                        return new_dist, path
                
        return float('inf'), None

