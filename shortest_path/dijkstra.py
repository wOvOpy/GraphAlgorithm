from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

INF = 20211120


class Dijkstra:
    def __init__(self, graph, num_nodes) -> None:
        self.graph = graph
        self.num_nodes = num_nodes

    def shortest_path(self, start_node):
        paths = [-1] * self.num_nodes
        distance = [INF] * self.num_nodes
        edge_weight = dict()
        node_neighbors = defaultdict(set)
        for (u, v, w) in self.graph:
            # 有向图
            edge_weight[(u, v)] = w
            node_neighbors[u].add(v)

        for neighbor in node_neighbors[start_node]:
            paths[neighbor] = start_node
            distance[neighbor] = edge_weight[(start_node, neighbor)]

            visited = [start_node]
            not_visit = [_ for _ in range(self.num_nodes) if _ != start_node]

            while len(not_visit):
                min_w_node = not_visit[0]
                for i in not_visit:
                    if distance[i] < distance[min_w_node]:
                        min_w_node = i
                not_visit.remove(min_w_node)
                visited.append(min_w_node)

                for i in not_visit:
                    if distance[min_w_node]+edge_weight.get((min_w_node, i), INF) < distance[i]:
                        distance[i] = distance[min_w_node]+edge_weight[(min_w_node, i)]
                        paths[i] = min_w_node
        return paths, distance

def paths_to_edges(paths, end_node):
    edges, nodes = [], []
    v = end_node
    nodes.append(v)
    while paths[end_node] != -1:
        u = paths[end_node]
        nodes.append(u)
        edges.append((u, v))
        end_node = u
        v = u
    return nodes[::-1], edges[::-1]

def draw(graph, color_nodes, color_edges):
    DG = nx.DiGraph()
    DG.add_weighted_edges_from(graph)
    edges = list(DG.edges)
    num_nodes = DG.number_of_nodes()
    num_edges = DG.number_of_edges()
    node_color = ['b'] * num_nodes
    edge_color = ['b'] * num_edges

    for i in color_nodes:
        node_color[i] = 'r'

    for i in range(num_edges):
        u, v = edges[i][0], edges[i][1]
        if (u, v) in set(color_edges):
            edge_color[i] = 'r'
    pos = nx.circular_layout(DG)
    nx.draw(DG, pos, with_labels=True, node_color=node_color, edge_color=tuple(edge_color))
    edge_labels = nx.get_edge_attributes(DG, 'weight')
    nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels)
    # plt.savefig('DG_SP.png', format='PNG')
    plt.show()


def main():
    num_nodes = 6
    graph = [(0, 1, 1), (0, 2, 12), (1, 2, 9), (1, 3, 3), (2, 4, 5), (2, 3, 4), (3, 4, 13), (3, 5, 15), (4, 5, 4)]

    start_node = 0
    paths, distance = Dijkstra(graph, num_nodes).shortest_path(start_node)
    for end_node in range(num_nodes):
        print('{}->{}: {} | {}'.format(start_node, end_node, paths_to_edges(paths, end_node), distance[end_node]))
    color_nodes, color_edges = paths_to_edges(paths, end_node=5)
    draw(graph, color_nodes, color_edges)

if __name__ == '__main__':
    main()
