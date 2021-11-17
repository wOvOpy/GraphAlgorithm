import torch
import dgl
import numpy as np
import sys
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict
import matplotlib.pyplot as plt

INF = 20211117

# 迪杰特斯拉算法
class dijkstra:
    def __init__(self, graph):
        self.graph = graph

    def shortest_path(self, start_node, end_node):
        n = len(self.graph)
        paths = [-1] * n
        for i in range(n):
            if self.graph[start_node][i] == INF or self.graph[start_node][i] == 0:
                paths[i] = -1
            else:
                paths[i] = start_node
        visited = [start_node]
        rested = [_ for _ in range(n) if _ != start_node]
        distance = self.graph[start_node]
        
        while len(rested):
            min_idx = rested[0]
            for i in rested:
                if distance[i] < distance[min_idx]:
                    min_idx = i
            if min_idx == end_node:
                break
            visited.append(min_idx)
            rested.remove(min_idx)
            
            for i in rested:
                if distance[min_idx] + self.graph[min_idx][i] < distance[i]:
                    distance[i] = distance[min_idx] + self.graph[min_idx][i]
                    paths[i] = min_idx
        return paths, distance[end_node]
    

# 最短路径上的节点转化为经过的边集合
def paths_to_edges(paths, end_node):
    edges = set()
    v = end_node
    while paths[end_node] != -1:
        u = paths[end_node]
        edges.add((u, v))
        end_node = u
        v = u
    return edges

# 画图
def draw_graph(sp_mat, count_node, edge_labels, edges):
    n = len(sp_mat.data)
    G = dgl.from_scipy(sp_mat, eweight_name='w')
    nx_G = G.to_networkx().to_undirected()
    edge_color = ['b'] * n
    node_color = [[.7, .7, .7]]
    node_dict = defaultdict(list)
    shortest_node = set()
    for u, v in edges:
        shortest_node.add(u)
        shortest_node.add(v)
    node_dict['r'] = list(shortest_node)
    node_dict['b'] = list(set(range(count_node)) - shortest_node)
    edges_all = list(nx_G.edges)
    for i in range(n):
        u = edges_all[i][0]
        v = edges_all[i][1]
        if (u, v) in edges or (v, u) in edges:
            edge_color[i] = 'r'
    
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_G)
    # for node_color, node_list in node_map.items():
    # print(node_dict)
    for node_color, node_list in node_dict.items():
        nx.draw(nx_G, pos, with_labels=True, node_color=node_color, nodelist = node_list, edge_color=tuple(edge_color))
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='b')
    # plt.show()
    plt.savefig("shortest_path.png", format="PNG") 

def main():
    row = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4])
    col = np.array([1, 2, 2, 3, 4, 3, 4, 5, 5])
    data = np.array([1, 12, 9, 3, 5, 4, 13, 15, 4])

    count_node = 6
    start_node = 0
    end_node = 4
    sp_mat = sp.coo_matrix((data, (row, col)), shape=(count_node, count_node))
    graph = sp_mat.toarray()
    # 无向图
    for i in range(len(graph)):
        for j in range(i):
            if graph[i][j] == 0 and graph[j][i] == 0:
                graph[i][j] = graph[j][i] = INF
            else:
                graph[i][j] += graph[j][i]
                graph[j][i] = graph[i][j]
    paths, distance = dijkstra(graph).shortest_path(start_node, end_node)
    edge_labels = defaultdict(tuple)
    for i in range(len(data)):
        edge_labels[(row[i], col[i])] = data[i]

    edges = paths_to_edges(paths, end_node)
    draw_graph(sp_mat, count_node, edge_labels, edges)
    print('{}到{}的最短路径是{}'.format(start_node, end_node, distance))
    print('经过了{}这些边到达'.format(edges))

if __name__ == '__main__':
    main()