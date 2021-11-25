import torch
import dgl
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt


INF = 20211120 #一个较大的数，graph_mat[i][j]=INF表示i不可以到达j


class FloydWarshall:
    def __init__(self, graph_mat) -> None:
        self.graph_mat = graph_mat # 图的邻接矩阵

    def shortest_path(self):
        m, n = len(self.graph_mat), len(self.graph_mat[0])
        distance_mat = self.graph_mat.copy()
        path = [[None] * n for _ in range(m)] # 记录路径
        for k in range(m):
            for i in range(m):
                for j in range(n):
                    if distance_mat[i][k]+distance_mat[k][j] < distance_mat[i][j]:
                        distance_mat[i][j] = distance_mat[i][k]+distance_mat[k][j]
                        path[i][j] = k
        return distance_mat, path
# 递归
# def get_nodes_edges_recursion(paths, i, j):
#         if i == j:
#             return [i], []
#         else:
#             if paths[i][j] == None:
#                 return [j], [(i, j)]
#             else:
#                 left_nodes, left_edges = get_nodes_edges(paths, i, paths[i][j])
#                 right_nodes, right_edges = get_nodes_edges(paths, paths[i][j], j)
#         return left_nodes+right_nodes, left_edges+right_edges

def get_nodes_edges(paths, i, j):
    pass_nodes = [i, j]
    pass_edges = []
    while paths[i][j] != None:
        j = paths[i][j]
        pass_nodes.insert(1, j)
        pass_edges.append((j, pass_nodes[2]))
    pass_edges.append((i, j))
    return pass_nodes, pass_edges[::-1]

def draw(sp_mat, color_nodes, color_edges):
    '''画一张图，并给指定的节点和边着色
    Parameters
    ----------
    sp_mat : scipy.sparse.spmatrix
    color_nodes : list, e.g. [0, 1, 2, ...]
        指定着色的节点
    color_edges : list, e.g. [(0, 1), (2, 5), ...]
        指定着色的边
        
    Returns
    -------
    None
    '''
    
    G = dgl.from_scipy(sp_mat, eweight_name='w') # 注意：dgl.from_scipy()默认为有向多重图
    # print(type(G))
    nx_multi_G = dgl.to_networkx(G, edge_attrs='w').to_undirected() # 变为无向简单图
    nx_G = nx.Graph(nx_multi_G)
    # print(nx_G.edges(data=True))
    edge_labels = nx.get_edge_attributes(nx_G, 'w')
    # tensor.item() tensor(0.)->0(tensor->int)
    edge_labels = { (key[0],key[1]): edge_labels[key].item() for key in edge_labels }
    edges = list(nx_G.edges)
    num_nodes = nx_G.number_of_nodes()
    num_edges = nx_G.number_of_edges()
    node_color = ['b'] * num_nodes
    edge_color = ['b'] * num_edges

    for i in color_nodes:
        node_color[i] = 'r'

    for i in range(num_edges):
        u, v = edges[i][0], edges[i][1]
        if (u, v) in set(color_edges) or (v, u) in set(color_edges):
            edge_color[i] = 'r'
    pos = nx.circular_layout(nx_G)
    plt.title('Undigraph: Shortest Path')
    nx.draw(nx_G, pos, with_labels=True, node_color=node_color, edge_color=edge_color)
    nx.draw_networkx_edge_labels(nx_G, pos, edge_labels=edge_labels)
    plt.savefig('floyd_warshall.png', format='PNG')
    plt.show()   

def main():
    row = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4])
    col = np.array([1, 2, 2, 3, 4, 3, 4, 5, 5])
    data = np.array([1, 12, 9, 3, 5, 4, 13, 15, 4])

    count_node = 6
    sp_mat = sp.coo_matrix((data, (row, col)), shape=(count_node, count_node))
    graph_mat = sp_mat.toarray()
    # 无向图
    for i in range(len(graph_mat)):
        for j in range(i):
            if graph_mat[i][j] == 0 and graph_mat[j][i] == 0:
                graph_mat[i][j] = graph_mat[j][i] = INF
            else:
                graph_mat[i][j] += graph_mat[j][i]
                graph_mat[j][i] = graph_mat[i][j]
    
    distance_mat, paths = FloydWarshall(graph_mat).shortest_path()
    print(distance_mat)
    print(paths)

    for i in range(count_node):
        for j in range(i+1, count_node):
            print('{}->{}: {} | {}'.format(i, j, get_nodes_edges(paths, i, j), distance_mat[i][j]))
            
    start_node, end_node = 0, 5 # 开始节点，结束节点
    pass_nodes, pass_edges = get_nodes_edges(paths, start_node, end_node)
    '''使用get_nodes_edges_recursion注意添加开始节点
    '''
    # pass_nodes, pass_edges = get_nodes_edges_recursion(paths, start_node, end_node)
    # pass_nodes.insert(0, start_node)
    draw(sp_mat, pass_nodes, pass_edges)
if __name__ == '__main__':
    main()
