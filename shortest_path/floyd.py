import torch
import dgl
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt


INF = 20211120


class Floyd:
    def __init__(self, graph_mat) -> None:
        self.graph_mat = graph_mat

    def shortest_path(self):
        m, n = len(self.graph_mat), len(self.graph_mat[0])
        path = [[-1] * n for _ in range(m)]
        for k in range(m):
            for i in range(m):
                for j in range(n):
                    if self.graph_mat[i][k]+self.graph_mat[k][j] < self.graph_mat[i][j]:
                        self.graph_mat[i][j] = self.graph_mat[i][k]+self.graph_mat[k][j]
                        path[i][j] = k
        return self.graph_mat, path
    

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
    
    mst_mat, path = Floyd(graph_mat).shortest_path()
    print(mst_mat)
    # print(path)
    def record_path(i, j):
        if i == j:
            return []
        else:
            if path[i][j] == -1:
                return [(i, j)]
            else:
                left = record_path(i, path[i][j])
                right = record_path(path[i][j], j)
        return left + right

    for i in range(count_node):
        for j in range(count_node):
            if i != j:
                print('{}到{}的最短路径经过了边：{}，长度为：{}'.format(i, j, record_path(i, j), mst_mat[i][j]))

if __name__ == '__main__':
    main()
