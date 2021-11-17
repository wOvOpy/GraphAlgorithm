import numpy as np
import networkx as nx
import dgl
import matplotlib.pyplot as plt
from collections import defaultdict

'''
kruskal算法：并查集实现，结果已经输出，后面会展现动图，给并查集添加rank降低时间复杂度
，我要去看论文了，后面再写。
'''
class kruskal:
    def __init__(self, graph) -> None:
        self.graph = graph
            
    def mst(self):
        n = len(self.graph)
        tree = []
        parent = [_ for _ in range(n)]
        
        def find_parent(v):
            while parent[v] != v:
                v = parent[v]
            return v 
        def union(u, v):
            parent[find_parent(v)] = find_parent(u)     
        
        edge_dict = defaultdict(int)
        for (u, v, w) in self.graph:
            edge_dict[(u, v)] = w
        
        sorted_edge_dict = sorted(edge_dict.items(), key=lambda x: x[1])
        for ((u, v), w) in sorted_edge_dict:
            if find_parent(u) == find_parent(v):
                continue
            else:
                union(u, v)
                tree.append((u, v, w))
        return tree        


def main():
    graph = [(0, 1, 6), (0, 2, 1), (0, 3, 5), (1, 2, 5), (1, 4, 3), (2, 3, 5), (2, 4, 6), (2, 5, 4), (3, 5, 2), (4, 5, 6)]
    
    tree = kruskal(graph).mst()
    print(tree)
    G = nx.Graph()
    G.add_weighted_edges_from(graph)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos)
    plt.savefig("kruskal.png")     

if __name__ == '__main__':
    main()