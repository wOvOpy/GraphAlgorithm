import numpy as np
import networkx as nx
import dgl
import matplotlib.pyplot as plt
from collections import defaultdict

'''
我要去看论文了，byebye!!
'''
class kruskal:
    def __init__(self, graph) -> None:
        self.graph = graph
            
    def mst(self):
        n = len(self.graph)
        tree = defaultdict(int)
        parent = [_ for _ in range(n)]
        rank = [0] * n
        
        def find_parent(v):
            while parent[v] != v:
                v = parent[v]
            return v 
        def union(u, v):
            u_root = find_parent(u)
            v_root = find_parent(v)

            if rank[u_root] > rank[v_root]:
                parent[v_root] = u_root
                rank[v] += 1
            else:
                parent[u_root] = v_root
                rank[u] += 1

        
        edge_dict = defaultdict(int)
        for (u, v, w) in self.graph:
            edge_dict[(u, v)] = w
        
        sorted_edge_dict = sorted(edge_dict.items(), key=lambda x: x[1])
        for ((u, v), w) in sorted_edge_dict:
            if find_parent(u) == find_parent(v):
                continue
            else:
                union(u, v)
                tree[(u, v)] = w
        return tree        


def main():
    graph = [(0, 1, 6), (0, 2, 1), (0, 3, 5), (1, 2, 5), (1, 4, 3), (2, 3, 5), (2, 4, 6), (2, 5, 4), (3, 5, 2), (4, 5, 6)]
    
    tree = kruskal(graph).mst()
    print(tree)
    n = len(graph)
    edge_color = ['b'] * n
    edge_set = set([_ for _ in tree.keys()])
    print(edge_set)
        
    G = nx.Graph()
    G.add_weighted_edges_from(graph)
    edge_all = list(G.edges)
    for i in range(n):
        if edge_all[i] in edge_set:
            edge_color[i] = 'r'
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, edge_color=tuple(edge_color))
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig("mst/kruskal.png", format="PNG")
    plt.show()

if __name__ == '__main__':
    main()