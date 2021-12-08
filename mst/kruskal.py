import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class Kruskal:
    def __init__(self, nodes, edges) -> None:
        self.nodes = nodes
        self.edges = edges
            
    def mst(self):
        num_nodes = len(self.nodes) # 节点数量
        mst_edges = defaultdict(int)
        root = [_ for _ in range(num_nodes)]
        rank = [0] * num_nodes
        '''并查集
        '''
        def find_root(v):
            while root[v] != v:
                v = root[v]
            return v 
        def union(u, v):
            u_root = find_root(u)
            v_root = find_root(v)

            if rank[u_root] > rank[v_root]:
                root[v_root] = u_root
            elif rank[u_root] < rank[v_root]:
                root[u_root] = v_root
            else:
                root[v_root] = u_root
                rank[u] += 1
  
        edge_dict = defaultdict(int) # 边权映射
        for (u, v, w) in self.edges:
            edge_dict[(u, v)] = w
        # 按权重大小排序（顺序）
        sorted_edge_dict = sorted(edge_dict.items(), key=lambda x: x[1])
        for ((u, v), w) in sorted_edge_dict:
            if find_root(u) == find_root(v):
                continue
            else:
                union(u, v)
                mst_edges[(u, v)] = w
        return mst_edges        

def draw(G, color_edges):
    edges = list(G.edges)
    num_edge = len(edges)
    edge_color = ['b'] * num_edge
    color_edges = set(color_edges)

    for i in range(num_edge):
        u, v = edges[i][0], edges[i][1]
        # 无向图
        if (u, v) in color_edges or (v, u) in color_edges:
            edge_color[i] = 'r'
    pos = nx.kamada_kawai_layout(G)
    plt.title('MST')
    nx.draw(G, pos, with_labels=True, edge_color=edge_color)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # plt.savefig("kruskal.png", format="PNG")
    plt.show()

def main():
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1, 6), (0, 2, 1), (0, 3, 5), (1, 2, 5), (1, 4, 3), (2, 3, 5), (2, 4, 6), (2, 5, 4), (3, 5, 2), (4, 5, 6)]
    
    mst_edges = Kruskal(nodes, edges).mst()
    print('{} | {}'.format(mst_edges, sum(mst_edges.values())))
        
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    draw(G, list(mst_edges.keys()))

if __name__ == '__main__':
    main()