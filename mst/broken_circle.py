import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict



class BrokenCircle:
    def __init__(self, graph) -> None:
        self.graph = graph

    def broken_circle(self):
        node_neighbors = defaultdict(set) # 节点和其邻居映射
        weight_edges = defaultdict(set) # 权重和边映射
        weights = []
        
        discard_edges = defaultdict(int) # 丢弃的边
        for (u, v, w) in self.graph:
            node_neighbors[u].add(v)
            node_neighbors[v].add(u)
            weights.append(w)
            weight_edges[w].add((u, v))
        weights.sort(reverse=True) # 权重逆序
        edges_visited = set() # 已经访问的边
        def bfs(start, end):
            '''广度优先搜索，判断start能否到达end（start和end是否连通）
            Parameters
            ----------
                start : int
                    开始节点
                end : int
                    结束节点
                    
            Returns
            -------
                bool
            '''
            queue = []
            nodes_visited = set()
            queue.append(start)
            while queue:
                cur = queue.pop(-1)
                nodes_visited.add(cur)
                if cur == end:
                    return True
                for cur_neighbor in node_neighbors[cur]:
                    if cur_neighbor not in nodes_visited:
                        queue.append(cur_neighbor)
            return False
                
        for w in weights:
            for edge in weight_edges[w]:
                if edge not in edges_visited:
                    u, v = edge[0], edge[1]
                    node_neighbors[u].remove(v)
                    node_neighbors[v].remove(u)
                    if bfs(u, v):
                        discard_edges[(u, v)] = w
                    else:
                        node_neighbors[u].add(v)
                        node_neighbors[v].add(u)    
                    edges_visited.add((u, v))
        return discard_edges
                        

def draw(G, color_edges):
    edges = list(G.edges)
    color_edges = set(color_edges)
    n = len(edges)
    edge_color = ['b'] * n

    for i in range(n):
        u, v = edges[i][0], edges[i][1]
        if (u, v) in color_edges or (v, u) in color_edges:
            edge_color[i] = 'r'
    
    # 边的长度和权重大小成正比
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, edge_color=edge_color)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig("broken_circle.png", format="PNG")
    plt.show()

def main():
    graph = [(0, 1, 6), (0, 2, 1), (0, 3, 5), (1, 2, 5), (1, 4, 3), (2, 3, 5), (2, 4, 6), (2, 5, 4), (3, 5, 2), (4, 5, 6)]
    G = nx.Graph()
    G.add_weighted_edges_from(graph)
    discard_edges = BrokenCircle(graph).broken_circle()
    mst_edges = defaultdict(int)
    for (u, v, w) in graph:
        if discard_edges[(u, v)] == w:
            continue
        else:
            mst_edges[(u, v)] = w
    print('{} | {}'.format(mst_edges, sum(mst_edges.values())))
    draw(G, list(mst_edges.keys()))

if __name__ == '__main__':
    main()