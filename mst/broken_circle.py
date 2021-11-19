import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# 破圈法求解最小生成树
class brokenCircle:
    def __init__(self, graph) -> None:
        self.graph = graph

    def broken_circle(self):
        node_neighbors = defaultdict(set)
        weights = []
        weight_edges = defaultdict(set)
        discard_edges = []
        for (u, v, w) in self.graph:
            node_neighbors[u].add(v)
            node_neighbors[v].add(u)
            weights.append(w)
            weight_edges[w].add((u, v))
        weights.sort(reverse=True)
        visited = set()
        # 广度优先搜索 function: start节点能否到达end节点，如果可以，返回True, 否则返回False
        def bfs(start, end):
            queue = []
            visited = set()
            queue.append(start)
            while queue:
                cur = queue.pop(-1)
                visited.add(cur)
                if cur == end:
                    return True
                for cur_neighbor in node_neighbors[cur]:
                    if cur_neighbor not in visited:
                        queue.append(cur_neighbor)
            return False
                
        for w in weights:
            for edge in weight_edges[w]:
                if edge not in visited:
                    u, v = edge[0], edge[1]
                    node_neighbors[u].remove(v)
                    node_neighbors[v].remove(u)
                    if bfs(u, v):
                        discard_edges.append((u, v))
                    else:
                        node_neighbors[u].add(v)
                        node_neighbors[v].add(u)    
                    visited.add((u, v))
        return discard_edges
                        

def draw(G, color_edges):
    # 图的所有边
    edges = list(G.edges)
    # 要着色的边
    color_edges = set(color_edges)
    n = len(edges)
    edge_color = ['r'] * n

    for i in range(n):
        u, v = edges[i][0], edges[i][1]
        if (u, v) in color_edges or (v, u) in color_edges:
            edge_color[i] = 'b'
    
    # 边的长度和权重大小成正比
    pos = nx.kamada_kawai_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # 画权重
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # 画边
    nx.draw(G, pos, with_labels=True, edge_color=tuple(edge_color))
    plt.savefig("broken_circle.png", format="PNG")
    plt.show()


def main():
    graph = [(0, 1, 6), (0, 2, 1), (0, 3, 5), (1, 2, 5), (1, 4, 3), (2, 3, 5), (2, 4, 6), (2, 5, 4), (3, 5, 2), (4, 5, 6)]
    G = nx.Graph()
    G.add_weighted_edges_from(graph)
    nodes = G.nodes
    edges = G.edges
    discard_edges = brokenCircle(graph).broken_circle()
    mst_edges = set(edges) - set(discard_edges)
    print('最小生成树包含了这些边{}'.format(mst_edges))
    draw(G, discard_edges)

if __name__ == '__main__':
    main()