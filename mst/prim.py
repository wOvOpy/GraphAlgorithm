import networkx as nx
import matplotlib.pyplot as plt
import sys
from collections import defaultdict


class prim:
    def __init__(self, graph, nodes) -> None:
        self.graph = graph
        self.nodes = nodes
    
    def mst(self, start_node):
        '''
        start_node: 生成最小生成树的开始节点


        return:
        mst_weight: 最小生成树的权重列表
        mst_edges: 最小生成树的边列表
        '''
        # 边和权重映射
        edge_weight = defaultdict(int)
        # 节点和邻居映射
        node_neighbors = defaultdict(set)
        # 最小生成树的边
        mst_edges = []
        # 最小生成树的权重
        mst_weights = []
        # 已经访问的节点
        visited = set()
        visited.add(start_node)

        for (u, v, w) in self.graph:
            node_neighbors[u].add(v)
            node_neighbors[v].add(u)
            edge_weight[(u, v)] = edge_weight[(v, u)] = w
        count = 1
        while count != len(self.nodes):
            min_weight = sys.maxsize
            cur_node, next_node = -1, -1
            for u in visited:
                for v in node_neighbors[u]:
                    if v not in visited and edge_weight[(u, v)] < min_weight:
                        min_weight = edge_weight[(u, v)]
                        cur_node = u
                        next_node = v 
            mst_weights.append(min_weight)     
            visited.add(next_node)
            mst_edges.append((cur_node, next_node))
            count += 1
        return mst_weights, mst_edges

# networkx画图函数，最小生成树边标红
def draw(G, color_edges):
    # 图的所有边
    edges = list(G.edges)
    # 要着色的边
    color_edges = set(color_edges)
    n = len(edges)
    edge_color = ['b'] * n

    for i in range(n):
        u, v = edges[i][0], edges[i][1]
        if (u, v) in color_edges or (v, u) in color_edges:
            edge_color[i] = 'r'
    
    # 边的长度和权重大小成正比
    pos = nx.kamada_kawai_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # 画权重
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # 画边
    nx.draw(G, pos, with_labels=True, edge_color=tuple(edge_color))
    plt.savefig("prim.png", format="PNG")
    plt.show()
    


def main():
    graph = [(0, 1, 6), (0, 2, 1), (0, 3, 5), (1, 2, 5), (1, 4, 3), (2, 3, 5), (2, 4, 6), (2, 5, 4), (3, 5, 2), (4, 5, 6)]
    G = nx.Graph()
    G.add_weighted_edges_from(graph)
    nodes = G.nodes
    mst_weights, mst_edges = prim(graph, nodes).mst(0)
    print('prim算法生成的最小生成树包含了这些边{}'.format(mst_edges))
    print('边对应的权重为：{}，总和为：{}'.format(mst_weights, sum(mst_weights)))
    draw(G, mst_edges)

if __name__ == '__main__':
    main()