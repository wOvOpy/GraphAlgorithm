import networkx as nx
import matplotlib.pyplot as plt
import sys
from collections import defaultdict


class Prim:
    def __init__(self, graph, num_nodes) -> None:
        self.graph = graph
        self.num_nodes = num_nodes
    
    def mst(self, start_node):
        ''' prim算法生成最小生成树
        
        Parameters
        ----------
        start_node : int
            开始节点

        Returns
        -------
        mst_edges : list, e.g. [(0, 1, 2), (1, 4, 1), ...]
            最小生成树的所有边

        mst_weights : int
            最小生成树所有边上的权重和
        '''
        mst_edges = defaultdict(int)
        
        edge_weight = defaultdict(int) # 边和权重映射
        node_neighbors = defaultdict(set) # 节点和邻居映射
        visited = set()  # 已经访问的节点
        visited.add(start_node)
        count = 1 # 已经访问的节点数目

        # 初始化
        for (u, v, w) in self.graph:
            node_neighbors[u].add(v)
            node_neighbors[v].add(u)
            edge_weight[(u, v)] = edge_weight[(v, u)] = w
            
        while count != self.num_nodes:
            min_weight = sys.maxsize
            cur_node, next_node = -1, -1
            for u in visited:
                for v in node_neighbors[u]:
                    if v not in visited and edge_weight[(u, v)] < min_weight:
                        min_weight = edge_weight[(u, v)]
                        cur_node = u
                        next_node = v     
            visited.add(next_node)
            count += 1
            mst_edges[(cur_node, next_node)] = min_weight
        return mst_edges

def draw(G, color_edges):
    ''' 画一张图，并给指定边着色
    
    Parameters
    ----------
        G : networkx.classes.graph.Graph
            要画的图
        color_edges : list, e.g. [(0, 1), (1, 4), ...]
    
    Returns
    -------
    None
    '''
    edges = list(G.edges) # 图的所有边
    color_edges = set(color_edges) # 要着色的边
    num_edges = G.number_of_edges() # 图所有边的数量
    edge_color = ['b'] * num_edges # 初始化所有的边为某种颜色

    for i in range(num_edges):
        u, v = edges[i][0], edges[i][1]
        # 无向图
        if (u, v) in color_edges or (v, u) in color_edges:
            edge_color[i] = 'r'
    
    # 边的长度和权重大小成正比
    pos = nx.kamada_kawai_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight') # 画权重
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos, with_labels=True, edge_color=tuple(edge_color)) # 画边
    plt.savefig("prim.png", format="PNG")
    plt.show()
    

def main():
    graph = [(0, 1, 6), (0, 2, 1), (0, 3, 5), (1, 2, 5), (1, 4, 3), (2, 3, 5), (2, 4, 6), (2, 5, 4), (3, 5, 2), (4, 5, 6)]
    G = nx.Graph()
    G.add_weighted_edges_from(graph)
    num_nodes = G.number_of_nodes() # 图所有节点的数量
    start_node = 0 # 开始节点
    mst_edges = Prim(graph, num_nodes).mst(start_node)
    print('{} | {}'.format(mst_edges, sum(mst_edges.values())))
    draw(G, list(mst_edges.keys()))

if __name__ == '__main__':
    main()