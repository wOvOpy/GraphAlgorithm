from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

INF = 20211120


class Dijkstra:
    def __init__(self, graph, num_nodes) -> None:
        self.graph = graph
        self.num_nodes = num_nodes # 图中所有节点的数目

    def shortest_path(self, start_node):
        paths = [-1] * self.num_nodes # 记录路径，初始化全为：-1
        distance = [INF] * self.num_nodes # 初始化最短距离全为：INF
        edge_weight = dict() # 边和权重映射
        node_neighbors = defaultdict(set) # 节点和其邻居映射
        # 初始化
        for (u, v, w) in self.graph:
            # 有向图
            edge_weight[(u, v)] = w
            node_neighbors[u].add(v)

        for neighbor in node_neighbors[start_node]:
            paths[neighbor] = start_node
            distance[neighbor] = edge_weight[(start_node, neighbor)]

            visited = [start_node] # 已经访问的节点
            not_visit = [_ for _ in range(self.num_nodes) if _ != start_node] # 还没有访问的节点

            while len(not_visit):
                min_w_node = not_visit[0] # min_w_node : 某阶段开始节点start_node到它的距离最短的点
                for i in not_visit:
                    if distance[i] < distance[min_w_node]:
                        min_w_node = i
                not_visit.remove(min_w_node)
                visited.append(min_w_node)

                # 更新最短距离和最短路径
                for i in not_visit:
                    if distance[min_w_node]+edge_weight.get((min_w_node, i), INF) < distance[i]:
                        distance[i] = distance[min_w_node]+edge_weight[(min_w_node, i)]
                        paths[i] = min_w_node
        return paths, distance

def get_nodes_edges(paths, end_node):
    '''由paths得出最短路径所有节点和所有边
    
    Parameters
    ----------
        paths: list
            记录路径的一维列表
        end_node: int
            结束节点
    
    Returns
    -------
        nodes: list
            最短路径上的所有节点
        edges: list
            最短路径上的所有边
    
    '''
    nodes, edges = [], []
    v = end_node
    nodes.append(v)
    while paths[end_node] != -1:
        u = paths[end_node]
        nodes.append(u)
        edges.append((u, v))
        end_node = u
        v = u
    return nodes[::-1], edges[::-1]

def draw(DG, color_nodes, color_edges):
    '''画一张图，并给指定的节点和边着色
    Parameters
    ----------
    DG : networkx.classes.digraph.DiGraph（有向图）
    color_nodes : list, e.g. [0, 1, 2, ...]
        指定着色的节点
    color_edges : list, e.g. [(0, 1), (2, 5), ...]
        指定着色的边
        
    Returns
    -------
    None
    '''
    print(type(DG))
    edges = list(DG.edges) # 图的所有边
    num_nodes = DG.number_of_nodes() # 图所有节点的数量
    num_edges = DG.number_of_edges() # 图所有边的数量
    node_color = ['b'] * num_nodes # 初始所有节点为某一种颜色
    edge_color = ['b'] * num_edges # 初始所有边为某一种颜色

    # 指定的节点着色
    for i in color_nodes:
        node_color[i] = 'r'

    # 指定的边着色
    for i in range(num_edges):
        u, v = edges[i][0], edges[i][1]
        if (u, v) in set(color_edges):
            edge_color[i] = 'r'
            
    pos = nx.circular_layout(DG)
    plt.title('Digraph-Dijkstra')
    nx.draw(DG, pos, with_labels=True, node_color=node_color, edge_color=edge_color) # 画图
    edge_labels = nx.get_edge_attributes(DG, 'weight')
    edge_labels = { (key[0],key[1]): "w:"+str(edge_labels[key]) for key in edge_labels }
    nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels) # 画权重
    plt.savefig('dijkstra.png', format='PNG')
    plt.show()


def main():
    num_nodes = 6
    graph = [(0, 1, 1), (0, 2, 12), (1, 2, 9), (1, 3, 3), (2, 4, 5), (2, 3, 4), (3, 4, 13), (3, 5, 15), (4, 5, 4)]

    start_node = 0 # 开始节点
    paths, distance = Dijkstra(graph, num_nodes).shortest_path(start_node)
    for end_node in range(num_nodes):
        if start_node != end_node:
            print('{}->{}: {} | {}'.format(start_node, end_node, get_nodes_edges(paths, end_node), distance[end_node]))
    color_nodes, color_edges = get_nodes_edges(paths, end_node=5)
    DG = nx.DiGraph()
    DG.add_weighted_edges_from(graph)
    draw(DG, color_nodes, color_edges)

if __name__ == '__main__':
    main()
