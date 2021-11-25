import networkx as nx
import matplotlib.pyplot as plt



class Bellman_ford:
    def __init__(self, nodes, edges, start_node) -> None:
        self.nodes = nodes
        self.edges = edges
        self.start_node = start_node
        
    def shortest_path(self):
        num_node = len(self.nodes)
        num_edge = len(self.edges)
        distance, parent = [None] * num_node, [None] * num_node
        distance[self.start_node] = 0
        times, flag = 0, True

        # 松弛函数
        def slack(edge, distance, parent):
            u, v, w = edge[0], edge[1], edge[2]
            if distance[u] == None:
                return False
            elif distance[v] == None or distance[u]+w < distance[v]:
                distance[v] = distance[u]+w
                parent[v] = u
                return True
            return False
            

        while flag and times < num_node-1:
            flag = False
            for i in range(num_edge):
                if slack(self.edges[i], distance, parent) and not flag:
                    flag = True
            times += 1
        # 判断是否含有负数权回路
        for i in range(num_edge):
            u, v, w = self.edges[i][0], self.edges[i][1], self.edges[i][2]
            if distance[u]+w < distance[v]:
                return False, distance, parent
        return True, distance, parent


def get_nodes_edges(parent, end_node):
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
    while parent[end_node] != None:
        u = parent[end_node]
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
    plt.title('Digraph: Shortest Path')
    nx.draw(DG, pos, with_labels=True, node_color=node_color, edge_color=edge_color) # 画图
    edge_labels = nx.get_edge_attributes(DG, 'weight')
    # edge_labels = { (key[0],key[1]): "w:"+str(edge_labels[key]) for key in edge_labels }
    nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels) # 画权重
    plt.savefig('bellman_ford.png', format='PNG')
    plt.show()

def main():
    '''test0: 不含负权回路的例子
    '''
    nodes = [0, 1, 2, 3, 4, 5, 6]
    edges = [(0, 1, 1), (0, 2, 12), (1, 2, 9), (1, 3, 3), (2, 4, 5), (2, 3, 4), (3, 4, 13), (3, 5, 15), (4, 5, 4)]
    '''test1: 含有负权回路的例子
    '''
    # nodes = [0, 1, 2]
    # edges = [(0, 1, 1), (1, 2, 2), (2, 0, -4)]
    start_node = 0 # 开始节点
    flag, distance, parent = Bellman_ford(nodes, edges, start_node).shortest_path()
    if flag:
        for i in range(len(nodes)):
            if start_node != nodes[i]:
                print('{}->{} {} | {}'.format(start_node, nodes[i], get_nodes_edges(parent, nodes[i]), distance[i]))
    else:
        print('图中含有负权回路')
    end_node = 5 # 终止节点（start_node->end_node的最短路径着色）
    pass_nodes, pass_edges = get_nodes_edges(parent, end_node)
    DG = nx.DiGraph()
    # 可以画出孤立节点
    DG.add_nodes_from(nodes)
    DG.add_weighted_edges_from(edges)
    draw(DG, pass_nodes, pass_edges)

if __name__ == '__main__':
    main()