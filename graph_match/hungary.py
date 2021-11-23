import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class Hungary:
    def __init__(self, nodes_one, nodes_two, edges) -> None:
        self.nodes_one = nodes_one
        self.nodes_two = nodes_two
        self.edges = edges

    def max_match(self):
        '''求最大匹配边的数量，标记匹配的节点
        '''
        match = defaultdict(lambda: -1) #记录节点匹配
        num_match = 0 #最大匹配数量
        node_neighbors = defaultdict(list) # 节点邻居映射

        for (u, v) in self.edges:
            node_neighbors[u].append(v)
            node_neighbors[v].append(u)
        
        def dfs(u):
            for v in node_neighbors[u]:
                if v not in visited:
                    visited.add(v)
                    if match[v] == -1 or dfs(match[v]):
                        match[v] = u
                        return True
            return False

        for node in self.nodes_one:
            visited = set()
            if (dfs(node)):
                num_match += 1
        return num_match, match

def draw(G, nodes_one, nodes_two, color_edges):
    nodes = list(G.nodes)
    edges = list(G.edges)
    num_node = len(nodes)
    num_edge = len(edges)
    node_color = ['b'] * num_node
    edge_color = ['b'] * num_edge

    for i in range(0, num_node):
        if isinstance(nodes[i], type(nodes_one[0])):
            node_color[i] = 'r'

    for i in range(num_edge):
        u, v = edges[i][0], edges[i][1]
        # 无向图
        if (u, v) in color_edges or (v, u) in color_edges:
            edge_color[i] = 'r'
    '''
    自定义pos
    '''
    # 对matplotlib不太熟悉，布局有待改进
    pos = dict()
    size = max(len(nodes_one), len(nodes_two)) + 2
    one_x, two_x = size//3, 2*size // 3
    one_y, two_y = size-1, size-1
    for node_one in nodes_one:
        pos[node_one] = [one_x, one_y]
        one_y -= 1

    for node_two in nodes_two:
        pos[node_two] = [two_x // 3, two_y]
        two_y -= 1
    # print(pos)
    nx.draw(G, pos, with_labels=True, node_color=node_color, edge_color=edge_color)
    plt.savefig('hungary.png', format='PNG')
    plt.show()
    

def main():
    ''' 测试用例0
    '''
    # nodes_one = [0, 1, 2, 3]
    # nodes_two = ['0', '1', '2', '3']
    # edges = [(0, '0'), (0, '1'), (1, '1'), (1, '2'), (2, '0'), (2, '1'), (3, '2')]

    ''' 测试用例1
    '''
    nodes_one = [0, 1, 2, 3] 
    nodes_two = ['a', 'b', 'c', 'd']
    edges = [(0, 'a'), (0, 'b'), (1, 'b'), (1, 'c'), (2, 'a'), (2, 'b'), (3, 'c'), (3, 'd')]
    
    num_match, match = Hungary(nodes_one, nodes_two, edges).max_match()
    match_edges = []
    for key, value in match.items():
        match_edges.append((value, key))
    print('{} | {}'.format(match_edges, num_match))

    G = nx.Graph()
    G.add_nodes_from(nodes_one+nodes_two)
    G.add_edges_from(edges)
    draw(G, nodes_one, nodes_two, match_edges) 

if __name__ == '__main__':
    main()
