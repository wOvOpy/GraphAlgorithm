import networkx as nx
import matplotlib.pyplot as plt
import sys
from collections import defaultdict


class KM:
    def __init__(self, nodes_one, nodes_two, edges) -> None:
        self.nodes_one = nodes_one # 二部图第一部分节点
        self.nodes_two = nodes_two # 二部图第二部分节点
        self.edges = edges # 所有边（带权重）
        
    def optimal_match(self):
        '''理解该算法建议手动模拟一下代码中的例子
        '''
        match = defaultdict(lambda: None) # 记录节点匹配情况
        sum_weights = 0 # 最优匹配权重总和
        node_neighbors = defaultdict(list) # 节点邻居映射
        edge_weight = defaultdict(int) # 边权重映射
        
        label_one = defaultdict(int) # 第一部分节点的可行顶标
        label_two = defaultdict(int) # 第二部分节点的可行顶标
        slack = defaultdict(lambda: sys.maxsize) # 记录第二部分节点能和第一部分节点匹配还需要多少权重
        one_visited, two_visited = set(), set()
        # 初始化
        for (u, v, w) in self.edges:
            node_neighbors[u].append(v)
            node_neighbors[v].append(u)
            edge_weight[(u, v)] = edge_weight[(v, u)] = w
        for node_one in self.nodes_one:
            max_w = 0
            for node_one_neighbor in node_neighbors[node_one]:
                max_w = max(max_w, edge_weight[(node_one, node_one_neighbor)])
            label_one[node_one] = max_w
        
        def dfs(u):
            one_visited.add(u)
            
            for v in node_neighbors[u]:
                if v in two_visited:
                    continue
                gap = label_one[u]+label_two[v] - edge_weight[(u, v)]
                
                if gap == 0:
                    two_visited.add(v)
                    if match[v] == None or dfs(match[v]):
                        match[v] = u
                        return True # 
                else:
                    slack[v] = min(slack[v], gap)
            return False
           
        for u in self.nodes_one:
            slack.clear()
            while not dfs(u):
                min_gap = sys.maxsize # 最小可以降低多少顶标值能够完成匹配
                for node_two in self.nodes_two:
                    if node_two not in two_visited:
                        min_gap = min(min_gap, slack[node_two])
                for node_one in self.nodes_one:
                    if node_one in one_visited:
                        label_one[node_one] -= min_gap
                for node_two in self.nodes_two:
                    if node_two in two_visited:
                        label_two[node_two] += min_gap
                    else:
                        slack[node_two] -= min_gap
                one_visited.clear()
                two_visited.clear()
        # 求最优匹配的权重和
        for key, value in match.items():
            sum_weights += edge_weight[(value, key)]
        return match, sum_weights

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
        pos[node_two] = [two_x, two_y]
        two_y -= 1
    plt.title('KM Algorithm: Optimal Matching')
    nx.draw(G, pos, with_labels=True, node_color=node_color, edge_color=edge_color)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # label_pos=0.8作用: 防止图中交叉边上权重值重叠
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.8)
    plt.savefig('kuhn_munkras.png', format='PNG')
    plt.show()       
def main():
    nodes_one = [0, 1, 2]
    nodes_two = ['a', 'b', 'c']
    edges = [(0, 'a', 3), (0, 'c', 4), (1, 'a', 2), (1, 'b', 1), (1, 'c', 3), (2, 'c', 5)]
    match, sum_weights = KM(nodes_one, nodes_two, edges).optimal_match()
    match_edges = [(u, v) for u, v in match.items()]
    print('{} | {}'.format(match_edges, sum_weights))
    G = nx.Graph()
    G.add_nodes_from(nodes_one+nodes_two)
    G.add_weighted_edges_from(edges)
    draw(G, nodes_one, nodes_two, match_edges)

if __name__ == '__main__':
    main()
