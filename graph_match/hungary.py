from dgl.convert import graph
import torch
import networkx
import dgl
import numpy as np

class hungary:
    def __init__(self, graph) -> None:
        self.graph = graph
            
    def biggest_match(self):
        m, n = len(self.graph), len(self.graph[0])
        match = [-1] * m
        def dfs(i, visited) -> bool:
            for j in range(n):
                if self.graph[i][j] and j not in visited:
                    visited.add(j)
                    if match[j] == -1 or dfs(match[j]):
                        match[j] = i
                        return True
            return False
               
        match_count = 0
        for i in range(m):
            visited = set()
            if dfs(i, visited):
                match_count += 1
        return match_count, match
    
def main():
    graph = [[False, False, True, False, True]
             , [False, True, False, False]
             , [True, False, True, False]
             , [False, False, False, True]]

    match_count, match = hungary(graph).biggest_match()
    print('最大匹配长度为：{}'.format(match_count))
    edges = []
    for i in range(len(graph)):
        if match[i] != -1:
            edges.append((i, match[i]))
    print(edges)

if __name__ == '__main__':
    main()
