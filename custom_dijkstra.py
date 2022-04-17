import sys
import math
import json

import networkx as nx
import matplotlib.pyplot as plt

class Vertex():
    def __init__(self, id: int, coord: list(), adjacent: list()):
        self.id = id # 고유 식별값
        self.coordinate = coord # 해당 Vertex()의 좌표, list

        self.isvisited = False
        self.adjacent = {}  # 인접한 Vertex() 딕셔너리

        for vertex_id in adjacent:
            self.adjacent[vertex_id] = None

    def getbetweendistance(self, dst_vertex): # 객체와 입력받은 dst_vertex의 거리를 계산한다.
        return math.sqrt(math.pow(self.coordinate[0] - dst_vertex.coordinate[0], 2) + math.pow(self.coordinate[1] - dst_vertex.coordinate[1], 2))

    def getcoord(self):
        return self.coordinate

class Graph(object):
    def __init__(self, init_graph: dict()):
        self.nodes = [str(n) for n in range(len(init_graph))]
        # 노드 이름 정의
        self.graph = self.construct_graph(init_graph)

    def construct_graph(self, init_graph):
        # init_graph에 명시된 값을 바탕으로 그래프를 생성한다.
        print(init_graph)
        graph = {}
        for name in init_graph:
            graph[name] = {}

        graph.update(init_graph)

        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value

        return graph

    def get_nodes(self):
        return self.nodes

    def get_outgoing_edges(self, node):
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections

    def value(self, node1, node2): # 두 노드 간 거리에 해당하는 값 리턴
        return self.graph[node1][node2]

class Dijkstra():
    def __init__(self):
        with open("vertexinput.json") as f:
            example_input = json.load(f)

        print(example_input)

        self.init_graph = {}

        for idx, vertex in enumerate(example_input):
            self.init_graph[str(idx)] = {}
            if vertex["adjacent"]:  # 인접한 Vertex가 있는 경우
                for adjacent in vertex["adjacent"]:  # 각 인접 vertex에 대한 dict 생성 및 거리값 초기화
                    self.init_graph[str(idx)][adjacent] = self.calcdistance(example_input, idx, adjacent)

        # print("init_graph")
        # print(init_graph)

        self.graph = Graph(self.init_graph)

        previous_nodes, shortest_path = self.dijkstra_algorithm(graph=self.graph, start_node="4")
        path = self.print_result(previous_nodes, shortest_path, start_node="4", target_node="5")

        # self.init_graph["0"]["4"] = 1
        # self.graph = Graph(self.init_graph)
        #
        # previous_nodes, shortest_path = self.dijkstra_algorithm(graph=self.graph, start_node="4")
        # path = self.print_result(previous_nodes, shortest_path, start_node="4", target_node="5")

        # self.graph_visualize(path) # 생성된 경로 시각화 함수

    def graph_visualize(self, path):
        G = nx.DiGraph()

        edges = list()

        for key, val in self.init_graph.items():
            for subkey, subval in val.items():
                edges.append((key, subkey))

        G.add_edges_from(edges)

        red_edges = list()
        for idx, edge in enumerate(path):
            if idx == len(path) - 1:
                break
            red_edges.append((edge, path[idx + 1]))

        black_edges = [edge for edge in G.edges()]

        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
        plt.show()

    def dijkstra_algorithm(self, graph, start_node):
        unvisited_nodes = list(graph.get_nodes())

        # 이 dict를 통해 각 노드 방문 비용을 절약하고 그래프를 따라 이동할 때 갱신한다.
        shortest_path = {}

        # 이 dict를 통해 지금까지 발견된 노드에 대한 알려진 최단 경로 저장
        previous_nodes = {}

        # 미방문한 노드들에 대해서는 표현가능한 최대 값 사용
        max_value = sys.maxsize
        for node in unvisited_nodes:
            shortest_path[node] = max_value

        # 시작 노드에 대한 최단 경로는 0
        shortest_path[start_node] = 0

        # 모든 노드를 방문할 때 까지 수행
        while unvisited_nodes:
            # The code block below finds the node with the lowest score
            current_min_node = None
            for node in unvisited_nodes:
                if current_min_node == None:
                    current_min_node = node
                elif shortest_path[node] < shortest_path[current_min_node]:
                    current_min_node = node

            # 현재 노드 이웃을 검색하고 거리를 업데이트
            neighbors = graph.get_outgoing_edges(current_min_node)
            for neighbor in neighbors:
                tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
                if tentative_value < shortest_path[neighbor]:
                    shortest_path[neighbor] = tentative_value
                    previous_nodes[neighbor] = current_min_node

            # 이웃을 방문한 후 노드를 "방문함"으로 표시합니다.
            unvisited_nodes.remove(current_min_node)

        return previous_nodes, shortest_path

    def print_result(self, previous_nodes, shortest_path, start_node, target_node):
        path = []
        refined_path = list()
        node = target_node

        while node != start_node: # 시작 노드에 도달할 때 까지 반복
            path.append(node)
            node = previous_nodes[node]

        path.append(start_node)

        print("최단 경로에 대한 거리값 : {}.".format(shortest_path[target_node]))
        print(" -> ".join(reversed(path)))
        for vertex in reversed(path):
            refined_path.append(vertex)
        return refined_path

    def calcdistance(self, input, start, dst):
        """ 입력받은 start, dst Vertex의 거리 값 계산 후 리턴"""
        return math.sqrt(math.pow(input[int(start)]["xy"][0] - input[int(dst)]["xy"][0], 2) + math.pow(input[int(start)]["xy"][1] - input[int(dst)]["xy"][1], 2))


if __name__ == '__main__':
    dijkstra = Dijkstra()