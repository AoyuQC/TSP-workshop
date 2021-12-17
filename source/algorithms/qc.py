# dependance
import numpy as np
import pandas as pd
import networkx as nx
from braket.ocean_plugin import BraketDWaveSampler
from dwave.system.composites import EmbeddingComposite
from os.path import abspath, dirname, join, pardir

seed = 1024

import sys
sys.dont_write_bytecode = True
path_app = dirname(abspath(__file__))
path_parent = abspath(join(path_app, pardir))
if path_app not in sys.path:
    sys.path.append(path_app)

from scripts.plots import plot_cities
from utils_tsp import get_distance, traveling_salesperson


class qcANN():

    def __init__(self, s3_folder, tsp):
        self.cities = tsp.cities
        self.distances = tsp.distances
        print(self.cities)
        print(self.distances)
        # for cityA in self.cities:
        #     for cityB in self.cities:
        #         print(f"{cityA} v.s {cityB} distance: {self.distances[cityA][cityB]}")

        self.answer = None
        print(f"Try to solve tsp of {len(self.cities)}")
        self.sampler = EmbeddingComposite(BraketDWaveSampler(
            s3_folder, 'arn:aws:braket:::device/qpu/d-wave/Advantage_system4'))

        self.city_map = {}
        self.num_shots = 1000
        self.total_dist = None
        self.distance_with_return = None
        self.optimize_routes = []

        self.solve_tsp()

    def solve_tsp(self):
        distance_matrix = self.get_distance_matrix_v2()
        data = pd.DataFrame(distance_matrix)

        G = nx.from_pandas_adjacency(data)

        # get corresponding QUBO step by step
        N = G.number_of_nodes()

        # set parameters
        lagrange = None
        weight = 'weight'

        start_city = 0

        if lagrange is None:
            # If no lagrange parameter provided, set to 'average' tour length.
            # Usually a good estimate for a lagrange parameter is between 75-150%
            # of the objective function value, so we come up with an estimate for
            # tour length and use that.
            if G.number_of_edges() > 0:
                lagrange = G.size(weight=weight) * G.number_of_nodes() / G.number_of_edges()
            else:
                lagrange = 2

        print('Running quantum annealing for TSP with Lagrange parameter=', lagrange)
        route_list = traveling_salesperson(G, self.sampler, lagrange=lagrange,
                                      start=start_city, num_reads=self.num_shots, answer_mode="histogram")

        # print distance
        min_distance = 999999999
        min_route = []
        for route in route_list:
            self.total_dist, self.distance_with_return = get_distance(route, data)
            route_anwser = {}
            route_anwser['route'] = route
            route_anwser['total_distance'] = self.total_dist
            route_anwser['total_distance_with_return'] = self.distance_with_return
            if self.distance_with_return < min_distance:
                min_distance = self.distance_with_return
                min_route = route
            
            self.optimize_routes.append(route_anwser)
        print(f"min route {min_route} with distance {min_distance}")

            # print route
            # print(
            #     f'Route found with D-Wave: {route}, with total distance {self.total_dist} and the distance with return {self.distance_with_return}')

    # helper function
    def create_cities(self, N):
        """
        Creates an array of random points of size N.
        """
        cities = []
        np.random.seed(seed)
        for i in range(N):
            cities.append(np.round((np.random.rand(2) * 100), 2))
        return np.array(cities)

    def get_distance_matrix_v2(self):
        number_of_cities = len(self.cities)
        matrix = np.zeros((number_of_cities, number_of_cities))
        # build city map
        for city in self.cities:
            if city not in self.city_map:
                self.city_map[len(self.city_map)] = city

        for i in range(number_of_cities):
            for j in range(i, number_of_cities):
                matrix[i][j] = self.distances[self.city_map[i]][self.city_map[j]]
                matrix[j][i] = matrix[i][j]
        return matrix
