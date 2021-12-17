from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.linear_programming import LinearProgramming
from algorithms.tour_construction import TourConstructionHeuristics
from algorithms.qc_construction import QuantumAnnealerConstructionHeuristics

class pyTSP(GeneticAlgorithm, LinearProgramming, TourConstructionHeuristics, QuantumAnnealerConstructionHeuristics):

    algorithms = (
        'nearest_neighbor',
        'nearest_insertion',
        'farthest_insertion',
        'cheapest_insertion',
        'pairwise_exchange',
        'node_insertion',
        'edge_insertion',
        'ILP_solver',
        'quantum_annealer',
        'opt3'
    )
