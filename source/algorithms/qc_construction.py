from .base_algorithm import BaseAlgorithm
from operator import itemgetter
from random import randrange
import random
import pickle

class QuantumAnnealerConstructionHeuristics(BaseAlgorithm):

     def quantum_annealer(self):
        rlt = pickle.load(open('anneal_task.pkl', 'rb'))
        rlt = sorted(rlt, key = lambda i: i['total_distance_with_return'], reverse=True)
        if len(rlt) > 15:
            rlt = random.sample(rlt[:-1], 10) + [rlt[-1]]
        tours = [list(map(lambda x:x+1,row['route'])) for row in rlt]
        best_lengths = [row['total_distance_with_return'] for row in rlt]
        rlt = [self.format_solution(step) for step in tours], best_lengths
        return rlt