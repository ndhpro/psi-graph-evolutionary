from os import replace
import numpy as np
from numpy import random
import pandas as pd
from scipy.sparse import load_npz, dok_matrix


class Evolutionary:
    def __init__(self, target, label) -> None:
        self.target = self.load_mtx(target)
        self.pop = self.create_pop(label)

    def load_mtx(self, graph_path):
        md5 = graph_path.split('/')[-1].replace('.txt', '')
        return load_npz(f'data/matrix/{md5}.npz')

    def distance(self, a, b):
        return np.sqrt(np.sum((a-b)**2))

    def create_pop(self, label):
        train_df = pd.read_csv('data/train.csv')
        paths = train_df[train_df['label'] == label, 'path'].values
        chosen_paths = random.choice(paths, 100, replace=False)

        pop = []
        for path in chosen_paths:
            mtx = self.load_mtx(path)
            dist = self.distance(mtx, self.target)
            pop.append((mtx, dist))

        return pop

    def select(self):
        tournament = random.choice(self.pop, 5, replace=False)
        tournament = sorted(tournament, key=lambda x: x[2])
        return tournament[0]

    def crossover(self, parent1, parent2):
        mtx1 = parent1[0]
        mtx2 = parent2[0]
        offstr = dok_matrix(mtx1.shape, dtype='int')

        ind1 = zip(mtx1.nonzero()[0], mtx1.nonzero()[1])
        ind2 = zip(mtx2.nonzero()[0], mtx2.nonzero()[1])
        ind = set()
        ind.update(ind1)
        ind.update(ind2)

        for i in ind:
            if random.uniform() > 0.5:
                offstr[i] = mtx1[i]
            else:
                offstr[i] = mtx2[i]

        offstr = offstr.tocsr()
        offstr_dist = self.distance(offstr, self.target)
        return (offstr, offstr_dist)

    def mutate(self, dna):
        mtx = dna[0]
        mutate_mtx = mtx.todok()

        ind = zip(mtx.nonzero()[0], mtx.nonzero()[1])
        chosen_i = random.choice(ind)
        mutate_value = random.randint(mtx.min(), mtx.max())
        mutate_mtx[chosen_i] = mutate_value

        dist = self.distance(mutate_mtx, self.target)
        return (mutate_mtx, dist)
