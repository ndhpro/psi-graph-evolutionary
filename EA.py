from os import replace
import numpy as np
from numpy import random
import pandas as pd
from scipy.sparse import load_npz, lil_matrix

NUM_GENERATION = 500


class Evolutionary:
    def __init__(self, target, label) -> None:
        self.target = self.load_mtx(target)
        self.pop = self.create_pop(label)

    def load_mtx(self, graph_path):
        md5 = graph_path.split('/')[-1].replace('.txt', '')
        return load_npz(f'data/mtx/{md5}.npz')

    def distance(self, x, y):
        dist = np.sqrt((x-y).power(2).sum())
        return dist

    def create_pop(self, label):
        train_df = pd.read_csv('data/train.csv')
        paths = train_df.loc[train_df['label'] == label, 'path'].values
        chosen_paths = random.choice(paths, 100, replace=False)

        pop = []
        for path in chosen_paths:
            mtx = self.load_mtx(path)
            dist = self.distance(mtx, self.target)
            pop.append([mtx, dist])
        pop = sorted(pop, key=lambda x: x[1])

        return pop

    def select(self):
        pop = np.array(self.pop)
        tournament = list(pop[random.choice(len(pop), 5, replace=False)])
        tournament = sorted(tournament, key=lambda x: x[1])
        return tournament[0], tournament[1]

    def crossover(self, parent1, parent2):
        mtx1 = parent1[0]
        mtx2 = parent2[0]
        offstr = lil_matrix(mtx1.shape, dtype='int')

        ind1 = zip(mtx1.nonzero()[0], mtx1.nonzero()[1])
        ind2 = zip(mtx2.nonzero()[0], mtx2.nonzero()[1])
        ind = set()
        ind.update(ind1)
        ind.update(ind2)

        for i in ind:
            if random.uniform() < 0.5:
                offstr[i] = mtx1[i]
            else:
                offstr[i] = mtx2[i]

        offstr = offstr.tocsr()
        offstr_dist = self.distance(offstr, self.target)
        off = [offstr, offstr_dist]

        return off

    def mutate(self, dna):
        mtx = dna[0]
        mutate_mtx = mtx.todok()

        ind = list(zip(mtx.nonzero()[0], mtx.nonzero()[1]))
        if not len(ind):
            return dna
        chosen_i = ind[random.choice(len(ind))]
        mutate_value = random.randint(mtx.min(), mtx.max())
        mutate_mtx[chosen_i] = mutate_value

        dist = self.distance(mutate_mtx, self.target)
        return [mutate_mtx, dist]

    def replace(self):
        sorted_pop = sorted(self.pop, key=lambda x: x[1])
        new_pop = sorted_pop[:100]
        return new_pop

    def run(self):
        fittest = np.inf

        for i in range(NUM_GENERATION):
            # print(f'Generation: {i+1}... Min distance: {self.pop[0][1]}')
            if self.pop[0][1] == 0:
                # print('Found target.')
                return 0
            fittest = min(fittest, self.pop[0][1])

            parent1, parent2 = self.select()

            offstr1 = self.crossover(parent1, parent2)
            offstr2 = self.crossover(parent1, parent2)

            if random.uniform() <= 0.1:
                offstr1 = self.mutate(offstr1)
            if random.uniform() <= 0.1:
                offstr2 = self.mutate(offstr2)

            self.pop.extend([offstr1, offstr2])
            self.pop = self.replace()

        return fittest


if __name__ == '__main__':
    test_path = pd.read_csv('data/test.csv')['path'].values[0]
    print(test_path)

    evo = Evolutionary(test_path, 0)
    mdist = evo.run()
    print(mdist)
