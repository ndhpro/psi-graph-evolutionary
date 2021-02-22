from EA import Evolutionary
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from glob import glob

N_JOBS = cpu_count()


def run_ea():
    test_df = pd.read_csv('data/test.csv')
    test_paths = test_df['path'].values

    def predict(path):
        md5 = path.split('/')[-1].replace('.txt', '')
        result_path = f'results/detail/{md5}.txt'
        if glob(result_path):
            return

        algo = Evolutionary(path, label=0)
        beg_dist = algo.run()

        algo = Evolutionary(path, label=1)
        mal_dist = algo.run()

        pred = 1 if mal_dist < beg_dist else 0
        label = 1 if 'malware' in path else 0

        with open(result_path, 'w') as f:
            f.write(f'{mal_dist=}\n')
            f.write(f'{beg_dist=}\n')
            f.write(f'{pred=}\n')
            f.write(f'{label=}\n')

    Parallel(N_JOBS)(delayed(predict)(path)
                     for path in tqdm(test_paths))


if __name__ == '__main__':
    run_ea()
