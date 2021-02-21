from glob import glob
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import cpu_count
from scipy.sparse import dok_matrix, save_npz
from sklearn.model_selection import train_test_split

N_JOBS = cpu_count()


def get_dataset():
    info_df = pd.read_csv('data/raw/elf_v2.csv')
    ucf = list(info_df.loc[info_df['Dataset'] == 'UCF', 'md5'].values)
    for md5 in tqdm(ucf):
        if not glob(f'data/raw/malware/{md5}.txt'):
            ucf.remove(md5)
    ucf_paths = [f'data/raw/malware/{md5}.txt' for md5 in ucf]
    beg_paths = [path.replace('\\', '/')
                 for path in glob('data/raw/benign/*.txt')]

    paths = ucf_paths + beg_paths
    label = [1] * len(ucf_paths) + [0] * len(beg_paths)
    train_paths, test_paths, y_train, y_test = train_test_split(
        paths, label, test_size=0.3, random_state=42)
    pd.DataFrame({'path': train_paths, 'label': y_train}).to_csv(
        'data/train.csv', index=None)
    pd.DataFrame({'path': test_paths, 'label': y_test}).to_csv(
        'data/test.csv', index=None)


def get_node_dict():
    train_df = pd.read_csv('data/train.csv')
    paths = train_df['path'].values

    def process_graph(path):
        nodes = set()
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines[2:]:
            e = line.split()
            if len(e) == 2:
                nodes.update(e)
        return list(nodes)

    outputs = Parallel(N_JOBS)(delayed(process_graph)(path)
                               for path in tqdm(paths))
    nodes = set()
    for output in outputs:
        nodes.update(output)
    pd.DataFrame({'nodes': list(nodes)}).to_csv(
        'data/nodes.csv', index=None)


def gen_matrix():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    train_paths = train_df['path'].values
    test_paths = test_df['path'].values
    paths = list(train_paths) + list(test_paths)
    nodes = list(pd.read_csv('data/nodes.csv')['nodes'].values)

    def create_mtx(path, nodes):
        md5 = path.split('/')[-1].replace('.txt', '')
        mtx_path = f'data/matrix/{md5}.npz'
        if glob(mtx_path):
            return

        mtx = dok_matrix((len(nodes), len(nodes)), dtype='int')

        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines[2:]:
            e = line.split()
            if len(e) == 2:
                if e[0] in nodes and e[1] in nodes:
                    u = nodes.index(e[0])
                    v = nodes.index(e[1])
                    mtx[u, v] += 1
        save_npz(mtx_path, mtx.tocsr())

    Parallel(N_JOBS)(delayed(create_mtx)(path, nodes) for path in tqdm(paths))


if __name__ == '__main__':
    # get_dataset()
    # get_node_dict()
    gen_matrix()
