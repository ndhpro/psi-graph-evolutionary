from EA import Evolutionary
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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


def get_res():
    y_true = []
    y_pred = []
    res_paths = glob('results/detail/*.txt')
    for path in tqdm(res_paths):
        with open(path, 'r') as f:
            lines = f.readlines()
        try:
            y_pred.append(int(lines[2].split('=')[-1]))
            y_true.append(int(lines[3].split('=')[-1]))
        except:
            print(path)

    target_names = ['benign', 'malware']
    cnf_matrix = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(
        y_true, y_pred, target_names=target_names, digits=4)
    tn, fp, fn, tp = cnf_matrix.ravel()
    fpr = fp / (fp + tn)
    roc_auc = roc_auc_score(y_true, y_pred)

    with open('results/report.txt', 'w') as f:
        f.write(f'Classification report:\n')
        f.write(clf_report + '\n')
        f.write('roc auc: %.4f\n' % roc_auc)
        f.write('fpr    : %.4f\n\n' % fpr)
        f.write(f'Confusion matrix:\n')
        f.write(str(cnf_matrix))


if __name__ == '__main__':
    # run_ea()
    get_res()
