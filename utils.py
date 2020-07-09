import os
from datetime import date
from io import StringIO

import pandas as pd
from google.cloud import storage
from tqdm import tqdm


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def download():
    storage_client = storage.Client.from_service_account_json(find('secretgc_ip.json', '/home'))

    files = os.listdir('./results')

    blobs = storage_client.list_blobs('ip-free-2')
    for blob in tqdm(blobs):
        if blob.name not in files:
            blob.download_to_filename(os.path.join('results', blob.name))


def upload():
    storage_client = storage.Client.from_service_account_json(find('secretgc_ip.json', '/home'))
    bucket = storage_client.bucket('ip-free-2')

    files = os.listdir('.')
    files = [f for f in files if 'csv' in f]

    for file in tqdm(files):
        data = pd.read_csv(os.path.join('.', file))
        name = file.split('.')[0]
        f = StringIO()
        data.to_csv(f, index_label=False)
        f.seek(0)
        blob = bucket.blob(name + '_' + str(date.today()) + '.csv')
        blob.upload_from_file(f, content_type='text/csv')


def plot_results(file_name, true, plot_lim):
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import numpy as np

    data = pd.read_csv(os.path.join('results', file_name))
    data = data.sort_values(by='loss').reset_index(drop=True)
    best = data.solution[0].replace('[', '').replace(']', '').split(',')
    best = [float(b) for b in best]
    worst = data.solution.iloc[-1].replace('[', '').replace(']', '').split(',')
    worst = [float(w) for w in worst]

    plt.rcParams['figure.figsize'] = 7, 7
    plt.style.use('seaborn-white')
    fig, ax = plt.subplots()
    plt.plot(np.linspace(0, 1, 10000), true(np.linspace(0, 1, 10000)), c='#46bac2', label='true')
    plt.plot(np.linspace(0, 1, 10000), worst, ':', c='#ae2c87', label='worst')
    plt.plot(np.linspace(0, 1, 10000), best, '--', c='#4378bf', label='best')
    plt.ylim(-0.1, plot_lim[0])
    plt.legend(loc='upper left', prop={'size': 16})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plot_name = file_name.split('.')[0] + '.png'
    plt.savefig(os.path.join('./plots/estimate', plot_name))
    plt.clf()

    plt.rcParams['figure.figsize'] = 7, 7
    plt.style.use('seaborn-white')
    fig, ax = plt.subplots()
    plt.scatter(np.sqrt(data.oracle_loss), np.sqrt(data.loss))
    plt.xlim(0, plot_lim[1])
    plt.ylim(0, plot_lim[1])
    plt.plot(np.linspace(-0.05, 1.5, 10000), np.linspace(-0.05, 1.5, 10000))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plot_name = file_name.split('.')[0] + '_loss.png'
    plt.savefig(os.path.join('./plots/loss', plot_name))
    plt.clf()


def filter_files(method='Landweber', size='10000', function='BIMODAL', tau='10', date1='2020-07-08', date2='2020-07-08'):
    files = os.listdir('results/')
    files = [f for f in files if 'csv' in f]
    files = [f for f in files if 'Test' not in f]
    files = [f for f in files if method in f]
    files = [f for f in files if 'size_' + size + '_' in f]
    files = [f for f in files if function in f]
    files = [f for f in files if 'tau_' + tau in f]
    files = [f for f in files if '05' not in f]
    files = [f for f in files if date1 in f or date2 in f]
    for f in files:
        print(f)
    return files, function
