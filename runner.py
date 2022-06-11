from torch.utils.data import DataLoader, Dataset
from rdflib import Graph
from om.match import aligns, onts, Runner
import pandas as pd
import json
import dill
import base64
from tinydb import TinyDB, Query
from datetime import datetime
import uuid
import multiprocessing as mp
from rdflib import BNode, Literal, URIRef
import time


with open('settings.json') as f:
    settings = json.loads(f.read())

def umap(f, t):
    for p in t:
        yield f(*p)


class MatchDataset(Dataset):

    def __init__(self, r, o1, o2, t_factor=1):
        super().__init__()
        self.transform = None
        self.als = set(umap(lambda x, y: (URIRef(x), URIRef(y)), aligns(r)))
        self.g1 = Graph()
        self.g1.parse(o1)

        self.g2 = Graph()
        self.g2.parse(o2)

        self.data = []

        tc = 0
        fc = 0

        for e1 in set(self.g1.subjects()):
            for e2 in set(self.g2.subjects()):
                if (e1, e2) in self.als:
                    sim = 1
                else:
                    sim = -1

                for _ in range(t_factor if sim == 1 else 1):
                    if sim == 1:
                        tc += 1
                    else:
                        fc += 1
                    self.data.append((e1, e2, sim))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def build_sets(base):
    refs = map(lambda x: x[0], onts(base + 'conference', base + 'reference'))
    refs = map(lambda x: x.split('/')[-1].split('.')[0], refs)

    pairs = set()

    for o1 in refs:
        for o2 in refs:
            if o1 == o2:
                continue
            pairs.add(frozenset((o1, o2)))

    res = []
    for pair in pairs:
        sets = set()
        for x in pair:
            sets.update(set(x.split('-')))
        train = []
        validation1 = []
        validation2 = []
        test = []
        for r, o1, o2 in onts(base + 'conference', base + 'reference'):
            os = set(r.split('/')[-1].split('.')[0].split('-'))
            if r.split('/')[-1].split('.')[0] in pair:
                train.append((r, o1, o2))
            elif len(os.intersection(sets)) == 0:
                test.append((r, o1, o2))
            elif len(os.intersection(sets)) == 1:
                validation2.append((r, o1, o2))
            else:
                validation1.append((r, o1, o2))

        res.append((train, validation1, validation2, test))

    return res


def rank(results):
    res = []
    for k in results:
        data = []
        for r in results[k]:
            m = r[['precision', 'recall', 'f1']].mean()
            data.append([m['precision'], m['recall'], m['f1']])
        res.append([k] + max(data, key=lambda x: x[2]))

    df = pd.DataFrame(res, columns=['Name', 'Precision', 'Recall', 'F1'])

    return df.sort_values('F1', ascending=False)


def run_experiment(request, db_lock):
    builder_data = request['request']['builder'].encode('utf-8')
    model_builder = dill.loads(base64.b64decode(builder_data))
    base = settings['base']
    train_data = build_sets(base)

    folds = []
    for f, (train, validation1, validation2, test) in enumerate(train_data):
        datasets = [MatchDataset(r, o1, o2, t_factor=800) for r, o1, o2 in train]

        models = model_builder.build(datasets)

        results = []
        sets = [('train', train), ('validation1', validation1), ('validation2', validation2), ('test', test)]
        for name, data in sets:
            r1 = dict()
            refs = list(map(lambda x: x[0], data))

            for config, model in models:
                runner = Runner(base + 'conference', base + 'reference', matcher=model)
                res = runner.run(refs=refs)
                res = list(map(lambda y: y.to_json(), res))
                r1[config['id']] = {'config': config, 'result': res}

            results.append((name, r1))

        folds.append({'fold': f, 'result': results})
    db_lock.acquire()
    with TinyDB(settings['db']) as db:
        results = db.table('results')
        results.insert(
            {'id': str(uuid.uuid4()), 'folds': folds, 'end': str(datetime.now()), 'request': request['request']})

        events = db.table('events')
        query = Query()
        events.remove(query.id == request['id'])
    db_lock.release()


def listen(is_running, db_lock):
    workers = []
    while is_running.value == 1:

        for worker in workers:
            if not worker.is_alive():
                workers.remove(worker)

        db_lock.acquire()
        with TinyDB(settings['db']) as db:
            events = db.table('events')
            all_events = events.search(Query().status == 'waiting')

            if len(workers) < 5 and len(all_events) > 0:
                min_event = min(all_events, key=lambda x: x['date'])
                events.update({'status': 'running'}, Query().id == min_event['id'])
                process = mp.Process(target=run_experiment, args=(min_event, db_lock))
                process.start()
                workers.append(process)

            db.close()

        db_lock.release()
        time.sleep(1)
