from fastapi import FastAPI
from tinydb import TinyDB
from datetime import datetime
import uuid
from runner import listen
import multiprocessing as mp
import time

app = FastAPI()

db_lock = mp.Lock()
state = mp.Value('i', 1)
process = mp.Process(target=listen, args=(state, db_lock))


@app.on_event('startup')
async def app_start():
    process.start()
    pass


@app.on_event('shutdown')
async def app_end():
    state.value = 0
    pass


@app.post('/run_experiment')
async def run_experiment(body: dict):
    db_lock.acquire()
    with TinyDB('db.json') as db:
        events = db.table('events')
        events.insert({'id': str(uuid.uuid4()), 'date': time.time(), 'request': body, 'status': 'waiting'})
    db_lock.release()
    return {'status': 'ok'}
