pip install billiard
import os
from billiard.pool import Pool

# process啟動時的callback function 
def init():
    pid = os.getpid()
    print(f'[init] pid: {pid}')
    
# process結束的callback function 
def on_exit(pid, exitcode):
    print(f'[on_exit] pid: {pid} , exitcode: {exitcode}')
    
# 每個process要執行的function
def f(x):
    pid = os.getpid()
    print(f'[f] input: {x} pid: {pid}')
    return x ** 2

# 平行處理寫法: 把1~13進行平方 
with Pool(cpu_count, initializer = init, on_process_exit = on_exit) as p:
    ans_gen = p.imap(f, range(13))
    ans = list(ans_gen)