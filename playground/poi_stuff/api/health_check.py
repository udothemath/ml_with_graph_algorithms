import sys
import psutil
import os
import logging
import socket


logger = logging.getLogger()
threshold = os.getenv('MEMORY_THRESHOLD')
if threshold:
    mem_threshold = int(threshold)
else:
    mem_threshold = 80

if os.getenv('PROCESS_CHECK'):
    process_name = os.getenv('PROCESS_NAME')
    process_num = int(os.getenv('PROCESS_NUM'))
else:
    process_name = 'gunicorn'
    process_num = 4


def check_memory():
    """
    Use /sys/fs/cgroup/memory to get pod memory info
    """

    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
        mem_total = f.read()
    with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
        mem_used = f.read()

    mem_usage = float("%.2f" % (100 * int(mem_used) / int(mem_total)))
    logger.info(f'Memory usage: {mem_usage}%, threshold: {mem_threshold}%')
    if mem_usage > mem_threshold:
        logger.error('Memory usage more than threshold')
        sys.exit(1)


def check_process():
    proc_count = 0
    for proc in psutil.process_iter():
        try:
            if process_name in proc.name():
                proc_count += 1
                logger.info(f'Get {process_name} pid: {proc.pid}')
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, psutil.ZombieProcess):
            pass
    if proc_count < process_num:
        logger.error(f'{process_name} process < {process_num}')
        sys.exit(1)


def check_port(port_num):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip = "0.0"
    result = sock.connect_ex((f'{ip}.{ip}', port_num))
    if result:
        logger.error(f'Check port {port_num} failed')
        sys.exit(1)


def run():
    check_memory()
    check_process()
    if os.getenv('PORT_CHECK'):
        port_num = int(os.getenv('PORT')) if os.getenv('PORT') else 8000
        check_port(port_num)
    else:
        logger.info('skip port check')


if __name__ == '__main__':
    run()
