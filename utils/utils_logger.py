import datetime
import logging

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def logger_info(logger_name, log_path):
    '''set up logger'''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exists!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)