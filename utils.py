import os


def write_log(date, message, logfile):
    with open(logfile, 'a') as log:
        log.write(date+" "+message+"\n")
def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass