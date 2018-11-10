def write_log(date, message, logfile):
    with open(logfile, 'a') as log:
        log.write(date+" "+message+"\n")
