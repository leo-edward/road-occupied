import os
import re
import time
import logging
import configparser
from logging.handlers import TimedRotatingFileHandler


class Log():
    
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger()
        self.logger.root.setLevel(logging.NOTSET)
        self.logger.propagate = False

    def logs_setup(self):
        path = os.path.abspath(".")
        config_path = os.path.join(path, 'configs', 'config.ini')
        config = configparser.ConfigParser()
        config.read(config_path,encoding='utf-8')
        interval = config.get("Log", "rotation_days")
        save_days = config.get("Log", "delete_old_logs")

        today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        logName = self.name + "-" + today + ".log"
        if not os.path.exists("./logs/"):
            os.makedirs("./logs/")
        if not os.path.exists("./logs/{}".format(logName)):
            reportFile = open("./logs/{}".format(logName), 'w')
            reportFile.close()
        log_path = os.path.join("./logs/{}".format(logName))
        file_handler = TimedRotatingFileHandler(
            filename=log_path, when="D", interval=int(interval), backupCount=int(save_days),
            encoding='utf8'
        )
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        console = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        return self.logger