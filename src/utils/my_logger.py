import logging
import datetime
import re


class MyLogger:
    def __init__(self, filename=None):
        if filename:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{current_time}.log"

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        if filename:
            # 创建文件处理器
            file_handler = logging.FileHandler(filename)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            # 将文件处理器添加到日志记录器
            self.logger.addHandler(file_handler)
        else:
            # 创建命令行处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)

            # 将命令行处理器添加到日志记录器
            self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    @staticmethod
    def parse_log_file(file_path):
        log_entries = []

        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+)', line)
                if match:
                    timestamp = match.group(1)
                    level = match.group(2)
                    message = match.group(3)
                    log_entry = {
                        'timestamp': timestamp,
                        'level': level,
                        'message': message
                    }
                    log_entries.append(log_entry)

        return log_entries

