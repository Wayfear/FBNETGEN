import logging


class Logger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            '[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def info(self, info: str):
        self.logger.info(info)
