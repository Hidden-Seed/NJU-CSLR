import logging
import os
import torch.distributed as dist


class Logger(object):
    def __init__(self, log_dir, log_name):

        if self.is_rank0():
            log_path = os.path.join(log_dir, log_name)
            if not os.path.exists(log_path):
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                with open(log_path, "w") as log:
                    pass

            # Configure logging
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

            class ColorFormatter(logging.Formatter):
                RED = '\033[91m'
                RESET = '\033[0m'

                def format(self, record):
                    # 设置高亮颜色（非 INFO 为红色）
                    if record.levelname != 'INFO':
                        record.levelname = f"{self.RED}{record.levelname}{self.RESET}"
                    return super().format(record)

            # 使用自定义格式化器
            formatter = ColorFormatter(
                '%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s: - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Use FileHandler to output to the file.
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)

            # Use StreamHandler to output to the screen.
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)

            # Add two Handlers
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        else:
            self.logger = None

    def info(self, message):
        if self.logger:
            self.logger.info(message)

    def warning(self, message):
        if self.logger:
            self.logger.warning(message)

    def error(self, message):
        if self.logger:
            self.logger.error(message)

    def is_rank0(self):
        try:
            if dist.get_rank() == 0:
                return True
            return False
        except ValueError:
            return True
