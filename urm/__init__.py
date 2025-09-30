import os
import logging.config
import sys
from datetime import datetime


def setup_logging(default_path='logging.conf', default_level=logging.INFO):
    """显式初始化日志配置（由用户决定是否调用）"""
    path = default_path
    if os.path.exists(path):
        logging.config.fileConfig(path)
    else:
        logging.basicConfig(level=default_level)


def setup_logging_without_conf(log_dir='log', log_name_prefix='app', console_level=logging.INFO, file_level=logging.DEBUG):
    """
    动态设置日志：控制台 + 按日期命名的日志文件
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 生成带日期的日志文件名：app-2025-09-29.log
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{log_name_prefix}-{today}.log")

    # 创建 formatter（和你 conf 中一致）
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(pathname)s - %(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 获取 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 设为最低级别，由 handler 控制实际输出

    # 清除已有 handler（避免重复）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # === 控制台 Handler ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # === 文件 Handler ===
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"日志已初始化，文件路径: {log_file}")