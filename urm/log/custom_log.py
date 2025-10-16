import os
import logging
import sys
from datetime import datetime
import threading

# 全局变量：记录当前进程/线程的 seed（用于日志）
_local = threading.local()


def set_remark_seed_for_logging(remark, seed):
    _set_current_remark_for_logging(remark)
    _set_current_seed_for_logging(seed)


def _set_current_remark_for_logging(remark):
    _local.remark = remark


def _set_current_seed_for_logging(seed):
    """在当前进程/线程中设置 seed，供日志 formatter 使用"""
    _local.seed = seed


def get_current_remark_for_logging():
    return getattr(_local, "remark", None)


def get_current_seed_for_logging():
    return getattr(_local, 'seed', None)


class CustomFormatter(logging.Formatter):
    """自定义 Formatter，在日志中插入 [Seed XXX]"""

    def format(self, record):
        remark = get_current_remark_for_logging()
        if remark is not None:
            record.msg = f"[Remark: {remark}] {record.msg}"
        return super().format(record)


def setup_shared_logging(
        log_dir='log',
        log_name_prefix='app',
        console_level=logging.INFO,
        file_level=logging.DEBUG
):
    """
    初始化共享日志系统（主进程调用一次）
    所有子进程/线程复用此配置，通过 set_current_seed_for_logging(seed) 注入 seed
    """
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{log_name_prefix}-{today}.log")

    # 使用自定义 formatter
    formatter = CustomFormatter(
        fmt='%(asctime)s - %(pathname)s - %(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 清除已有 handler（防止重复）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件（进程安全）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"✅ 共享日志系统初始化完成，日志文件: {log_file}")
