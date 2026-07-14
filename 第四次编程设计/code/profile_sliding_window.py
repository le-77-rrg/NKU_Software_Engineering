#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: 林盛森

"""滑动窗口最大值算法的性能分析脚本。"""

from __future__ import annotations

import cProfile
import io
import pstats
import random
import sys
import time
from collections.abc import Callable

from sliding_window_max import (
    sliding_window_max_array_queue,
    sliding_window_max_brute_force,
    sliding_window_max_monotonic_queue,
)


RANDOM_SEED = 20260518
DATA_SIZE = 12_000
WINDOW_SIZE = 120


def build_profile_data(size: int = DATA_SIZE) -> list[int]:
    """构造可复现的整数测试数据。"""
    random.seed(RANDOM_SEED)
    return [random.randint(-10_000, 10_000) for _ in range(size)]


def measure_algorithm(
    algorithm: Callable[[list[int], int], list[int]],
    nums: list[int],
    k: int,
) -> tuple[list[int], float]:
    """运行一次算法，并返回计算结果和耗时。"""
    start_time = time.perf_counter()
    result = algorithm(nums, k)
    elapsed_time = time.perf_counter() - start_time
    return result, elapsed_time


def profile_algorithm(
    algorithm: Callable[[list[int], int], list[int]],
    nums: list[int],
    k: int,
) -> str:
    """返回按累计耗时排序的 cProfile 文本结果。"""
    profiler = cProfile.Profile()
    profiler.enable()
    algorithm(nums, k)
    profiler.disable()

    output_buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=output_buffer)
    stats.strip_dirs().sort_stats("cumtime").print_stats(8)
    return output_buffer.getvalue()


def run_profile() -> None:
    """执行三种算法的计时和 cProfile 分析。"""
    nums = build_profile_data()

    brute_force_result, brute_force_time = measure_algorithm(
        sliding_window_max_brute_force, nums, WINDOW_SIZE
    )
    monotonic_queue_result, monotonic_queue_time = measure_algorithm(
        sliding_window_max_monotonic_queue, nums, WINDOW_SIZE
    )
    array_queue_result, array_queue_time = measure_algorithm(
        sliding_window_max_array_queue, nums, WINDOW_SIZE
    )

    if brute_force_result != monotonic_queue_result:
        raise RuntimeError("暴力版与单调队列版输出不一致。")

    if brute_force_result != array_queue_result:
        raise RuntimeError("暴力版与数组模拟队列版输出不一致。")

    # 只有三种算法结果一致时，性能比较才具有实际意义。
    monotonic_speedup = brute_force_time / monotonic_queue_time
    array_speedup = brute_force_time / array_queue_time
    array_speed_ratio = monotonic_queue_time / array_queue_time
    print(f"data_size: {DATA_SIZE}")
    print(f"window_size: {WINDOW_SIZE}")
    print(f"result_length: {len(array_queue_result)}")
    print(f"brute_force_time_seconds: {brute_force_time:.6f}")
    print(f"monotonic_queue_time_seconds: {monotonic_queue_time:.6f}")
    print(f"array_queue_time_seconds: {array_queue_time:.6f}")
    print(f"monotonic_speedup_vs_brute_force: {monotonic_speedup:.2f}x")
    print(f"array_speedup_vs_brute_force: {array_speedup:.2f}x")
    print(f"array_speed_ratio_vs_monotonic_queue: {array_speed_ratio:.2f}x")
    print()

    print("Brute-force cProfile top calls:")
    print(
        profile_algorithm(
            sliding_window_max_brute_force, nums, WINDOW_SIZE
        ).strip()
    )
    print()
    print("Monotonic-queue cProfile top calls:")
    print(
        profile_algorithm(
            sliding_window_max_monotonic_queue, nums, WINDOW_SIZE
        ).strip()
    )
    print()
    print("Array-queue cProfile top calls:")
    print(
        profile_algorithm(
            sliding_window_max_array_queue, nums, WINDOW_SIZE
        ).strip()
    )


def main(argv: list[str] | None = None) -> int:
    """执行命令行性能分析入口。

    参数:
        argv: 命令行参数列表。本性能脚本不需要额外命令行参数。

    返回:
        程序退出状态码，0 表示正常结束。
    """
    _ = argv
    run_profile()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
