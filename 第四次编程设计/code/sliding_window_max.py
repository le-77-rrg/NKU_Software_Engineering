#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: 林盛森

"""滑动窗口最大值算法实现。

本模块提供暴力遍历版、单调队列版和数组模拟队列版。公开算法函数
共用同一个输入校验函数，便于后续新增算法时复用边界检查逻辑。
"""

import sys
from collections import deque


DEMO_NUMBERS = [1, 3, -1, -3, 5, 3, 6, 7]
DEMO_WINDOW_SIZE = 3


def validate_window_input(nums: list[int], k: int) -> None:
    """校验滑动窗口输入数据。

    参数:
        nums: 用于滑动窗口计算的整数列表。
        k: 正整数窗口大小。

    异常:
        TypeError: 当 nums 不是列表、k 不是整数，或 nums 中含非整数元素时抛出。
        ValueError: 当 nums 为空、k 非正数，或 k 大于数组长度时抛出。
    """
    if not isinstance(nums, list):
        raise TypeError("nums 必须是整数列表。")

    if not nums:
        raise ValueError("nums 不能为空。")

    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k 必须是整数。")

    if k <= 0:
        raise ValueError("k 必须大于 0。")

    if k > len(nums):
        raise ValueError("k 不能大于 nums 的长度。")

    for index, value in enumerate(nums):
        # bool 是 int 的子类，需要单独排除，避免 True/False 被当作整数数据。
        if not isinstance(value, int) or isinstance(value, bool):
            value_type = type(value).__name__
            raise TypeError(
                f"nums[{index}] 必须是整数，实际类型为 {value_type}。"
            )


def sliding_window_max_brute_force(nums: list[int], k: int) -> list[int]:
    """使用暴力遍历法计算每个滑动窗口的最大值。

    该函数逐个扫描窗口并计算窗口内最大值，适合作为正确性基准。
    由于相邻窗口之间有大量重叠元素，暴力遍历会重复进行比较。

    参数:
        nums: 整数列表。
        k: 正整数窗口大小。

    返回:
        每个滑动窗口最大值组成的列表。
    """
    validate_window_input(nums, k)

    max_values = []
    window_count = len(nums) - k + 1
    for left_index in range(window_count):
        right_index = left_index + k
        max_values.append(max(nums[left_index:right_index]))

    return max_values


def sliding_window_max_monotonic_queue(nums: list[int], k: int) -> list[int]:
    """使用单调队列计算每个滑动窗口的最大值。

    队列保存数组下标而不是直接保存数值。队列中下标对应的数值保持
    非递增顺序，因此队首下标总是指向当前窗口最大值。

    参数:
        nums: 整数列表。
        k: 正整数窗口大小。

    返回:
        每个滑动窗口最大值组成的列表。
    """
    validate_window_input(nums, k)

    max_values = []
    candidate_indexes: deque[int] = deque()

    for current_index, current_value in enumerate(nums):
        # 队首下标若已离开窗口，就不能再参与当前窗口最大值判断。
        while candidate_indexes and candidate_indexes[0] <= current_index - k:
            candidate_indexes.popleft()

        # 新元素更大时，队尾较小候选不会再成为后续窗口最大值。
        while (
            candidate_indexes
            and nums[candidate_indexes[-1]] <= current_value
        ):
            candidate_indexes.pop()

        candidate_indexes.append(current_index)

        if current_index >= k - 1:
            max_values.append(nums[candidate_indexes[0]])

    return max_values


def sliding_window_max_array_queue(nums: list[int], k: int) -> list[int]:
    """使用数组模拟队列计算每个滑动窗口的最大值。

    该函数与单调队列版保持相同算法思想，但使用列表和首尾指针模拟队列，
    减少 deque 方法调用带来的常数开销，便于进行同层级性能对比。

    参数:
        nums: 整数列表。
        k: 正整数窗口大小。

    返回:
        每个滑动窗口最大值组成的列表。
    """
    validate_window_input(nums, k)

    max_values = []
    queue_indexes = [0] * len(nums)
    head_index = 0
    tail_index = 0

    for current_index, current_value in enumerate(nums):
        if (
            head_index < tail_index
            and queue_indexes[head_index] <= current_index - k
        ):
            head_index += 1

        while (
            head_index < tail_index
            and nums[queue_indexes[tail_index - 1]] <= current_value
        ):
            tail_index -= 1

        queue_indexes[tail_index] = current_index
        tail_index += 1

        if current_index >= k - 1:
            max_values.append(nums[queue_indexes[head_index]])

    return max_values


def run_demo() -> None:
    """运行题目给定样例并输出三种算法的结果。"""
    brute_force_result = sliding_window_max_brute_force(
        DEMO_NUMBERS, DEMO_WINDOW_SIZE
    )
    monotonic_queue_result = sliding_window_max_monotonic_queue(
        DEMO_NUMBERS, DEMO_WINDOW_SIZE
    )
    array_queue_result = sliding_window_max_array_queue(
        DEMO_NUMBERS, DEMO_WINDOW_SIZE
    )

    print(f"nums = {DEMO_NUMBERS}")
    print(f"k = {DEMO_WINDOW_SIZE}")
    print(f"brute_force = {brute_force_result}")
    print(f"monotonic_queue = {monotonic_queue_result}")
    print(f"array_queue = {array_queue_result}")


def main(argv: list[str] | None = None) -> int:
    """执行命令行示例入口。

    参数:
        argv: 命令行参数列表。本实验示例不需要额外命令行参数。

    返回:
        程序退出状态码，0 表示正常结束。
    """
    _ = argv
    run_demo()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
