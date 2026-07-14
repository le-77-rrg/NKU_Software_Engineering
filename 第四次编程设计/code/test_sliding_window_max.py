#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: 林盛森

"""滑动窗口最大值算法的单元测试。"""

import unittest
from unittest.mock import call, patch

from sliding_window_max import (
    main as run_sample_main,
    run_demo,
    sliding_window_max_array_queue,
    sliding_window_max_brute_force,
    sliding_window_max_monotonic_queue,
    validate_window_input,
)


class SlidingWindowTestCase(unittest.TestCase):
    """提供三种算法共用的断言工具。"""

    def setUp(self) -> None:
        """准备每个测试用例需要调用的算法函数。"""
        self.algorithms = (
            ("brute_force", sliding_window_max_brute_force),
            ("monotonic_queue", sliding_window_max_monotonic_queue),
            ("array_queue", sliding_window_max_array_queue),
        )

    def tearDown(self) -> None:
        """清理每个测试用例使用的算法函数集合。"""
        self.algorithms = ()

    def assert_algorithm_results(
        self, nums: list[int], k: int, expected: list[int]
    ) -> None:
        """断言三种算法都能得到预期结果。"""
        for algorithm_name, algorithm in self.algorithms:
            with self.subTest(algorithm=algorithm_name):
                self.assertEqual(algorithm(nums, k), expected)

    def assert_result_length(self, nums: list[int], k: int) -> None:
        """断言三种算法输出长度均满足 n-k+1。"""
        expected_length = len(nums) - k + 1
        for algorithm_name, algorithm in self.algorithms:
            with self.subTest(algorithm=algorithm_name):
                self.assertEqual(len(algorithm(nums, k)), expected_length)


class TestBlackBoxEquivalenceClass(SlidingWindowTestCase):
    """黑盒测试：等价类划分。"""

    def test_sample_case(self) -> None:
        """题目样例代表普通有效等价类。"""
        nums = [1, 3, -1, -3, 5, 3, 6, 7]
        self.assert_algorithm_results(nums, 3, [3, 3, 5, 5, 6, 7])

    def test_negative_numbers(self) -> None:
        """全负数数组代表有效整数数组等价类。"""
        self.assert_algorithm_results(
            [-8, -3, -5, -2, -9], 2, [-3, -3, -2, -2]
        )

    def test_zero_values(self) -> None:
        """包含零值的数组仍属于有效整数数组等价类。"""
        self.assert_algorithm_results([0, 0, -1, 0], 2, [0, 0, 0])

    def test_all_equal_values(self) -> None:
        """全部相等的整数数组仍属于有效等价类。"""
        self.assert_algorithm_results([7, 7, 7, 7], 2, [7, 7, 7])

    def test_validation_accepts_minimum_valid_input(self) -> None:
        """最小合法输入应通过输入校验。"""
        self.assertIsNone(validate_window_input([1], 1))

    def test_validation_accepts_nominal_input(self) -> None:
        """普通合法输入应通过输入校验。"""
        self.assertIsNone(validate_window_input([1, -2, 3], 2))

    def test_nums_must_be_list_tuple(self) -> None:
        """nums 为元组时属于无效等价类。"""
        with self.assertRaises(TypeError):
            validate_window_input((1, 2, 3), 2)  # type: ignore[arg-type]

    def test_nums_must_be_list_none(self) -> None:
        """nums 为 None 时属于无效等价类。"""
        with self.assertRaises(TypeError):
            validate_window_input(None, 2)  # type: ignore[arg-type]

    def test_nums_items_must_be_integer_string(self) -> None:
        """字符串元素属于无效等价类。"""
        with self.assertRaises(TypeError):
            validate_window_input([1, "2", 3], 2)  # type: ignore[list-item]

    def test_k_must_be_integer_float(self) -> None:
        """浮点数 k 属于无效等价类。"""
        with self.assertRaises(TypeError):
            validate_window_input([1, 2, 3], 2.5)  # type: ignore[arg-type]

    def test_k_must_be_integer_string(self) -> None:
        """字符串 k 属于无效等价类。"""
        with self.assertRaises(TypeError):
            validate_window_input([1, 2, 3], "2")  # type: ignore[arg-type]

    def test_nums_cannot_be_empty_invalid_equivalence(self) -> None:
        """空列表属于无效等价类。"""
        with self.assertRaises(ValueError):
            validate_window_input([], 1)

    def test_bool_item_invalid_equivalence(self) -> None:
        """布尔值元素属于无效等价类。"""
        with self.assertRaises(TypeError):
            validate_window_input([1, True, 3], 2)

    def test_bool_k_invalid_equivalence(self) -> None:
        """布尔值 k 属于无效等价类。"""
        with self.assertRaises(TypeError):
            validate_window_input([1, 2, 3], True)

    def test_non_positive_k_invalid_equivalence(self) -> None:
        """非正整数 k 属于无效等价类。"""
        with self.assertRaises(ValueError):
            validate_window_input([1, 2, 3], 0)

    def test_oversized_k_invalid_equivalence(self) -> None:
        """大于数组长度的 k 属于无效等价类。"""
        with self.assertRaises(ValueError):
            validate_window_input([1, 2, 3], 4)


class TestBlackBoxBoundaryValueAnalysis(SlidingWindowTestCase):
    """黑盒测试：边界值分析。"""

    def test_single_element_array(self) -> None:
        """数组长度和窗口长度同时取最小合法值。"""
        self.assert_algorithm_results([42], 1, [42])

    def test_window_size_is_one(self) -> None:
        """窗口长度取下边界 min。"""
        nums = [2, -1, 5, 0]
        self.assert_algorithm_results(nums, 1, nums)

    def test_window_size_is_two(self) -> None:
        """窗口长度取 min+。"""
        self.assert_algorithm_results([4, 1, 3, 2], 2, [4, 3, 3])

    def test_window_size_is_nominal_value(self) -> None:
        """窗口长度取普通中间值 nom。"""
        self.assert_algorithm_results([6, 1, 5, 2, 4, 3], 3, [6, 5, 5, 4])

    def test_window_size_is_max_minus(self) -> None:
        """窗口长度取 max-。"""
        self.assert_algorithm_results([1, 9, 2, 8, 3], 4, [9, 9])

    def test_window_size_equals_array_length(self) -> None:
        """窗口长度取上边界 max。"""
        self.assert_algorithm_results([4, 1, 8, 2], 4, [8])

    def test_maximum_at_window_left_boundary(self) -> None:
        """输出值来自窗口左边界时应保持正确。"""
        self.assert_algorithm_results([9, 1, 1, 1], 2, [9, 1, 1])

    def test_maximum_at_window_right_boundary(self) -> None:
        """输出值来自窗口右边界时应及时更新。"""
        self.assert_algorithm_results([1, 1, 1, 9], 2, [1, 1, 9])

    def test_result_length_follows_formula(self) -> None:
        """输出数组长度边界应满足 len(nums)-k+1。"""
        self.assert_result_length([3, 1, 4, 1, 5, 9], 4)


class TestBlackBoxRobustnessTesting(SlidingWindowTestCase):
    """黑盒测试：健壮性测试。"""

    def test_nums_cannot_be_empty(self) -> None:
        """空数组是数组长度的越界输入。"""
        with self.assertRaises(ValueError):
            validate_window_input([], 1)

    def test_k_must_be_positive_zero(self) -> None:
        """k=0 对应窗口大小 min-。"""
        with self.assertRaises(ValueError):
            validate_window_input([1, 2, 3], 0)

    def test_k_must_be_positive_negative(self) -> None:
        """负数 k 是小于最小边界的非法输入。"""
        with self.assertRaises(ValueError):
            validate_window_input([1, 2, 3], -1)

    def test_k_cannot_exceed_array_length(self) -> None:
        """k=len(nums)+1 对应窗口大小 max+。"""
        with self.assertRaises(ValueError):
            validate_window_input([1, 2, 3], 4)

    def test_k_must_not_be_bool(self) -> None:
        """布尔值 k 是类型边界上的特殊非法输入。"""
        with self.assertRaises(TypeError):
            validate_window_input([1, 2, 3], True)

    def test_nums_items_must_be_integer_float(self) -> None:
        """浮点数元素是元素类型边界上的非法输入。"""
        with self.assertRaises(TypeError):
            validate_window_input([1, 2.0, 3], 2)  # type: ignore[list-item]

    def test_nums_items_must_be_integer_none(self) -> None:
        """None 元素是元素类型边界上的非法输入。"""
        with self.assertRaises(TypeError):
            validate_window_input([1, None, 3], 2)  # type: ignore[list-item]

    def test_nested_list_item_is_rejected(self) -> None:
        """嵌套列表元素是元素类型边界上的非法输入。"""
        with self.assertRaises(TypeError):
            validate_window_input([1, [2], 3], 2)  # type: ignore[list-item]


class TestBlackBoxErrorGuessing(SlidingWindowTestCase):
    """黑盒测试：错误推测法。"""

    def test_duplicate_values(self) -> None:
        """重复最大值容易导致候选下标维护错误。"""
        self.assert_algorithm_results([5, 5, 5, 2, 5], 3, [5, 5, 5])

    def test_strictly_increasing_numbers(self) -> None:
        """递增数组容易暴露队尾弹出错误。"""
        self.assert_algorithm_results([1, 2, 3, 4, 5], 3, [3, 4, 5])

    def test_strictly_decreasing_numbers(self) -> None:
        """递减数组容易暴露队首过期处理错误。"""
        self.assert_algorithm_results([5, 4, 3, 2, 1], 3, [5, 4, 3])

    def test_bool_is_not_accepted_as_integer_item(self) -> None:
        """布尔值元素容易被 Python 误认为整数。"""
        with self.assertRaises(TypeError):
            validate_window_input([1, True, 3], 2)


class TestSupplementaryAlgorithmBehavior(SlidingWindowTestCase):
    """补充验证：典型输入组合和窗口移动行为。"""

    def test_negative_values_with_small_window(self) -> None:
        """负数数组与较小窗口应得到正确最大值。"""
        self.assert_algorithm_results([-4, -1, -7, -2], 2, [-1, -1, -2])

    def test_duplicate_values_with_large_window(self) -> None:
        """重复元素数组与较大窗口应得到正确最大值。"""
        self.assert_algorithm_results([3, 3, 1, 3, 2], 4, [3, 3])

    def test_valid_input_enters_normal_calculation(self) -> None:
        """所有输入条件合法时应进入正常计算。"""
        self.assert_algorithm_results([8, 3, 5], 2, [8, 5])

    def test_oversized_window_raises_value_error(self) -> None:
        """窗口越界时应产生 ValueError。"""
        with self.assertRaises(ValueError):
            validate_window_input([1, 2, 3], 5)

    def test_window_moves_across_array(self) -> None:
        """窗口连续右移时最大值应保持正确。"""
        self.assert_algorithm_results([1, 4, 2, 6], 2, [4, 4, 6])


class TestSupplementaryMetamorphicRelation(SlidingWindowTestCase):
    """补充验证：变形关系和交叉检查。"""

    def test_translation_relation_with_added_constant(self) -> None:
        """所有输入同时加常数时，输出也应加同一常数。"""
        self.assert_algorithm_results([11, 13, 11, 7, 15], 3, [13, 13, 15])

    def test_positive_scale_relation(self) -> None:
        """所有输入乘以正数时，输出最大值应按比例放大。"""
        self.assert_algorithm_results([6, -3, 12, 0], 2, [6, 12, 12])


class TestWhiteBoxCoverageCriteria(SlidingWindowTestCase):
    """白盒测试：六种代码覆盖标准。"""

    def test_monotonic_queue_expires_front_candidate(self) -> None:
        """覆盖单调队列队首过期分支。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([9, 1, 2, 3], 2),
            [9, 2, 3],
        )

    def test_monotonic_queue_pops_smaller_tail(self) -> None:
        """覆盖单调队列队尾弹出分支。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([1, 2, 3, 4], 2),
            [2, 3, 4],
        )

    def test_monotonic_queue_keeps_larger_tail(self) -> None:
        """覆盖单调队列队尾不弹出的分支。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([4, 3, 2, 1], 2),
            [4, 3, 2],
        )

    def test_monotonic_queue_handles_equal_values(self) -> None:
        """覆盖单调队列相等元素条件组合。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([2, 2, 2, 1], 2),
            [2, 2, 2],
        )


class TestWhiteBoxBasicPathAndLoopTesting(SlidingWindowTestCase):
    """白盒测试：基本路径测试和循环测试。"""

    def test_basic_path_window_forms_immediately(self) -> None:
        """基本路径：窗口在第一次循环时立即形成。"""
        self.assertEqual(sliding_window_max_monotonic_queue([6], 1), [6])

    def test_basic_path_window_waits_before_output(self) -> None:
        """基本路径：窗口尚未形成时不追加结果。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([6, 1], 2),
            [6],
        )

    def test_basic_path_expires_front_candidate(self) -> None:
        """基本路径：队首候选下标过期并被移除。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([9, 1, 2, 3], 2),
            [9, 2, 3],
        )

    def test_basic_path_pops_tail_candidate(self) -> None:
        """基本路径：队尾候选值较小并被弹出。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([1, 2, 3, 4], 2),
            [2, 3, 4],
        )

    def test_basic_path_keeps_tail_candidate(self) -> None:
        """基本路径：队尾候选值较大并被保留。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([4, 3, 2, 1], 2),
            [4, 3, 2],
        )

    def test_algorithm_loop_executes_once(self) -> None:
        """单调队列主循环执行一次。"""
        self.assertEqual(sliding_window_max_monotonic_queue([6], 1), [6])

    def test_algorithm_loop_executes_twice(self) -> None:
        """单调队列主循环执行两次。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([6, 1], 1),
            [6, 1],
        )

    def test_algorithm_loop_executes_many_times(self) -> None:
        """单调队列主循环执行多次。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([4, 1, 7, 0, 3], 2),
            [4, 7, 7, 3],
        )

    def test_nested_loop_repeated_tail_pops(self) -> None:
        """嵌套循环中队尾弹出循环可连续执行。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([1, 3, 2, 4], 3),
            [3, 4],
        )

    def test_concatenated_loops_expire_then_pop(self) -> None:
        """串接循环中先移除过期队首再弹出较小队尾。"""
        self.assertEqual(
            sliding_window_max_monotonic_queue([9, 1, 10], 2),
            [9, 10],
        )


class TestUnittestAndMockIsolation(SlidingWindowTestCase):
    """unittest 框架和 Mock 隔离测试。"""

    def test_run_demo_prints_required_sample_result(self) -> None:
        """Mock 捕获示例入口打印的样例结果。"""
        with patch("builtins.print") as mock_print:
            run_demo()

        printed_lines = [call_item.args[0] for call_item in mock_print.call_args_list]
        self.assertIn("brute_force = [3, 3, 5, 5, 6, 7]", printed_lines)
        self.assertIn("monotonic_queue = [3, 3, 5, 5, 6, 7]", printed_lines)
        self.assertIn("array_queue = [3, 3, 5, 5, 6, 7]", printed_lines)

    def test_run_demo_print_order_with_mock(self) -> None:
        """Mock 断言示例入口打印调用顺序。"""
        with patch("builtins.print") as mock_print:
            run_demo()

        mock_print.assert_has_calls(
            [
                call("nums = [1, 3, -1, -3, 5, 3, 6, 7]"),
                call("k = 3"),
                call("brute_force = [3, 3, 5, 5, 6, 7]"),
                call("monotonic_queue = [3, 3, 5, 5, 6, 7]"),
                call("array_queue = [3, 3, 5, 5, 6, 7]"),
            ]
        )

    def test_run_demo_uses_mocked_algorithm_results(self) -> None:
        """Mock 控制算法返回值并验证示例入口调用参数。"""
        with (
            patch(
                "sliding_window_max.sliding_window_max_brute_force",
                return_value=[10],
            ) as mock_brute_force,
            patch(
                "sliding_window_max.sliding_window_max_monotonic_queue",
                return_value=[20],
            ) as mock_monotonic_queue,
            patch(
                "sliding_window_max.sliding_window_max_array_queue",
                return_value=[30],
            ) as mock_array_queue,
            patch("builtins.print") as mock_print,
        ):
            run_demo()

        expected_nums = [1, 3, -1, -3, 5, 3, 6, 7]
        mock_brute_force.assert_called_once_with(expected_nums, 3)
        mock_monotonic_queue.assert_called_once_with(expected_nums, 3)
        mock_array_queue.assert_called_once_with(expected_nums, 3)
        mock_print.assert_has_calls(
            [
                call("brute_force = [10]"),
                call("monotonic_queue = [20]"),
                call("array_queue = [30]"),
            ]
        )

    def test_main_returns_success_status(self) -> None:
        """主入口应返回正常结束状态码。"""
        with patch("builtins.print"):
            self.assertEqual(run_sample_main([]), 0)

    def test_main_accepts_unused_arguments(self) -> None:
        """主入口接收额外参数时仍应返回正常状态码。"""
        with patch("builtins.print"):
            self.assertEqual(run_sample_main(["unused"]), 0)


def main() -> None:
    """运行本文件中的单元测试。"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
