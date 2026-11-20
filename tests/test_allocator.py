import random
import unittest

from kvsim.allocator import (
    AllocationError, ContiguousAllocator, PagedAllocator, next_power_of_two, reservation_tokens,
)
from kvsim.config import ConfigError
from kvsim.kvpool import KVPool
from kvsim.workload import Request


def paged(total_tokens=32, block_size=4, watermark=0.0):
    return PagedAllocator(KVPool(1, total_tokens), block_size, watermark)


def contiguous(total_tokens=100, fit="first", reservation="oracle", max_model_len=2048):
    return ContiguousAllocator(KVPool(1, total_tokens), max_model_len, reservation, fit)


def sized(seq_id, size):
    """A request whose oracle reservation is exactly ``size`` slots (P = size, O = 1)."""
    return Request(seq_id, 0, size, 1)


class TestPagedAllocator(unittest.TestCase):
    def test_blocks_are_claimed_only_when_needed(self):
        a = paged()  # 8 blocks of 4 tokens
        a.admit(1, 5)  # prompt of 5 tokens -> 2 blocks
        self.assertEqual((a.blocks_used, a.held_tokens(1), a.num_tokens(1)), (2, 8, 5))
        self.assertEqual(a.free_tokens(), 24)
        a.append(1, 3)  # tokens 6-8 fill the second block
        self.assertEqual(a.blocks_used, 2)
        a.append(1)  # token 9 needs a third block
        self.assertEqual((a.blocks_used, a.num_tokens(1)), (3, 9))
        a.check_invariants()

    def test_free_returns_blocks_and_they_are_reused_first(self):
        a = paged()
        a.admit(1, 8)
        a.admit(2, 4)
        first_blocks = a.block_table(1)
        self.assertEqual(a.free(1), 8)
        self.assertNotIn(1, a)
        a.admit(3, 8)
        self.assertEqual(a.block_table(3), first_blocks)  # LIFO free list
        a.check_invariants()

    def test_full_pool(self):
        a = paged(total_tokens=8)  # 2 blocks
        a.admit(1, 8)
        self.assertFalse(a.can_append(1))
        with self.assertRaises(AllocationError):
            a.append(1)
        self.assertFalse(a.can_admit(2, 1))
        self.assertEqual(a.largest_free_run(), 0)

    def test_watermark_applies_to_admission_not_growth(self):
        a = paged(total_tokens=400, block_size=4, watermark=0.05)  # 100 blocks, keep 5 free
        self.assertEqual(a.watermark_blocks, 5)
        self.assertTrue(a.can_admit(1, 380))  # 95 blocks
        self.assertFalse(a.can_admit(1, 381))  # 96 blocks
        a.admit(1, 380)
        a.append(1, 20)  # growth may use the last 5 blocks
        self.assertEqual(a.blocks_free, 0)
        a.check_invariants()

    def test_leftover_slots_are_not_usable(self):
        a = paged(total_tokens=30)  # 7 whole blocks, 2 slots left over
        self.assertEqual((a.num_blocks, a.capacity_tokens), (7, 28))

    def test_block_table_metadata(self):
        a = paged()
        a.admit(1, 9)
        self.assertEqual(a.block_table_bytes, 3 * 4)

    def test_misuse_is_an_error(self):
        a = paged()
        a.admit(1, 4)
        with self.assertRaises(AllocationError):
            a.admit(1, 4)
        for call in (lambda: a.free(2), lambda: a.append(2), lambda: a.num_tokens(2),
                     lambda: a.can_append(2), lambda: a.admit(3, 0), lambda: a.append(1, 0)):
            with self.assertRaises(AllocationError):
                call()

    def test_bad_configuration(self):
        with self.assertRaises(ConfigError):
            paged(total_tokens=3, block_size=4)  # not even one block
        with self.assertRaises(ConfigError):
            paged(watermark=1.0)
        self.assertEqual(paged(block_size=16, total_tokens=64).name, "paged-16")

    def test_invariants_hold_under_random_use(self):
        rng = random.Random(0)
        a = paged(total_tokens=256, block_size=8)
        live, next_id = [], 0
        for _ in range(3000):
            action = rng.random()
            if action < 0.3:
                n = rng.randint(1, 40)
                if a.can_admit(next_id, n):
                    a.admit(next_id, n)
                    live.append(next_id)
                next_id += 1
            elif action < 0.85 and live:
                seq = rng.choice(live)
                n = rng.randint(1, 10)
                if a.can_append(seq, n):
                    a.append(seq, n)
            elif live:
                a.free(live.pop(rng.randrange(len(live))))
            a.check_invariants()
            slack = sum(a.held_tokens(s) - a.num_tokens(s) for s in live)
            self.assertLess(slack, 8 * max(len(live), 1))  # under one block per request


class TestReservations(unittest.TestCase):
    def test_the_four_strategies(self):
        r = Request(0, 0, prompt_len=100, output_len=25, max_new_tokens=500)
        sizes = {s: reservation_tokens(r, s, 2048) for s in ("max", "request", "pow2", "oracle")}
        self.assertEqual(sizes, {"max": 2048, "request": 600, "pow2": 132, "oracle": 124})

    def test_limits(self):
        self.assertEqual(reservation_tokens(Request(0, 0, 100, 25), "request", 2048), 2048)
        self.assertEqual(reservation_tokens(Request(0, 0, 2000, 40), "pow2", 2048), 2048)  # capped
        self.assertEqual([next_power_of_two(n) for n in (1, 2, 3, 25, 32, 33)], [1, 2, 4, 32, 32, 64])
        with self.assertRaises(ConfigError):
            reservation_tokens(Request(0, 0, 1, 1), "guess", 2048)


class TestContiguousAllocator(unittest.TestCase):
    def test_external_fragmentation(self):
        a = contiguous()  # 100 slots
        for i in range(3):
            a.admit(i, 30, sized(i, 30))  # A, B, C at 0, 30, 60
        a.free(1)  # B leaves a 30-slot hole between A and C
        self.assertEqual(a.holes(), [(30, 30), (90, 10)])
        self.assertEqual((a.free_tokens(), a.largest_free_run()), (40, 30))
        self.assertFalse(a.can_admit(3, 35, sized(3, 35)))  # 40 free, but no 35-slot hole
        a.free(0)  # A's region merges with B's old hole
        self.assertEqual(a.holes(), [(0, 60), (90, 10)])
        a.admit(3, 35, sized(3, 35))
        self.assertEqual(a.region(3), (0, 35))
        a.check_invariants()

    def test_first_fit_and_best_fit_choose_different_holes(self):
        chosen = {}
        for fit in ("first", "best"):
            a = contiguous(fit=fit)
            for i in range(3):
                a.admit(i, 30, sized(i, 30))
            a.free(1)  # holes: 30 slots at 30, 10 slots at 90
            a.admit(9, 10, sized(9, 10))
            chosen[fit] = a.region(9)[0]
        self.assertEqual(chosen, {"first": 30, "best": 90})

    def test_free_merges_on_both_sides(self):
        a = contiguous(total_tokens=90)
        for i in range(3):
            a.admit(i, 30, sized(i, 30))
        a.free(0)
        a.free(2)
        self.assertEqual(a.free(1), 30)
        self.assertEqual(a.holes(), [(0, 90)])
        a.check_invariants()

    def test_region_is_reserved_up_front_and_never_grows(self):
        a = contiguous(reservation="max", max_model_len=50)
        a.admit(0, 10, Request(0, 0, 10, 5))  # max reserves 50 slots for 14 tokens
        self.assertEqual((a.held_tokens(0), a.num_tokens(0)), (50, 10))
        a.append(0, 40)
        self.assertFalse(a.can_append(0))
        with self.assertRaises(AllocationError):
            a.append(0)
        self.assertEqual(a.held_tokens(0), 50)
        self.assertEqual(a.name, "contig-max")
        self.assertEqual(contiguous(fit="best").name, "contig-oracle-best")

    def test_misuse_is_an_error(self):
        a = contiguous()
        with self.assertRaises(AllocationError):
            a.admit(0, 10)  # no request to size the reservation
        with self.assertRaises(AllocationError):
            a.admit(0, 11, sized(0, 10))  # more tokens than the reservation
        a.admit(0, 10, sized(0, 10))
        with self.assertRaises(AllocationError):
            a.admit(0, 10, sized(0, 10))
        for call in (lambda: a.free(5), lambda: a.can_append(5), lambda: a.held_tokens(5)):
            with self.assertRaises(AllocationError):
                call()
        for bad in ({"fit": "worst"}, {"reservation": "all"}, {"max_model_len": 1}):
            with self.assertRaises(ConfigError):
                contiguous(**bad)

    def test_invariants_hold_under_random_use(self):
        for fit in ("first", "best"):
            rng = random.Random(1)
            a = contiguous(total_tokens=500, fit=fit)
            live = []
            for step in range(3000):
                if rng.random() < 0.55:
                    size = rng.randint(1, 80)
                    if a.can_admit(step, size, sized(step, size)):
                        a.admit(step, size, sized(step, size))
                        live.append(step)
                elif live:
                    a.free(live.pop(rng.randrange(len(live))))
                a.check_invariants()
                self.assertEqual(a.allocated_tokens() + a.free_tokens(), 500)


if __name__ == "__main__":
    unittest.main()
