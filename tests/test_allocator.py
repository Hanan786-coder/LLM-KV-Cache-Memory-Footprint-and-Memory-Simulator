import random
import unittest

from kvsim.allocator import AllocationError, PagedAllocator
from kvsim.config import ConfigError
from kvsim.kvpool import KVPool


def paged(total_tokens=32, block_size=4, watermark=0.0):
    return PagedAllocator(KVPool(1, total_tokens), block_size, watermark)


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


if __name__ == "__main__":
    unittest.main()
