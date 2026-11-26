"""Value tests for a lane mask read as a number (SHADER-MASK-AS-DATA, crosslane_fuzz.py "maskdata").

On hardware a lane mask is one word for the wave, a bit per lane; the translator keeps only each lane's own bit
(0 / ~0), so a 32-bit integer op that reads the mask has to rebuild the word. Wave32 and native wave64 run one whole
subgroup in spirv_mini_interp.py. Standard library only; no GPU, no LLVM, no game files.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

rdna2spirv = support.rdna2spirv
M32 = 0xFFFFFFFF
K = 0x1234


def fssm_for(wave):
    fssm = bytearray(0x148)
    fssm[:4] = b"FSSM"
    struct.pack_into("<I", fssm, 0xD4, 0x20 if wave == 32 else 0)
    struct.pack_into("<4I", fssm, 0x138, 0x8088, wave, 1, 1)
    return bytes(fssm)


TAIL = ["v_lshlrev_b32_e32 v9, 2, v0", "buffer_store_dword v4, v9, s[0:3], 0 offen", "s_endpgm"]


def translate(lines, wave):
    return support.translate_lines("cs_mask_data", lines + TAIL, fssm=fssm_for(wave))


def run(lines, wave):
    """v4 of every lane."""
    memory = si.run_workgroup(translate(lines, wave), wave, wave,
                              inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                              push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={})
    return [memory[4 * lane] for lane in range(wave)]


def mask_setup(wave, lanes):
    """(lines, [(source register, its word)]) for a mask of the lanes below `lanes`, in an SGPR and in VCC."""
    bits = (1 << lanes) - 1
    if wave == 32:
        return [f"v_cmp_gt_u32_e64 s10, {lanes}, v0", f"v_cmp_gt_u32_e32 vcc_lo, {lanes}, v0"], \
            [("s10", bits), ("vcc_lo", bits)]
    return [f"v_cmp_gt_u32_e64 s[10:11], {lanes}, v0", f"v_cmp_gt_u32_e32 vcc, {lanes}, v0"], \
        [("s10", bits & M32), ("s11", bits >> 32), ("vcc_lo", bits & M32), ("vcc_hi", bits >> 32)]


def signed(x):
    return x - (1 << 32) if x >> 31 else x


SCALAR = {
    "s_add_u32 s20, {m}, 0x1234": lambda m: m + K,
    "s_add_i32 s20, 0x1234, {m}": lambda m: m + K,
    "s_sub_u32 s20, {m}, 0x1234": lambda m: m - K,
    "s_sub_i32 s20, 0x1234, {m}": lambda m: K - m,
    "s_mul_i32 s20, {m}, 0x1234": lambda m: m * K,
    "s_mul_hi_u32 s20, {m}, 0x1234": lambda m: (m * K) >> 32,
    "s_lshl_b32 s20, {m}, 3": lambda m: m << 3,
    "s_lshr_b32 s20, {m}, 3": lambda m: m >> 3,
    "s_ashr_i32 s20, {m}, 3": lambda m: signed(m) >> 3,
    "s_bfe_u32 s20, {m}, 0x60002": lambda m: (m >> 2) & 63,
    "s_min_u32 s20, {m}, 0x1234": lambda m: min(m, K),
    "s_max_u32 s20, {m}, 0x1234": lambda m: max(m, K),
    "s_brev_b32 s20, {m}": lambda m: int(f"{m:032b}"[::-1], 2),
    "s_lshl2_add_u32 s20, {m}, 0x1234": lambda m: (m << 2) + K,
}
VECTOR = {
    "v_mov_b32_e32 v4, {m}": lambda m: m,
    "v_add_nc_u32_e32 v4, {m}, v3": lambda m: m + K,
    "v_sub_nc_u32_e32 v4, {m}, v3": lambda m: m - K,
    "v_subrev_nc_u32_e32 v4, {m}, v3": lambda m: K - m,
    "v_lshlrev_b32_e64 v4, 3, {m}": lambda m: m << 3,
    "v_lshrrev_b32_e64 v4, 3, {m}": lambda m: m >> 3,
    "v_ashrrev_i32_e64 v4, 3, {m}": lambda m: signed(m) >> 3,
    "v_mul_lo_u32 v4, {m}, v3": lambda m: m * K,
    "v_min_u32_e32 v4, {m}, v3": lambda m: min(m, K),
    "v_max_u32_e32 v4, {m}, v3": lambda m: max(m, K),
    "v_bfe_u32 v4, {m}, 2, 6": lambda m: (m >> 2) & 63,
    "v_add3_u32 v4, {m}, v3, v3": lambda m: m + 2 * K,
    "v_lshl_add_u32 v4, {m}, 2, v3": lambda m: (m << 2) + K,
    "v_bcnt_u32_b32 v4, {m}, v3": lambda m: bin(m).count("1") + K,
}


class MaskAsDataTests(unittest.TestCase):
    def check(self, lines, wave, want):
        self.assertEqual({hex(x) for x in run(lines, wave)}, {hex(want & M32)})

    def test_scalar_ops(self):
        for wave, lanes in ((32, 10), (64, 43)):
            setup, sources = mask_setup(wave, lanes)
            for (op, fn), (reg, word) in ((o, s) for o in SCALAR.items() for s in sources):
                with self.subTest(wave=wave, op=op, reg=reg):
                    self.check(setup + [op.format(m=reg), "v_mov_b32_e32 v4, s20"], wave, fn(word))

    def test_vector_ops(self):
        for wave, lanes in ((32, 10), (64, 43)):
            setup, sources = mask_setup(wave, lanes)
            for (op, fn), (reg, word) in ((o, s) for o in VECTOR.items() for s in sources):
                with self.subTest(wave=wave, op=op, reg=reg):
                    self.check(setup + [f"v_mov_b32_e32 v3, {K:#x}", op.format(m=reg)], wave, fn(word))

    def test_exec_as_data(self):
        self.check(["v_mov_b32_e32 v4, exec_lo"], 32, M32)
        self.check(["v_cmp_gt_u32_e32 vcc_lo, 10, v0", "s_and_saveexec_b32 s12, vcc_lo", "s_add_u32 s20, exec_lo, 1",
                    "s_mov_b32 exec_lo, s12", "v_mov_b32_e32 v4, s20"], 32, 0x400)

    def test_copied_mask(self):
        self.check(["v_cmp_gt_u32_e64 s10, 10, v0", "s_mov_b32 s11, s10", "s_add_u32 s20, s11, 1",
                    "v_mov_b32_e32 v4, s20"], 32, 0x400)

    def test_both_sources_are_masks(self):
        self.check(["v_cmp_gt_u32_e64 s10, 10, v0", "v_cmp_gt_u32_e64 s11, 4, v0", "s_sub_u32 s20, s10, s11",
                    "v_mov_b32_e32 v4, s20"], 32, 0x3FF - 0xF)

    def test_v_mov_under_exec_keeps_inactive_lanes_and_sees_every_lane(self):
        # lanes 5-31 are off at the v_mov: they keep v4, and the active lanes still get the whole 10-lane mask
        got = run(["v_mov_b32_e32 v4, 7", "v_cmp_gt_u32_e64 s10, 10, v0", "v_cmp_gt_u32_e32 vcc_lo, 5, v0",
                   "s_and_saveexec_b32 s12, vcc_lo", "v_mov_b32_e32 v4, s10", "s_mov_b32 exec_lo, s12"], 32)
        self.assertEqual(got, [0x3FF] * 5 + [7] * 27)

    def test_a_mask_that_goes_through_a_vgpr_stays_a_mask(self):
        # v_mov / v_or3 / v_readfirstlane and back into EXEC: the idiom keeps each lane's own bit, no rebuild
        lines = ["v_cmp_gt_u32_e64 s10, 10, v0", "v_cmp_lt_u32_e64 s11, 20, v0", "v_mov_b32_e32 v5, s10",
                 "v_or3_b32 v5, v5, s11, s11", "v_readfirstlane_b32 s14, v5", "v_mov_b32_e32 v4, 1",
                 "s_and_saveexec_b32 s12, s14", "v_mov_b32_e32 v4, 2", "s_mov_b32 exec_lo, s12"]
        self.assertEqual(run(lines, 32), [2 if i < 10 or i > 20 else 1 for i in range(32)])

    def test_a_register_reused_for_another_mask(self):
        # v5 goes through the idiom first; the second mask copied into it is only read as a number
        lines = ["v_cmp_gt_u32_e64 s10, 10, v0", "v_mov_b32_e32 v5, s10", "v_readfirstlane_b32 s14, v5",
                 "v_cmp_gt_u32_e64 s11, 4, v0", "v_mov_b32_e32 v5, s11", "v_add_nc_u32_e32 v4, 1, v5"]
        self.check(lines, 32, 0x10)

    def test_a_number_before_the_register_is_used_for_the_idiom(self):
        lines = ["v_cmp_gt_u32_e64 s11, 4, v0", "v_mov_b32_e32 v5, s11", "v_add_nc_u32_e32 v4, 1, v5",
                 "v_cmp_gt_u32_e64 s10, 10, v0", "v_mov_b32_e32 v5, s10", "v_readfirstlane_b32 s14, v5",
                 "s_and_saveexec_b32 s12, s14", "v_mov_b32_e32 v4, 2", "s_mov_b32 exec_lo, s12"]
        self.assertEqual(run(lines, 32), [2 if i < 10 else 0x10 for i in range(32)])

    def test_copies_of_the_vgpr(self):
        # a chain of VGPR copies read as a number, and one that ends in the idiom
        head = ["v_cmp_gt_u32_e64 s10, 10, v0", "v_mov_b32_e32 v5, s10", "v_mov_b32_e32 v6, v5"]
        self.check(head + ["v_mov_b32_e32 v4, v6"], 32, 0x3FF)
        self.assertEqual(run(head + ["v_mov_b32_e32 v4, 1", "v_readfirstlane_b32 s14, v6", "s_and_saveexec_b32 s12, s14",
                                     "v_mov_b32_e32 v4, 2", "s_mov_b32 exec_lo, s12"], 32),
                         [2 if i < 10 else 1 for i in range(32)])

    def test_a_copy_read_by_readfirstlane_under_exec_is_the_whole_word(self):
        # lanes 5-31 are off at the v_mov and keep v5: the first active lane's copy is the 10-lane mask, not their bits
        lines = ["v_mov_b32_e32 v5, 0x7777", "v_cmp_gt_u32_e64 s10, 10, v0", "v_cmp_gt_u32_e32 vcc_lo, 5, v0",
                 "s_and_saveexec_b32 s12, vcc_lo", "v_mov_b32_e32 v5, s10", "v_readfirstlane_b32 s14, v5",
                 "s_mov_b32 exec_lo, s12", "v_mov_b32_e32 v4, s14"]
        self.check(lines, 32, 0x3FF)

    def test_a_copy_that_a_bitwise_op_combines_with_data_is_the_whole_word(self):
        # the high half of a 43-lane mask is 0x7ff for every lane, the ones outside the mask included
        self.check(["v_cmp_gt_u32_e64 s[10:11], 43, v0", "v_mov_b32_e32 v3, s11", "v_and_b32_e32 v4, 1, v3"], 64, 1)

    def test_a_mask_computation_nothing_reads_as_a_mask_keeps_the_idle_lanes(self):
        # crosslane_fuzz 30472: the xor only feeds a store, so the lanes EXEC has off keep v4 (5-31 here)
        lines = ["v_mov_b32_e32 v4, 7", "v_cmp_gt_u32_e64 s10, 10, v0", "v_cmp_gt_u32_e32 vcc_lo, 5, v0",
                 "s_and_saveexec_b32 s12, vcc_lo", "v_mov_b32_e32 v5, s10", "v_xor_b32_e32 v4, v5, v5",
                 "s_mov_b32 exec_lo, s12"]
        self.assertEqual(run(lines, 32), [0] * 5 + [7] * 27)

    def test_a_computed_mask_read_through_a_copy_stays_a_mask_under_a_partial_exec(self):
        # the idiom with EXEC off from lane 20: the v_or3 computes in every lane, a copy and v_readfirstlane carry it on
        lines = ["v_cmp_gt_u32_e64 s10, 10, v0", "v_cmp_gt_u32_e64 s11, 4, v0", "v_mov_b32_e32 v4, 1",
                 "v_cmp_gt_u32_e32 vcc_lo, 20, v0", "s_and_saveexec_b32 s13, vcc_lo", "v_mov_b32_e32 v5, s10",
                 "v_or3_b32 v6, v5, s11, s11", "v_mov_b32_e32 v7, v6", "v_readfirstlane_b32 s14, v7",
                 "s_mov_b32 exec_lo, s13", "s_and_saveexec_b32 s12, s14", "v_mov_b32_e32 v4, 2", "s_mov_b32 exec_lo, s12"]
        self.assertEqual(run(lines, 32), [2 if i < 10 else 1 for i in range(32)])

    def test_two_moves_that_meet_at_a_mask_read(self):
        # one of two copies on two paths, then read as a mask: both keep the lane's own bit
        lines = ["v_cmp_gt_u32_e64 s10, 10, v0", "v_cmp_gt_u32_e64 s11, 4, v0", "v_mov_b32_e32 v5, s10",
                 "v_mov_b32_e32 v6, s11", "v_or_b32_e32 v5, v5, v6", "v_readfirstlane_b32 s14, v5",
                 "v_mov_b32_e32 v4, 1", "s_and_saveexec_b32 s12, s14", "v_mov_b32_e32 v4, 2", "s_mov_b32 exec_lo, s12"]
        self.assertEqual(run(lines, 32), [2 if i < 10 else 1 for i in range(32)])

    def test_plain_data_is_untouched(self):
        lines = ["s_mov_b32 s10, 0x3ff", "s_add_u32 s20, s10, 1", "v_mov_b32_e32 v4, s20"]
        self.check(lines, 32, 0x400)
        self.assertNotIn("OpGroupNonUniform", translate(lines, 32))

    def test_wave64_emulate_translates_with_an_exchange(self):
        old = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            spv = translate(["v_cmp_gt_u32_e64 s[10:11], 43, v0", "s_add_u32 s20, s11, 1", "v_mov_b32_e32 v4, s20"], 64)
        finally:
            rdna2spirv.WAVE64_EMULATE = old
        self.assertIn("%x_buf", spv)


if __name__ == "__main__":
    unittest.main()
