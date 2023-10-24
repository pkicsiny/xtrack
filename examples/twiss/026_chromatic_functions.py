from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xdeps as xd

import numpy as np


collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider['lhcb1'].twiss_default['method'] = '4d'
collider['lhcb2'].twiss_default['method'] = '4d'

# Generate MAD-X maps

mad = Madx()
mad.input(f"""
call,file="../../test_data/hllhc15_thick/lhc.seq";
call,file="../../test_data/hllhc15_thick/hllhc_sequence.madx";
beam, sequence=lhcb1, particle=proton, pc=7000;
beam, sequence=lhcb2, particle=proton, pc=7000, bv=-1;
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";
""")

line = collider.lhcb1
mad.use(sequence="lhcb1")
seq = mad.sequence.lhcb1

tw = line.twiss(only_markers=True)
twmad = mad.twiss(chrom=True)

tw_test = tw.rows['.*_exit']

wx_ref = np.interp(tw_test['s'], twmad['s'], twmad['wx'])
wy_ref = np.interp(tw_test['s'], twmad['s'], twmad['wy'])
phix_ref = np.interp(tw_test['s'], twmad['s'], twmad['phix'])
phiy_ref = np.interp(tw_test['s'], twmad['s'], twmad['phiy'])

zx_ref = wx_ref * np.exp(1j * 2 * np.pi * phix_ref)
zy_ref = wy_ref * np.exp(1j * 2 * np.pi * phiy_ref)
ax_ref = np.imag(zx_ref)
ay_ref = np.imag(zy_ref)
bx_ref = np.real(zx_ref)
by_ref = np.real(zy_ref)

assert np.allclose(tw_test.wx_chrom, wx_ref, rtol=0, atol=2e-3 * np.max(wx_ref))
assert np.allclose(tw_test.wy_chrom, wy_ref, rtol=0, atol=2e-3 * np.max(wy_ref))
assert np.allclose(tw_test.ax_chrom, ax_ref, rtol=0, atol=2e-3 * np.max(ax_ref))
assert np.allclose(tw_test.ay_chrom, ay_ref, rtol=0, atol=2e-3 * np.max(ay_ref))
assert np.allclose(tw_test.bx_chrom, bx_ref, rtol=0, atol=2e-3 * np.max(bx_ref))
assert np.allclose(tw_test.by_chrom, by_ref, rtol=0, atol=2e-3 * np.max(by_ref))
