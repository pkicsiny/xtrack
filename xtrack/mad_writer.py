import numpy as np
import xtrack as xt
import xdeps as xd

def expr_to_mad_str(expr):

    expr_str = str(expr)

    fff = xt.line.Functions()
    for nn in fff._mathfunctions:
        expr_str = expr_str.replace(f'f.{nn}(', f'{nn}(')

    expr_str = expr_str.replace("'", "")
    expr_str = expr_str.replace('"', "")

    # transform vars[...] in (...)
    while "vars[" in expr_str:
        before, after = tuple(*[expr_str.split("vars[", 1)])
        # find the corresponding closing bracket
        count = 1
        for ii, cc in enumerate(after):
            if cc == "]":
                count -= 1
            elif cc == "[":
                count += 1
            if count == 0:
                break

        expr_str = before + "(" + after[:ii] + ")" + after[ii+1:]

        expr_str = expr_str.replace("**", "^")

    return expr_str

def mad_str_or_value(var):
    vv = _ge(var)
    if _is_ref(vv):
        out = expr_to_mad_str(vv)
        out = out.strip('._expr')
        return out
    else:
        return vv

def mad_assignment(lhs, rhs):
    if _is_ref(rhs):
        rhs = mad_str_or_value(rhs)
    if isinstance(rhs, str):
        return f"{lhs} := {rhs}"
    else:
        return f"{lhs} = {rhs}"


_ge = xt.elements._get_expr
_is_ref = xd.refs.is_ref



def _get_eref(line, name):
    return line.element_refs[name]

def cavity_to_madx_str(name, line):
    cav = _get_eref(line, name)
    tokens = []
    tokens.append('rfcavity')
    tokens.append(mad_assignment('freq', _ge(cav.frequency) * 1e-6))
    tokens.append(mad_assignment('volt', _ge(cav.voltage) * 1e-6))
    tokens.append(mad_assignment('lag', _ge(cav.lag) / 360.))

    return ', '.join(tokens)

def marker_to_madx_str(name, line):
    return 'marker'

def drift_to_madx_str(name, line):
    drift = _get_eref(line, name)
    tokens = []
    tokens.append('drift')
    tokens.append(mad_assignment('l', _ge(drift.length)))
    return ', '.join(tokens)

def multipole_to_madx_str(name, line):
    mult = _get_eref(line, name)

    if (len(mult.knl._value) == 1 and len(mult.ksl._value) == 1
        and mult.hxl._value == 0 and mult.hyl._value == 0):
        # It is a dipole corrector
        tokens = []
        tokens.append('kicker')
        tokens.append(mad_assignment('hkick', -1 * _ge(mult.knl[0])))
        tokens.append(mad_assignment('vkick', _ge(mult.ksl[0])))
        tokens.append(mad_assignment('lrad', _ge(mult.length)))
        return ', '.join(tokens)

    # correctors are not handled correctly!!!!
    # https://github.com/MethodicalAcceleratorDesign/MAD-X/issues/911
    assert mult.hyl._value == 0

    tokens = []
    tokens.append('multipole')
    knl_mad = []
    ksl_mad = []
    for kl, klmad in zip([mult.knl, mult.ksl], [knl_mad, ksl_mad]):
        for ii in range(len(kl._value)):
            item = mad_str_or_value(_ge(kl[ii]))
            if not isinstance(item, str):
                item = str(item)
            klmad.append(item)
    tokens.append('knl:={' + ','.join(knl_mad) + '}')
    tokens.append('ksl:={' + ','.join(ksl_mad) + '}')
    tokens.append(mad_assignment('lrad', _ge(mult.length)))
    tokens.append(mad_assignment('angle', _ge(mult.hxl)))

    return ', '.join(tokens)

def rfmultipole_to_madx_str(name, line):
    rfmult = _get_eref(line, name)

    tokens = []
    tokens.append('rfmultipole')
    knl_mad = []
    ksl_mad = []
    for kl, klmad in zip([rfmult.knl, rfmult.ksl], [knl_mad, ksl_mad]):
        for ii in range(len(kl._value)):
            item = mad_str_or_value(_ge(kl[ii]))
            if not isinstance(item, str):
                item = str(item)
            klmad.append(item)
    pnl_mad = []
    psl_mad = []
    for pp, plmad in zip([rfmult.pn, rfmult.ps], [pnl_mad, psl_mad]):
        for ii in range(len(pp._value)):
            item = mad_str_or_value(_ge(pp[ii]) / 360)
            if not isinstance(item, str):
                item = str(item)
            plmad.append(item)

    tokens.append('knl:={' + ','.join(knl_mad) + '}')
    tokens.append('ksl:={' + ','.join(ksl_mad) + '}')
    tokens.append('pnl:={' + ','.join(pnl_mad) + '}')
    tokens.append('psl:={' + ','.join(psl_mad) + '}')
    tokens.append(mad_assignment('freq', _ge(rfmult.frequency) * 1e-6))
    tokens.append(mad_assignment('volt', _ge(rfmult.voltage) * 1e-6))
    tokens.append(mad_assignment('lag', _ge(rfmult.lag) / 360.))

    return ', '.join(tokens)

def dipoleedge_to_madx_str(name, line):
    if line.get_compound_for_element(name) is None:
        raise NotImplementedError("isolated dipole edges are not yet supported")
    return 'marker'

def bend_to_madx_str(name, line):
    bend = _get_eref(line, name)

    if isinstance(line.get_compound_by_name(line.get_compound_for_element(name)),
                                            xt.slicing.SlicedCompound):
        raise NotImplementedError("thick slicing of bends is not yet supported")

    tokens = []
    tokens.append('sbend')
    tokens.append(mad_assignment('l', _ge(bend.length)))
    tokens.append(mad_assignment('angle', _ge(bend.h) * _ge(bend.length)))
    tokens.append(mad_assignment('k0', _ge(bend.k0)))
    # k1, k2, knl, ksl need to be implemented
    if name + '_den' in line.element_dict.keys():
        edg_entry = line[name + '_den']
        tokens.append(mad_assignment('e1', _ge(edg_entry.e1)))
        tokens.append(mad_assignment('fint', _ge(edg_entry.fint)))
        tokens.append(mad_assignment('hgap', _ge(edg_entry.hgap)))
    if name + '_dex' in line.element_dict.keys():
        edg_exit = line[name + '_dex']
        tokens.append(mad_assignment('e2', _ge(edg_exit.e1)))
    return ', '.join(tokens)

def sextupole_to_madx_str(name, line):
    sext = _get_eref(line, name)
    tokens = []
    tokens.append('sextupole')
    tokens.append(mad_assignment('l', _ge(sext.length)))
    tokens.append(mad_assignment('k2', _ge(sext.k2)))
    tokens.append(mad_assignment('k2s', _ge(sext.k2s)))
    return ', '.join(tokens)

def quadrupole_to_madx_str(name, line):
    quad = _get_eref(line, name)
    tokens = []
    tokens.append('quadrupole')
    tokens.append(mad_assignment('l', _ge(quad.length)))
    tokens.append(mad_assignment('k1', _ge(quad.k1)))
    return ', '.join(tokens)

def solenoid_to_madx_str(name, line):
    sol = _get_eref(line, name)
    tokens = []
    tokens.append('solenoid')
    tokens.append(mad_assignment('l', _ge(sol.length)))
    tokens.append(mad_assignment('ks', _ge(sol.ks)))
    tokens.append(mad_assignment('ksi', _ge(sol.ksi)))
    return ', '.join(tokens)

def srotation_to_madx_str(name, line):
    srot = _get_eref(line, name)
    tokens = []
    tokens.append('srotation')
    tokens.append(mad_assignment('angle', _ge(srot.angle)*np.pi/180.))
    return ', '.join(tokens)

xsuite_to_mad_conveters={
    xt.Cavity: cavity_to_madx_str,
    xt.Marker: marker_to_madx_str,
    xt.Drift: drift_to_madx_str,
    xt.Multipole: multipole_to_madx_str,
    xt.DipoleEdge: marker_to_madx_str,
    xt.Bend: bend_to_madx_str,
    xt.Sextupole: sextupole_to_madx_str,
    xt.Quadrupole: quadrupole_to_madx_str,
    xt.Solenoid: solenoid_to_madx_str,
    xt.SRotation: srotation_to_madx_str,
    xt.RFMultipole: rfmultipole_to_madx_str,
}

def to_madx_sequence(line, name='seq'):
    # build variables part
    vars_str = ""
    for vv in line.vars.keys():
        if vv == '__vary_default':
            continue
        vars_str += mad_assignment(vv, line.vars[vv]) + ";\n"

    elements_str = ""
    for nn in line.element_names:
        el = line[nn]
        el_str = xsuite_to_mad_conveters[type(el)](nn, line)
        elements_str += f"{nn}: {el_str};\n"

    beam_str = (f"beam, mass={line.particle_ref.mass0*1e-9}, "
                f"charge={line.particle_ref.q0}, "
                f"gamma={line.particle_ref.gamma0[0]};\n")

    line_str = f'{name}: line=(' + ', '.join(line.element_names) + ');'

    mad_input = vars_str + '\n' + elements_str + '\n' + line_str + '\n' + beam_str
    return mad_input