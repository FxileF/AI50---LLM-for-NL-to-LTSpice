import random

def _choice(vals, rng):
    return rng.choice(vals)

def _kOhm(rng):
    return _choice(["1k", "2.2k", "4.7k", "10k", "22k", "47k", "100k"], rng)

def _cap(rng):
    return _choice(["10n", "47n", "100n", "220n", "470n", "1u", "2.2u", "10u"], rng)

def _volt(rng):
    return _choice(["3.3", "5", "9", "12"], rng)

def _mk_nodes(depth):
    """Create node chain: in -> n1 -> n2 -> ... -> out"""
    if depth == 1:
        return ["in", "out"]
    mids = [f"n{i}" for i in range(1, depth)]
    return ["in"] + mids + ["out"]

def _nl_header(ordering, depth, has_load, tap, ladder):
    """Generate natural language description header"""
    ftype = "low-pass" if ordering == "LP" else "high-pass"
    stage_txt = "single-stage" if depth == 1 else f"{depth}-stage"
    parts = [f"A {stage_txt} RC {ftype} filter"]
    if ladder:
        parts.append("implemented as an RC ladder network")
    if tap != "out":
        parts.append(f"with the output taken at {tap}")
    if has_load:
        parts.append("and a resistive load at the output")
    return " ".join(parts) + "."

def _nl_params(V, ordering, depth, Rs, Cs, loadR, ladder):
    """Generate natural language description of component values"""
    ftype = "low-pass" if ordering == "LP" else "high-pass"
    s = [f"It is powered by a {V}V DC source."]
    s.append(f"The filter type is {ftype}.")
    if ladder:
        s.append(f"The network has {depth} stage(s) connected in cascade.")
    else:
        s.append(f"The filter has {depth} stage(s).")

    for i in range(depth):
        s.append(f"Stage {i+1} uses R{i+1}={Rs[i]} and C{i+1}={Cs[i]}.")
    if loadR is not None:
        s.append(f"The load resistor is RL={loadR}.")
    return " ".join(s)

def _spice_lp(nodes, Rs, Cs, loadR=None, tap_node="out"):
    """Generate low-pass filter SPICE netlist: series R then shunt C"""
    lines = []
    lines.append(f"V1 in 0 DC {{V}}")
    for i in range(len(Rs)):
        n_left = nodes[i]
        n_right = nodes[i+1]
        lines.append(f"R{i+1} {n_left} {n_right} {Rs[i]}")
        lines.append(f"C{i+1} {n_right} 0 {Cs[i]}")
    if loadR is not None:
        lines.append(f"RL {tap_node} 0 {loadR}")
    lines.append(".end")
    return lines

def _spice_hp(nodes, Rs, Cs, loadR=None, tap_node="out"):
    """Generate high-pass filter SPICE netlist: series C then shunt R"""
    lines = []
    lines.append(f"V1 in 0 DC {{V}}")
    for i in range(len(Rs)):
        n_left = nodes[i]
        n_right = nodes[i+1]
        lines.append(f"C{i+1} {n_left} {n_right} {Cs[i]}")
        lines.append(f"R{i+1} {n_right} 0 {Rs[i]}")
    if loadR is not None:
        lines.append(f"RL {tap_node} 0 {loadR}")
    lines.append(".end")
    return lines

def generate(n_samples, seed=42):
    """
    Generate RC filter circuit training samples.
    
    Produces diverse topologies by combining:
      - ordering: LP/HP (low-pass/high-pass)
      - depth: 1..3 (number of stages)
      - ladder: True/False (network structure variation)
      - load: None / RL at tap node
      - tap: output taken at out / n1 / n2 (when available)
    """
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        ordering = rng.choice(["LP", "HP"])
        depth = rng.choice([1, 2, 3])
        ladder = rng.choice([True, False])
        has_load = rng.choice([True, False])

        V = _volt(rng)
        Rs = [_kOhm(rng) for _ in range(depth)]
        Cs = [_cap(rng) for _ in range(depth)]
        loadR = _kOhm(rng) if has_load else None

        nodes = _mk_nodes(depth)

        possible_taps = ["out"]
        if depth >= 2:
            possible_taps.append("n1")
        if depth >= 3:
            possible_taps.append("n2")
        tap_node = rng.choice(possible_taps)

        nl = _nl_header(ordering, depth, has_load, tap_node, ladder) + " " + _nl_params(V, ordering, depth, Rs, Cs, loadR, ladder)

        if ordering == "LP":
            lines = _spice_lp(nodes, Rs, Cs, loadR=loadR, tap_node=tap_node)
        else:
            lines = _spice_hp(nodes, Rs, Cs, loadR=loadR, tap_node=tap_node)

        lines[0] = lines[0].format(V=V)
        spice = "\n".join(lines)

        samples.append((nl, spice))

    return samples
