import random

def _choice(vals, rng):
    return rng.choice(vals)

def _kohm(rng):
    return _choice(["1k", "2.2k", "4.7k", "10k", "22k", "47k"], rng)

def _cap(rng):
    return _choice(["10n", "47n", "100n", "220n", "1u", "10u"], rng)

def _volt(rng):
    return _choice(["3.3", "5", "9", "12"], rng)

def generate(n_samples, seed=99):
    """Generate cascaded multi-stage circuit training samples"""
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        depth = rng.choice([2, 3, 4])
        modules = rng.choices(["LP", "HP", "DIV"], k=depth)
        has_load = rng.choice([True, False])

        V = _volt(rng)

        # Create node chain: in -> n1 -> n2 -> ... -> out
        nodes = ["in"] + [f"n{i}" for i in range(1, depth)] + ["out"]

        tap_node = rng.choice(nodes[1:])

        module_desc = {
            "LP": "an RC low-pass stage",
            "HP": "an RC high-pass stage",
            "DIV": "a resistive divider stage"
        }

        nl_parts = [
            f"A cascaded circuit composed of {depth} stages",
            f"powered by a {V}V DC source."
        ]

        for i, m in enumerate(modules):
            nl_parts.append(f"Stage {i+1} is {module_desc[m]}.")

        nl_parts.append(f"The output is taken from node {tap_node}.")
        if has_load:
            nl_parts.append("A load resistor is connected at the output.")

        nl = " ".join(nl_parts)

        lines = []
        lines.append(f"V1 in 0 DC {V}")

        for i, m in enumerate(modules):
            n_left = nodes[i]
            n_right = nodes[i+1]

            if m == "LP":
                R = _kohm(rng)
                C = _cap(rng)
                lines.append(f"R{i+1} {n_left} {n_right} {R}")
                lines.append(f"C{i+1} {n_right} 0 {C}")

            elif m == "HP":
                R = _kohm(rng)
                C = _cap(rng)
                lines.append(f"C{i+1} {n_left} {n_right} {C}")
                lines.append(f"R{i+1} {n_right} 0 {R}")

            elif m == "DIV":
                R1 = _kohm(rng)
                R2 = _kohm(rng)
                lines.append(f"R{i+1}a {n_left} {n_right} {R1}")
                lines.append(f"R{i+1}b {n_right} 0 {R2}")

        if has_load:
            RL = _kohm(rng)
            lines.append(f"RL {tap_node} 0 {RL}")

        lines.append(".end")

        spice = "\n".join(lines)
        samples.append((nl, spice))

    return samples
