import random

def _choice(vals, rng):
    return rng.choice(vals)

def _kohm(rng):
    return _choice(["1k", "2.2k", "4.7k", "10k", "22k", "47k", "100k"], rng)

def _cap(rng):
    return _choice(["10n", "47n", "100n", "220n", "1u", "10u"], rng)

def _volt(rng):
    return _choice(["3.3", "5", "9", "12"], rng)

def generate(n_samples, seed=7):
    """Generate feedback circuit training samples"""
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        depth = rng.choice([1, 2])
        ordering = rng.choice(["LP", "HP"])

        V = _volt(rng)

        # Create node chain based on depth
        nodes = ["in"] + [f"n{i}" for i in range(1, depth+1)] + ["out"]

        fb_type = rng.choice(["R", "C", "RC"])
        fb_from = rng.choice(nodes[1:])
        fb_to = rng.choice(["in", "n1"])

        # Prevent meaningless self-connection
        if fb_from == fb_to:
            fb_to = "in"

        nl_parts = [
            "A feedback circuit based on an RC network",
            f"powered by a {V}V DC source."
        ]

        nl_parts.append(
            "The main path is a low-pass RC stage."
            if ordering == "LP"
            else "The main path is a high-pass RC stage."
        )

        if depth == 2:
            nl_parts.append("The circuit uses two cascaded RC stages.")

        fb_desc = {
            "R": "a resistive feedback path",
            "C": "a capacitive feedback path",
            "RC": "a resistive-capacitive feedback network"
        }
        nl_parts.append(
            f"The feedback is implemented using {fb_desc[fb_type]}, "
            f"from node {fb_from} to node {fb_to}."
        )

        nl = " ".join(nl_parts)

        lines = []
        lines.append(f"V1 in 0 DC {V}")

        # Main path stages
        for i in range(depth):
            n_left = nodes[i]
            n_right = nodes[i+1]

            R = _kohm(rng)
            C = _cap(rng)

            if ordering == "LP":
                lines.append(f"R{i+1} {n_left} {n_right} {R}")
                lines.append(f"C{i+1} {n_right} 0 {C}")
            else:
                lines.append(f"C{i+1} {n_left} {n_right} {C}")
                lines.append(f"R{i+1} {n_right} 0 {R}")

        # Feedback path
        if fb_type == "R":
            lines.append(f"Rf {fb_from} {fb_to} {_kohm(rng)}")
        elif fb_type == "C":
            lines.append(f"Cf {fb_from} {fb_to} {_cap(rng)}")
        else:
            lines.append(f"Rf {fb_from} nf {_kohm(rng)}")
            lines.append(f"Cf nf {fb_to} {_cap(rng)}")

        lines.append(".end")

        spice = "\n".join(lines)
        samples.append((nl, spice))

    return samples
