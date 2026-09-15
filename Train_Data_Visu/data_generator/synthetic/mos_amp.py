import random

def _choice(vals, rng):
    return rng.choice(vals)

def _ohm(rng):
    return _choice(
        ["220", "470", "1k", "2.2k", "4.7k", "10k", "22k", "47k"],
        rng
    )

def _kohm(rng):
    return _choice(
        ["10k", "22k", "47k", "100k", "220k", "470k"],
        rng
    )

def _cap(rng):
    return _choice(
        ["10n", "47n", "100n", "220n", "1u", "10u"],
        rng
    )

def _volt(rng):
    return _choice(["5", "9", "12"], rng)

def _nl_header(cfg):
    """Generate natural language description header for MOSFET amplifier"""
    parts = ["A MOSFET amplifier circuit"]

    topo_desc = {
        "CS": "in a common-source configuration",
        "CD": "in a common-drain (source follower) configuration",
        "CG": "in a common-gate configuration"
    }
    parts.append(topo_desc[cfg["topology"]])

    if cfg["gate_bias"] == "divider":
        parts.append("using a resistive voltage divider gate bias")
    else:
        parts.append("using a single gate bias resistor")

    if cfg["source_deg"]:
        parts.append("with source degeneration")

    if cfg["input_coupling"]:
        parts.append("and an input coupling capacitor")

    if cfg["output_coupling"]:
        parts.append("and an output coupling capacitor")

    if cfg["load"]:
        parts.append("driving a resistive load")

    return " ".join(parts) + "."

def _nl_params(cfg):
    """Generate natural language description of component values"""
    s = [f"The circuit is powered by a {cfg['V']}V DC supply."]
    s.append(f"The drain resistor is {cfg['Rd']}.")

    if cfg["source_deg"]:
        s.append(f"The source resistor is {cfg['Rs']}.")

    if cfg["gate_bias"] == "divider":
        s.append(f"The gate bias resistors are {cfg['Rg1']} and {cfg['Rg2']}.")
    else:
        s.append(f"The gate bias resistor is {cfg['Rg']}.")

    if cfg["load"]:
        s.append(f"The load resistor is {cfg['Rl']}.")

    return " ".join(s)

def generate(n_samples, seed=202):
    """Generate MOSFET amplifier circuit training samples"""
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        cfg = {
            "topology": rng.choice(["CS", "CD", "CG"]),
            "gate_bias": rng.choice(["divider", "single"]),
            "source_deg": rng.choice([True, False]),
            "input_coupling": rng.choice([True, False]),
            "output_coupling": rng.choice([True, False]),
            "load": rng.choice([True, False]),
        }

        cfg["V"] = _volt(rng)
        cfg["Rd"] = _ohm(rng)
        cfg["Rs"] = _ohm(rng)

        if cfg["gate_bias"] == "divider":
            cfg["Rg1"] = _kohm(rng)
            cfg["Rg2"] = _kohm(rng)
        else:
            cfg["Rg"] = _kohm(rng)

        if cfg["load"]:
            cfg["Rl"] = _kohm(rng)

        nl = _nl_header(cfg) + " " + _nl_params(cfg)

        lines = []
        lines.append(f"VDD vdd 0 DC {cfg['V']}")

        if cfg["gate_bias"] == "divider":
            lines.append(f"R1 vdd gate {cfg['Rg1']}")
            lines.append(f"R2 gate 0 {cfg['Rg2']}")
        else:
            lines.append(f"Rg vdd gate {cfg['Rg']}")

        if cfg["input_coupling"]:
            lines.append(f"CIN in gate {_cap(rng)}")
        else:
            lines.append("Vin in gate AC 1")

        if cfg["topology"] == "CS":
            lines.append(f"Rd vdd drain {cfg['Rd']}")
            if cfg["source_deg"]:
                lines.append(f"Rs source 0 {cfg['Rs']}")
            else:
                lines.append("Rs source 0 0")
            lines.append("M1 drain gate source source NMOS")

        elif cfg["topology"] == "CD":
            lines.append("M1 vdd gate source source NMOS")
            lines.append(f"Rs source 0 {cfg['Rs']}")

        else:
            lines.append(f"Rd vdd drain {cfg['Rd']}")
            lines.append("M1 drain gate source source NMOS")
            if cfg["source_deg"]:
                lines.append(f"Rs source 0 {cfg['Rs']}")

        out_node = "drain" if cfg["topology"] != "CD" else "source"

        if cfg["output_coupling"]:
            lines.append(f"COUT {out_node} out {_cap(rng)}")
            if cfg["load"]:
                lines.append(f"RL out 0 {cfg['Rl']}")
        else:
            if cfg["load"]:
                lines.append(f"RL {out_node} 0 {cfg['Rl']}")

        lines.append(".end")

        spice = "\n".join(lines)
        samples.append((nl, spice))

    return samples
