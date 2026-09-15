import random

def _choice(vals, rng):
    return rng.choice(vals)

def _ohm(rng):
    return _choice(
        ["220", "470", "1k", "2.2k", "4.7k", "10k", "22k", "47k", "100k"],
        rng
    )

def _kohm(rng):
    return _choice(
        ["2.2k", "4.7k", "10k", "22k", "47k", "100k", "220k"],
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
    """Generate natural language description header for BJT amplifier"""
    parts = ["A BJT amplifier circuit"]

    parts.append(
        "in a common-emitter configuration"
        if cfg["topology"] == "CE"
        else "in a common-collector configuration"
    )

    if cfg["bias"] == "divider":
        parts.append("using a voltage divider bias")
    else:
        parts.append("using a single-resistor base bias")

    if cfg["emitter_bypass"]:
        parts.append("with an emitter bypass capacitor")

    if cfg["input_coupling"]:
        parts.append("and an input coupling capacitor")

    if cfg["output_coupling"]:
        parts.append("and an output coupling capacitor")

    if cfg["feedback"]:
        parts.append("with resistive feedback from output to base")

    return " ".join(parts) + "."

def _nl_params(cfg):
    """Generate natural language description of component values"""
    s = [f"The circuit is powered by a {cfg['V']}V DC supply."]
    s.append(f"The collector resistor is {cfg['Rc']} and the emitter resistor is {cfg['Re']}.")

    if cfg["bias"] == "divider":
        s.append(f"The base bias resistors are {cfg['Rb1']} and {cfg['Rb2']}.")
    else:
        s.append(f"The base bias resistor is {cfg['Rb']}.")

    if cfg["load"]:
        s.append(f"The load resistor is {cfg['Rl']}.")

    return " ".join(s)

def generate(n_samples, seed=123):
    """Generate BJT amplifier circuit training samples"""
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples):
        cfg = {
            "topology": rng.choice(["CE", "CC"]),
            "bias": rng.choice(["divider", "single"]),
            "emitter_bypass": rng.choice([True, False]),
            "input_coupling": rng.choice([True, False]),
            "output_coupling": rng.choice([True, False]),
            "feedback": rng.choice([True, False]),
            "load": rng.choice([True, False]),
        }

        cfg["V"] = _volt(rng)
        cfg["Rc"] = _ohm(rng)
        cfg["Re"] = _ohm(rng)

        if cfg["bias"] == "divider":
            cfg["Rb1"] = _kohm(rng)
            cfg["Rb2"] = _kohm(rng)
        else:
            cfg["Rb"] = _kohm(rng)

        if cfg["load"]:
            cfg["Rl"] = _kohm(rng)

        nl = _nl_header(cfg) + " " + _nl_params(cfg)

        lines = []
        lines.append(f"VCC vcc 0 DC {cfg['V']}")

        if cfg["bias"] == "divider":
            lines.append(f"R1 vcc base {cfg['Rb1']}")
            lines.append(f"R2 base 0 {cfg['Rb2']}")
        else:
            lines.append(f"Rb vcc base {cfg['Rb']}")

        if cfg["input_coupling"]:
            lines.append(f"CIN in base {_cap(rng)}")
        else:
            lines.append("Vin in base AC 1")

        if cfg["topology"] == "CE":
            lines.append(f"Rc vcc collector {cfg['Rc']}")
            lines.append(f"Re emitter 0 {cfg['Re']}")
            lines.append("Q1 collector base emitter QNPN")
        else:
            lines.append("Q1 vcc base emitter QNPN")
            lines.append(f"Re emitter 0 {cfg['Re']}")

        if cfg["emitter_bypass"]:
            lines.append(f"CE emitter 0 {_cap(rng)}")

        out_node = "collector" if cfg["topology"] == "CE" else "emitter"

        if cfg["output_coupling"]:
            lines.append(f"COUT {out_node} out {_cap(rng)}")
            if cfg["load"]:
                lines.append(f"RL out 0 {cfg['Rl']}")
        else:
            if cfg["load"]:
                lines.append(f"RL {out_node} 0 {cfg['Rl']}")

        if cfg["feedback"]:
            lines.append(f"RF out base {_kohm(rng)}")

        lines.append(".end")

        spice = "\n".join(lines)
        samples.append((nl, spice))

    return samples
