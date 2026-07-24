"""
Standalone XFoil alpha-sweep runner.

Drives the xfoil binary directly via stdin/stdout instead of using
aerosandbox's XFoil wrapper. This avoids the PACC polar-file-write
codepath in iopol.f, which crashes on some gfortran-built xfoil
binaries with:

    Fortran runtime error: Sequential READ or WRITE not allowed
    after EOF marker, possibly use REWIND or BACKSPACE

Instead, this parses the "a = ... CL = ..." / "Cm = ... CD = ..."
lines that XFoil prints straight to stdout after each converged
operating point -- no file I/O involved.
"""

import subprocess
import re
import numpy as np
import pandas as pd


def xfoil_alpha_sweep(
    xfoil_path,
    naca_code,
    alphas,
    Re,
    n_iter=100,
    mach=0.0,
    timeout=120,
):

    alphas = np.atleast_1d(alphas)

    # Build the command script: init once, then loop ALFA commands.
    # Blank line after each ALFA lets XFoil settle before the next command;
    # not strictly required but harmless.
    cmds = [
        f"NACA {naca_code}",
        "OPER",
        f"MACH {mach}",
        f"VISC {Re}",
        f"ITER {n_iter}",
        "PACC",  # NOTE: intentionally NOT used -- see below
    ]
    # We do NOT actually enable polar accumulation (PACC needs a filename
    # and that's exactly the crashing codepath) -- remove the PACC call.
    cmds.pop()  # drop the stray "PACC" above

    for a in alphas:
        cmds.append(f"ALFA {a}")

    cmds.append("")
    cmds.append("QUIT")
    command_str = "\n".join(cmds) + "\n"

    result = subprocess.run(
        [xfoil_path],
        input=command_str,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    stdout = result.stdout

    # Each converged point prints a block like:
    #        a =  5.000      CL =  0.8094
    #       Cm = -0.0540     CD =  0.00775   =>   CDf =  0.00497    CDp =  0.00278
    pattern = re.compile(
        r"a\s*=\s*(-?[\d.]+)\s*CL\s*=\s*(-?[\d.]+)\s*\n"
        r"\s*Cm\s*=\s*(-?[\d.]+)\s*CD\s*=\s*(-?[\d.]+)"
    )

    matches = pattern.findall(stdout)

    if not matches:
        print("---- XFoil stdout (no points parsed) ----")
        print(stdout)
        raise RuntimeError(
            "No converged points found. Full XFoil output printed above "
            "for debugging."
        )

    rows = []
    for a, cl, cm, cd in matches:
        rows.append(
            {
                "alpha": float(a),
                "CL": float(cl),
                "CD": float(cd),
                "CM": float(cm),
            }
        )

    df = pd.DataFrame(rows)

    # XFoil sometimes repeats the same "a = ..." block multiple times per
    # alpha while iterating toward convergence (each Newton iteration can
    # print its own progress block that happens to match this pattern in
    # noisy cases). Keep only the LAST result for each unique alpha, since
    # that's the converged value.
    df = df.drop_duplicates(subset="alpha", keep="last").reset_index(drop=True)

    n_requested = len(alphas)
    n_converged = len(df)
    if n_converged < n_requested:
        converged_alphas = set(df["alpha"].round(3))
        missing = [a for a in alphas if round(float(a), 3) not in converged_alphas]
        print(
            f"Warning: {n_requested - n_converged} of {n_requested} points "
            f"did not converge and were skipped: {missing}"
        )

    return df


if __name__ == "__main__":
    alphas = np.arange(-10, 20, 0.5)

    df = xfoil_alpha_sweep(
        xfoil_path="/home/poogle/Xfoil/bin/xfoil",
        naca_code="2412",
        alphas=alphas,
        Re=1e6,
    )

    print(df)

    # Optional: save to CSV and plot
    df.to_csv("naca2412_polar.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        axes[0].plot(df["alpha"], df["CL"], "o-")
        axes[0].set_xlabel("Alpha (deg)")
        axes[0].set_ylabel("CL")
        axes[0].grid(True)

        axes[1].plot(df["CD"], df["CL"], "o-")
        axes[1].set_xlabel("CD")
        axes[1].set_ylabel("CL")
        axes[1].grid(True)

        axes[2].plot(df["alpha"], df["CM"], "o-")
        axes[2].set_xlabel("alpha")
        axes[2].set_ylabel("CM")
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig("naca2412_polar.png", dpi=150)
        print("Saved plot to naca2412_polar.png")
    except ImportError:
        pass
    print(df["alpha"])
    print(df["CL"].to_numpy())
    print(df["CD"].to_numpy())
    print(df["CM"].to_numpy())