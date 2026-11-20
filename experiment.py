"""
Empirical check of Berend & Kontorovich (2016): the minimal DFA of a uniformly
random n-state DFA over a k-letter alphabet has about alpha_k * n states, where
alpha_k is the positive root of x = 1 - e^(-kx).

Produces alpha_k_convergence.png and prints the alpha_k comparison table.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

KS = (2, 3, 4)                      # alphabet sizes
NS = 2 ** np.arange(3, 18)          # n = 8, 16, ..., 131072
TRIALS = 200                        # random DFAs per (k, n)


def alpha(k):
    """Positive root of x = 1 - e^(-kx), via the Lambert W function."""
    return 1 + lambertw(-k * np.exp(-k)).real / k


def random_dfa(n, k, rng):
    """Uniform random DFA: every transition and every accept bit chosen independently."""
    delta = rng.integers(0, n, size=(n, k))   # delta[q, a] = next state
    accepting = rng.random(n) < 0.5
    return delta, accepting                    # start state is 0


def reachable(delta):
    """Boolean mask of states reachable from state 0 (breadth-first search)."""
    seen = np.zeros(len(delta), dtype=bool)
    seen[0] = True
    frontier = np.array([0])
    while frontier.size:
        nxt = np.unique(delta[frontier])
        frontier = nxt[~seen[nxt]]
        seen[frontier] = True
    return seen


def moore_minimize(delta, accepting):
    """Number of states in the minimal DFA (Moore's partition refinement).
    Assumes every state is reachable."""
    block = accepting.astype(int)
    num_blocks = len(np.unique(block))
    while True:
        # signature of q = (block of q, block of delta(q,0), ..., block of delta(q,k-1))
        signature = np.column_stack([block, block[delta]])
        _, new_block = np.unique(signature, axis=0, return_inverse=True)
        new_block = new_block.ravel()
        if new_block.max() + 1 == num_blocks:   # no block split: partition is stable
            return num_blocks
        block, num_blocks = new_block, new_block.max() + 1


def trial(n, k, rng):
    """Return (#reachable states, #states in minimal DFA) for one random DFA."""
    delta, accepting = random_dfa(n, k, rng)
    mask = reachable(delta)
    new_id = np.cumsum(mask) - 1                 # renumber reachable states 0..R-1
    sub_delta = new_id[delta[mask]]
    return mask.sum(), moore_minimize(sub_delta, accepting[mask])


def main():
    rng = np.random.default_rng(475)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    print(f"{'k':>2} {'alpha_k':>9} {'R/n':>9} {'|M|/n':>9}   (n = {NS[-1]})")
    for k in KS:
        reach_frac, min_frac = [], []
        for n in NS:
            res = np.array([trial(n, k, rng) for _ in range(TRIALS)])
            reach_frac.append(res[:, 0].mean() / n)
            min_frac.append(res[:, 1].mean() / n)
        line, = ax.plot(NS, min_frac, "o-", label=f"k = {k}")
        ax.axhline(alpha(k), ls="--", color=line.get_color())
        print(f"{k:>2} {alpha(k):>9.5f} {reach_frac[-1]:>9.5f} {min_frac[-1]:>9.5f}")
    ax.set(xscale="log", xlabel="n (number of states)", ylabel="minimal DFA size / n",
           title="Minimal DFA size of random DFAs vs. $\\alpha_k n$ (dashed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("alpha_k_convergence.png", dpi=300)


if __name__ == "__main__":
    main()
