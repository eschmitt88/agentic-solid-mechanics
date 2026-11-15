"""Orchestrator-side grading of the frozen DMC prediction (written after the
prediction was committed in 8a5afd7).  Experimental values are read off
Fig. 3, 4 and 9 of Morris et al., Eng. Fract. Mech. 310 (2024) 110421 by eye;
the paper tabulates none of them."""
import json, sys
sys.path.insert(0, "../2026-11-14-instability-path-following")
from plotstyle import setup, C
plt = setup()
pred = json.load(open("results/dmc_prediction.json"))["Q1_load_displacement"]
meas = [305, 330, 345]          # N, read-off +-10 N
teams = (320, 335)              # N, four expert teams, read-off from Fig. 9
mean = sum(meas) / 3
g = {
 "source": "Morris et al., Eng. Fract. Mech. 310 (2024) 110421, Figs 3, 4, 9 (graphical read-off by the orchestrator; exact per-specimen data are on PURR doi:10.4231/ZX5A-QJ30 behind Globus and were not obtained)",
 "prediction_commit": "8a5afd7", "prediction_sha256": "33e0fb4356f7d0e25c4901958a2b7358f1d4380d975e16191b0394ac11f843a2",
 "peak_load": {"predicted_N": pred["peak_load_N"], "measured_peaks_N_readoff": meas, "readoff_uncertainty_N": 10,
   "measured_mean_N": round(mean, 1), "error_pct": round(100 * (pred["peak_load_N"] / mean - 1), 1),
   "inside_own_80pct_interval_for_mean": bool(pred["peak_load_80pct_interval_mean_of_specimens_N"][0] <= mean <= pred["peak_load_80pct_interval_mean_of_specimens_N"][1]),
   "expert_teams_range_N_readoff": list(teams), "verdict": "under-predicted by roughly 9 % (5-12 % allowing for read-off error); below all four expert teams"},
 "displacement_at_peak": {"predicted_mm": pred["displacement_at_peak_mm"], "measured_mm_readoff": [0.37, 0.385, 0.44], "verdict": "inside the measured range"},
 "stiffness": {"predicted_N_per_mm": 800, "verdict": "consistent with the measured slope; the measured curves stiffen with load, which the contact law reproduces only on average"},
 "post_peak": {"predicted": "abrupt drop, low tail", "measured": "abrupt drop, low tail", "verdict": "correct"},
 "crack_path": {"predicted_top_exit_mm": -3.8, "predicted_range_mm": [-6.4, -2.8], "measured_top_exit_front_mm_readoff": -4, "measured_top_exit_back_mm_readoff": -2, "readoff_uncertainty_mm": 1,
   "predicted_start_front_back_mm": [-11.6, -7.1], "measured_start_front_back_mm_readoff": [-12, -6],
   "verdict": "start positions and front exit within about 1 mm; back exit about 2 mm off and just outside the stated range; predicted one exit position for both faces whereas the measured faces differ by about 2 mm"},
 "not_graded": "organisers' area-based error metrics (need the digital experimental envelope); displacement fields and roughness were not predicted",
}
json.dump(g, open("results/dmc_grading.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(7, 3.4))
lo, hi = pred["peak_load_80pct_interval_mean_of_specimens_N"]
ax.plot([lo, hi], [2, 2], color=C["blue"], lw=6, alpha=.3, solid_capstyle="round")
ax.plot(pred["peak_load_N"], 2, "o", color=C["blue"], ms=11)
ax.plot(teams, [1, 1], color=C["orange"], lw=6, solid_capstyle="round")
ax.plot(meas, [0, 0, 0], "D", color=C["ref"], ms=9)
ax.axvline(mean, color=C["ref"], lw=1, ls="--")
ax.text(mean + 2, 2.42, f"measured mean ≈ {mean:.0f} N", fontsize=10)
ax.set_yticks([2, 1, 0], ["agent: 298 N\n(80 % interval)", "four expert teams", "experiment\n(3 curves)"])
ax.set_ylim(-.6, 2.8); ax.set_xlim(255, 355); ax.set_xlabel("peak load (N)")
ax.set_title("Damage Mechanics Challenge: blind peak-load prediction, graded")
ax.grid(axis="y", visible=False)
fig.savefig("results/fig_dmc_grading.png")
print(json.dumps(g["peak_load"]))
