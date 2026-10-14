# DeepPicar: single MPC controller on sloped, multi-surface terrain with moving obstacles

Physics simulation of a 1:24 RC car (DeepPicar) for the course 22AIE441 Dynamics and Control of Robotics.
One Model Predictive Controller outputs steering and throttle/brake (replacing CNN + PD + PI).

## Files
| File | What it does |
|---|---|
| `deeppicar_sim_v4_mpc.py` | Main simulation: MPC, terrain, static + moving obstacles, braking, road bounds, timing sweep, figures |
| `deeppicar_sim_v3.py` | Physics, surfaces, terrain/log loaders, CNN stand-in (used by v4 for the comparison) |
| `deeppicar_3d.html` | 3D simulator with live graphs (open in a browser; needs internet for the Three.js CDN) |
| `data/` | Small SYNTHETIC test dataset (steering log, terrain table, surface values) |
| `images/` | Result figures, flowchart, architecture |

## Run
```
pip install -r requirements.txt
python3 deeppicar_sim_v4_mpc.py                 # full run (about 20 s), figures in runs/v4_mpc
python3 deeppicar_sim_v4_mpc.py --quick         # skip the sweeps
python3 deeppicar_sim_v4_mpc.py --terrain data/terrain_SYNTHETIC.csv --surfaces-json data/surfaces.json
python3 deeppicar_sim_v3.py --dataset data/steering_log_SYNTHETIC.csv --terrain data/terrain_SYNTHETIC.csv --surfaces-json data/surfaces.json
python3 deeppicar_sim_v3.py --list-datasets     # where to get real DeepPicar / Udacity logs
```
`--dataset` accepts DeepPicar logs (`ts_ms,frame_id,angle_rad[,btn,latency_ms]`) and Udacity `driving_log.csv`; `--terrain` accepts a GPX track or a CSV (`distance_m`, `elevation_m` or `slope_deg`, optional `surface`).

## Results (mixed course, target 1.2 m/s)
- MPC with 3 cones + 2 pedestrians + 1 slow car: k=1 and 4 no collisions; k=8 one; k=11.6 two (loop slowdown k: period x k, latency 50 ms x k). No curb contacts.
- No obstacles, RMS cross-track error (m): CNN stand-in 0.097 / 0.227 / 0.335 / 0.495 vs MPC 0.088 / 0.072 / 0.064 / 0.113 for k = 1 / 4 / 8 / 11.6.

## Assumptions and limits
- The test dataset is synthetic, not measured. Surface mu and C_rr, servo lag, motor and the 1.0 N brake force are assumptions or typical published values.
- The MPC knows the exact model, terrain and obstacle positions (no sensor noise or model error).
- The CNN is a stand-in, not the trained network. With a very low target speed the car can stall (no reverse).
