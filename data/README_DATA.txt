SMALL TEST DATASET - SYNTHETIC (not measured).
steering_log_SYNTHETIC.csv : DeepPicar log format (ts_ms, frame_id, angle_rad, btn, latency_ms).
  Angles = the report's left-right slalom schedule + Gaussian noise; timing jitter and latency (50 ms +/- 6, 2% spikes)
  are ASSUMED values chosen to resemble the report's nominal 43.7 Hz / 50 ms.
terrain_SYNTHETIC.csv      : elevation every 0.5 m, built from designed slopes (0,+4,0,-5,0 deg) and 5 surfaces.
surfaces.json              : mu, C_rr per surface - typical published car-tyre values / assumptions; measure your own.
Replace with real logs when available (e.g. head -500 of a real DeepPicar or Udacity log).
