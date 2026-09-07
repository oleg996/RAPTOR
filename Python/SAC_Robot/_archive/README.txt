ARCHIVE — reference only, NOT part of the active training/eval path.

- comb, comb.py, combine.sh .... generated concatenation dumps (not runnable).
- MJDATA.TXT .................... MuJoCo memory dump (scratch).
- robot.txt, code.txt .......... stale copies of simulation/robot.xml (old kp=30).
- train_tests_threaded.py ..... Godot-era gym trainer; calls update_from_buf with the OLD
                                 2-arg signature -> broken vs current sac_agent. Perf-oriented
                                 (was needed for Godot, not for MuJoCo).
- train_tests_threaded_tcp.py - Godot-era threaded trainer over TCP (same 2-arg bug).
- evaluate_AntV5_example.py ... evaluates gym Ant-v5, not this robot. Kept as an example.

The tcp/ socket code (../tcp) is still intended for the planned physical robot and was NOT
archived.
