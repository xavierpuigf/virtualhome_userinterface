export EXEC="/data/vision/torralba/frames/data_acquisition/SyntheticStories/website/release/simulator/v2.0/online_wah/online_wah_exec_2/linux_exec.online_wah_2.x86_64"
python vh_demo.py --deployment remote --execname $EXEC --task_group 3 304 420 --extra_agent none none none --exp_name testing_final_single_v
