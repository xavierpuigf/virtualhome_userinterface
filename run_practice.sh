export EXEC="/data/vision/torralba/frames/data_acquisition/SyntheticStories/website/release/simulator/v2.0/online_wah/online_wah_exec_2/linux_exec.online_wah_2.x86_64"
python vh_demo.py --deployment remote --execname $EXEC --task_group 270 --extra_agent none --exp_name $1 --portflask $2 --portvh $3
