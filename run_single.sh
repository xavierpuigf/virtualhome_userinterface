#!/bin/bash

export EXEC="/data/vision/torralba/frames/data_acquisition/SyntheticStories/website/release/simulator/v2.0/online_wah/online_wah_exec_2/linux_exec.online_wah_2.x86_64"
task=$4
types=$5
export tasks=${task//\,/ }
export types=${types//\,/ }
python vh_demo.py --deployment remote --execname $EXEC --task_group $tasks --extra_agent $types --exp_name $1 --portflask $2 --portvh $3
