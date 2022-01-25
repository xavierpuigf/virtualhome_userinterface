# VirtualHome Data Collection 
Interface for collecting experiments in VirtualHome

![img](assets/screenshot_interface.png)

## Setup
Clone the VirtualHome repo inside this folder, or create a symlink.

```
cd virtualhome_userinterface
git clone https://github.com/xavierpuigf/virtualhome.git
```

Download and unzip the Unity Executable.

```
wget http://virtual-home.org/release/simulator/last_release/linux_exec.zip
unzip linux_exec.zip
```

If you are running the demo locally, run:

```
export EXEC="virtualhome/simulation/linux_exec.v2.2.4.x86_64"
python vh_demo.py --deployment local --execname $EXEC
```
Otherwise run

```
export EXEC="virtualhome/simulation/linux_exec.v2.2.4.x86_64"
python vh_demo.py --deployment remote --execname $EXEC
```

## Graph to GIF
Put the json files to `vh_collect_data/graph_dir`, run
```
cd tools
python plotting_code.py
```
The generated GIF can be found in `tools/video_name.gif`.

## Cite
If you find this code useful, please consider citing our work:

```
@inproceedings{puig2020watch,
  title={Watch-And-Help: A Challenge for Social Perception and Human-AI Collaboration},
  author={Puig, Xavier and Shu, Tianmin and Li, Shuang and Wang, Zilin and Liao, Yuan-Hong and Tenenbaum, Joshua B and Fidler, Sanja and Torralba, Antonio},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```
