# VirtualHome Data Collection 
Interface for collecting experiments in VirtualHome

## Setup
Clone the VirtualHome repo inside this folder, or create a symlink.

```
cd virtualhome_userinterface
git clone https://github.com/xavierpuigf/virtualhome.git
```

Download the Unity Executable and put it under `simulation/unity_simulator`.

```
cd virtualhome
# Download 'exec_linux.06.04.x86_64' and 'exec_linux.06.04_Data' and put them under 'simulation/unity_simulator 
virtualhome/simulation/unity_simulator/exec_linux.06.04.x86_64 -batchmode & 
```

If you are running the demo locally, run:

```
python vh_demo.py --deployment local
```
Otherwise run

```
python vh_demo.py --deployment remote
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
