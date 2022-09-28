# Duckrace Implementation

Folder containing the implementation of the duckrace algorithm in the real track.
To use install the duckietown shell DTS and run the following command to build:
```
dts devel build -f
```
And to run the built code:
```
dts devel run
```
Based on ROS, find the scripts inside packages/lmpc.

In the latest version the nodes are meant to be run on a pc connectected to the same network as the duckiebots.
If you want to run the code in the duckiebots use the flag -H in both build and run followed by the robot name (duckwalker.local or duckvader.local, duckvader is preferred) and remove ROS_MASTER_URI from launchers/default.sh (a script that runs when the container is launched).

## Packages:

### caller.py
Sends a message every n seconds to syncronize the MPC.

### controller.py
Runs the MPC

### run_lmpc.py
Runs the actual LMPC. IS makes at first a loop as MPC and then it starts to explore.
