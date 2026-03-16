# MuJoCoGelSight
This is an attempt at an implementation of GelSight into MuJoCo.

After downloading the files, to check that the MuJoCo simulation is working properly run:

```bash
python3 grasp_controller.py
```

To run, first install packages:

```bash
pip install stable-baselines3 gymnasium opencv-python
```

Then to train:

```bash
python3 train.py
```

Lastly, to evaluate:

```bash
python3 evaluate.py
```
