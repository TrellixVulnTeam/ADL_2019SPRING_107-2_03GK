# ADL HW3
Please don't revise test.py, environment.py,  atari_wrapper.py, mario_env.py, agent_dir/agent.py

## How to train:
training policy gradient:
* `$ python3 main.py --train_pg`

training DQN:
* `$ python3 main.py --train_dqn`

training Mario:
* `$ python3 main.py --train_mario`

improve A2C model:
* `$ cd a2c`
* `$ python3 train.py`
There will be figure out after training, but it will cost long time.

improve ddqn model:
* `$ cd ddqn`
* `$ python3 PongNoFrameskip_ddqn.py` or `$ python3 PongNoFrameskip_dqn.py`
There will be figure out after training, but it will cost long time.

## How to plot:
1.If you want to get the figure of pg, just run `$ python3 main.py --train_pg`.After training, you will get the figure. And it won't takes lots of time.

2.If you want to get the figure which inclused all 4 different hyperparameters, you should run `$ python3 plot_4_imgs.py`. Learning curve of DQN is in it.


