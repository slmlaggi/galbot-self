## WIP
`manager.py` is broken-ish(?), but `training.py` in theory should work and have no issues.

## config
Should be in line 331 of the `training.py`.
- Total episodes: If not using cuda, ideally do not put more than 400 episodes. Else, after tuning the other config parameters or a new algorithm and it works within ~400 episodes, can increase num episodes up to 1000-2000
- Max steps: 80-250, depend on the size of your environment and your desired task
- update/save freq, headless & random seed: Keep unchanged, changing might break the alg
- lr: change if needed, keep at or under 3e-4  for stability
- gamma & eps clip: PPO initialization parameters, change to test the best parameters
- k_epochs and hidden_dim: ideally want k to be 16, but will take more time to train. Same with hidden_dim at 1024.
