# Usage 

## Abberior

It is assumed that the pretrained model is available in a pretrained folder in this directory.

Update the date in `run_abberior.py` file.

Launch the experiment 
```bash
python run_abberior.py --model-name 20230711-100513_8abb567f --savedir ./pretrained --checkpoint 12000000 --env gym_sted:AbberiorMOSTEDCountRate-v0
```

### Configuration

Update `abberior_action_spaces` from `gym_sted/defaults.py`

Update the default configuration of the abberior microscope from `gym_sted/microscopes/config/default-abberior-config.yml`

Update `self.conf_params` in `AbberiorSTEDCountRateMultiObjectivesEnv` from `gym_sted/envs/abberior_env.py`

### Video tutorial

We provide a video tutorial that goes through the steps required to install and run an experiment. Click on the image below.

[![Watch the video](https://img.youtube.com/vi/bnkxeaaof14/hqdefault.jpg)](https://www.youtube.com/watch?v=bnkxeaaof14)
