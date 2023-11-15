export WANDB_API_KEY= #Your API Wandb Key
export HYDRA_FULL_ERROR=1

python src/train.py experiment=cifar trainer=gpu