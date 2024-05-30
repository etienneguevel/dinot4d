# How FSDP is implemented

The different steps that are implemented in order to make FSDP work are listed in this note.
**[Fully Sharded Distributed Parallelism](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)** is a setup where each process/ worker owns a replica of the model and processes a batch of the data. Afterwards, it uses all reduce algorithm to sum up gradients over different workers.

When training with FSDP, the GPU memory footprint is smaller than when training with DDP across all workers. This makes the training of some very large models feasible by allowing larger models or batch sizes to fit on device. This comes with the cost of increased communication volume. The communication overhead is reduced by internal optimizations like overlapping communication and computation.

## TLDR

The model class that is used to define the model is deeply implemented for fsdp, with several methods made for this framework. Especially, it has a method to shard the models, that is called at the start of the training script.
The submit script makes a Trainer object that calls the training script. This Trainer is setup and submitted to the slurm env thanks to submitit functions that gather informations about the cluster, and then makes the .sh script that is executed on the slurm env.

## Setup the model for distributed training (or evaluation)

The model is defined in the `dinov2/train/ssl_meta_arch.py` file and with the class `SSLMetaArch`.
There are several methods and attributes in this class that are made for FSDP, and every method is configured to work for this framework.
This class has a method called `prepare_for_distributed_training()`, that loop over every params of the student / teacher and:

- decide the precision for this param (fp16 or fp32)
- shard these params across data parallel workers according to the parameters decided in the cfg file

This function is called in the `main` function when using the `dinov2/train/train.py` file after having initialized the model.
> After creating the model, it is sharded across the different workers of the slurm environment

## Define the slurm environment and submit the jobs

The training loop is called by using the `dinov2/run/train/train.py` script.
First the script parses the args that are indicated after the command line, that will be used for the configuration of the .sh command and of the setup of the slurm env.
Then the script uses the submitit library to submit jobs with the corresponding args with `submit_jobs()` from the `dinov2/run/submit.py` file.

This function:

- creates an `AutoExecutor` for job submission
- sets additional Slurm job parameters
- initialize a tasl instance with the provided args
- submits the task to the slurm cluster using the executor
