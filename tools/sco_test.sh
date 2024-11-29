#!/usr/bin/env bash

set -x

GPUS=${GPUS:-1}
PWD=$(pwd)
CONFIG=$1
CKPT=$2
PART=${PART:-38}

if [ -z $CONFIG ]; then
    echo "Usage: $0 <config> [arg1] [arg2] ..."
    exit 1
fi

JOBNAME=$CONFIG
JOBNAME=${JOBNAME##*/}
JOBNAME=${JOBNAME%.*}

if [[ $PART -eq 38 ]]; then
    PART="--aec2-name vc3-node38 --worker-spec N1lS.Ib.I00.$GPUS"
elif [[ $PART -eq 17 ]] || [[ $PART -eq 10 ]]; then
    PART="--aec2-name vc2-node10 --worker-spec N1lS.Ib.I20.$GPUS"
else
    echo "Invalid PART: $PART"
    exit 1
fi

if [[ $GPUS -eq 1 ]]; then
    CMD="python tools/test.py $PROG $CONFIG $CKPT ${*:3}"
else
    CMD="bash tools/dist_test.sh"
fi

WORKSPACE_NAME=${WORKSPACE_NAME:-fvg-vc-research}
sco acp jobs create \
    --workspace-name $WORKSPACE_NAME \
    $PART \
    -f pt \
    -j j${JOBNAME:0:60} \
    --container-image-url registry.st-sh-01.sensecore.cn/studio-aicl/ubuntu20.04-py3.8-cuda11.8-cudnn8.9-torch2.0-mmcv2.0:v1.0.0-20230718-102821-65259bc \
    --storage-mount 588236f0-e523-11ee-99c2-5ef5bb6aa06a:/mnt/storage \
    --env HOME:/mnt/storage/user/wangruohui,WANDB_API_KEY:a754553c6f6914d238151c82c798beaacd64d1f5,WANDB_JOB_NAME:$JOBNAME \
    --command "set -x; useradd $USER -u `id -u` -d $HOME -s $SHELL; cd $PWD; echo $SHELL; cd $PWD; source activate pt2; $CMD; ex=\$?; chown -R $USER:$USER $PWD/work_dirs; exit \$ex" \
    | tee /tmp/joboutput

jobid=$( grep -oP "(?<=id : )\w+-\w+"  /tmp/joboutput )

sco acp jobs stream-logs --workspace-name fvg-vc-research $jobid