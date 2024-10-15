#!/usr/bin/env bash

set -x

NNODES=${N:-1}
GPUS=${GPUS:-8}
PWD=$(pwd)
CONFIG=$1
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

WORKSPACE_NAME=${WORKSPACE_NAME:-fvg-vc-research}
sco acp jobs create \
    --workspace-name $WORKSPACE_NAME \
    $PART \
    -N $NNODES \
    -f pt \
    -j j$JOBNAME \
    --container-image-url registry.st-sh-01.sensecore.cn/studio-aicl/ubuntu20.04-py3.8-cuda11.8-cudnn8.9-torch2.0-mmcv2.0:v1.0.0-20230718-102821-65259bc \
    --storage-mount 588236f0-e523-11ee-99c2-5ef5bb6aa06a:/mnt/storage \
    --env HOME:/mnt/storage/user/wangruohui,WANDB_API_KEY:a754553c6f6914d238151c82c798beaacd64d1f5,WANDB_NAME:$JOBNAME \
    --command "set -x; useradd $USER -u `id -u` -d $HOME -s $SHELL; cd $PWD; echo $SHELL; cd $PWD; source ~/.mybashrc; pon; source activate pt2; tools/dist_train.sh $CONFIG $GPUS ${*:2} 2>&1 | tee \$HOSTNAME.log; ex=\${PIPESTATUS[0]}; chown -R $USER:$USER $PWD/work_dirs; sleep 120; exit \$ex;" \
    | tee /tmp/joboutput

jobid=$( grep -oP "(?<=id : )\w+-\w+"  /tmp/joboutput )

# ln -s logs/$JOBNAME $jobid.log

sco acp jobs stream-logs --workspace-name fvg-vc-research $jobid