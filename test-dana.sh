#!/bin/bash

# python tools/test.py configs/nsfw/resnet50_lv1_8xb256-rsb-a1-600e.py \
#     work_dirs/resnet50_lv1_8xb256-rsb-a1-600e/epoch_100.pth \
#     --cfg-options \
#     test_dataloader.dataset.data_root=$HOME/涉政与色情低俗图包 \
#     test_dataloader.dataset.ann_file=$HOME/涉政与色情低俗图包/ann_lv1.txt \
#     visualizer.vis_backends= \
#     --out test-0827-resnet50_lv1_8xb256.pkl &

# python tools/test.py configs/nsfw/convnext-v2-tiny_32xb32_in1k-384px.py \
#     work_dirs/convnext-v2-tiny_32xb32_in1k-384px/epoch_100.pth \
#     --cfg-options test_dataloader.dataset.data_root=$HOME/涉政与色情低俗图包 \
#     test_dataloader.dataset.ann_file=$HOME/涉政与色情低俗图包/ann_sq.txt \
#     visualizer.vis_backends= \
#     --out test-0827-convnext-v2-tiny_32xb32_in1k-384px.pkl &

# python tools/test.py configs/nsfw/resnet50_8xb64_448-rsb-a3-100e_in1k.py \
#     work_dirs/resnet50_8xb64_448-rsb-a3-100e_in1k/epoch_100.pth \
#     --cfg-options test_dataloader.dataset.data_root=$HOME/涉政与色情低俗图包 \
#     test_dataloader.dataset.ann_file=$HOME/涉政与色情低俗图包/ann_sq.txt \
#     visualizer.vis_backends= \
#     --out test-0827-resnet50_8xb64_448-rsb-a3-100e_in1k.pkl &

# python tools/test.py configs/nsfw/resnet50-smooth_8xb32_in1k-448px.py \
#     work_dirs/resnet50-smooth_8xb32_in1k-448px/epoch_40.pth \
#     --cfg-options test_dataloader.dataset.data_root=$HOME/涉政与色情低俗图包 \
#     test_dataloader.dataset.ann_file=$HOME/涉政与色情低俗图包/ann_sq.txt \
#     visualizer.vis_backends= \
#     --out test-0827-resnet50-smooth_8xb32_in1k-448px.pkl &

# python tools/test.py configs/nsfw/vit-base-p16_64xb64_in1k-384px.py \
#     work_dirs/vit-base-p16_64xb64_in1k-384px/epoch_100.pth \
#     --cfg-options test_dataloader.dataset.data_root=$HOME/涉政与色情低俗图包 \
#     test_dataloader.dataset.ann_file=$HOME/涉政与色情低俗图包/ann_sq.txt \
#     visualizer.vis_backends= \
#     --out test-0827-vit-base-p16_64xb64_in1k-384px.pkl &

# python tools/test.py configs/nsfw/resnet50_8xb256-rsb-a3-100e_in1k.py \
#     work_dirs/resnet50_8xb256-rsb-a3-100e_in1k/epoch_100.pth \
#     --cfg-options test_dataloader.dataset.data_root=$HOME/涉政与色情低俗图包 \
#     test_dataloader.dataset.ann_file=$HOME/涉政与色情低俗图包/ann_sq.txt \
#     visualizer.vis_backends= \
#     --out test-0827-resnet50_8xb256-rsb-a3-100e_in1k.pkl &

# python tools/test.py configs/nsfw/convnext-tiny_32xb128_in1k.py \
#     work_dirs/convnext-tiny_32xb128_in1k/epoch_100.pth \
#     --cfg-options test_dataloader.dataset.data_root=$HOME/涉政与色情低俗图包 \
#     test_dataloader.dataset.ann_file=$HOME/涉政与色情低俗图包/ann_sq.txt \
#     visualizer.vis_backends= \
#     --out test-0827-convnext-tiny_32xb128_in1k.pkl &

# CONFIG=configs/huangfan/convnext-v2-tiny_8xb64_in1k-384px.py
# configs/huangfan/resnet50_8xb256-rsb-a3-100e_huangfan.py
# CONFIG=configs/huangfan/vit-base-p32_64xb64_in1k-384px.py
# CONFIG=configs/huangfan/resnet50_8xb256-rsb-a3-100e_huangfan.py
NAME=$(basename $CONFIG .py)
WORKDIR=work_dirs/$NAME

export PART=38
SEQ=`seq 10 10 100`


# for ep in $SEQ; do
#     tools/sco_test.sh $CONFIG \
#         $WORKDIR/epoch_$ep.pth \
#         --cfg-options test_dataloader.dataset.data_root=$HOME/涉政与色情低俗图包 \
#         test_dataloader.dataset.ann_file=$HOME/涉政与色情低俗图包/ann_dana.txt \
#         visualizer.vis_backends= \
#         --out $WORKDIR/test-dana-$NAME_$ep.pkl &
#     sleep 1
# done

# wait

# for ep in $SEQ; do
#     python tools/analysis_tools/confusion_matrix.py $CONFIG $WORKDIR/test-dana-$NAME_$ep.pkl --show-path $WORKDIR/dana-confmat/confmat_$ep.png --include-values &
# done


for ep in $SEQ; do
    tools/sco_test.sh $CONFIG \
        $WORKDIR/epoch_$ep.pth \
        --cfg-options visualizer.vis_backends= \
        --out $WORKDIR/test-val-$NAME_$ep.pkl &
    sleep 1
done

wait

for ep in $SEQ; do
    python tools/analysis_tools/confusion_matrix.py $CONFIG $WORKDIR/test-val-$NAME_$ep.pkl --show-path $WORKDIR/confmat/confmat_$ep.png --include-values &
done

# tools/sco_test.sh configs/huangfan/convnext-tiny_32xb128_in1k.py \
#     work_dirs/convnext-tiny_32xb128_in1k/epoch_100.pth \
#     --cfg-options test_dataloader.dataset.data_root=$HOME/涉政与色情低俗图包 \
#     test_dataloader.dataset.ann_file=$HOME/涉政与色情低俗图包/ann_sq.txt \
#     visualizer.vis_backends= \
#     --out test-0827-convnext-tiny_32xb128_in1k.pkl &
