#!/usr/bin/env bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/.." || exit 1
DATASET_ROOT=/mnt/a/cs_ai_wys/Marui/Data4DG
DATASET=domainnet
NET=resnet50_clip
DATASET_YAML=domainnet_source_free

ND=0
BATCH=128
I2_EPOCHS=100
DEVICE=0

DOMAINS=(clipart infograph painting quickdraw real sketch)
DATASET_NAME=DomainNet_SF


for SEED in 1 2 3
do
    OUT_DIR=output/${DATASET}/${NET}/PS_re_train_style/seed${SEED}

    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python train.py \
    --root ${DATASET_ROOT} \
    --seed ${SEED} \
    --trainer GeoStylerTrainer \
    --source-domains none\
    --target-domains "${DOMAINS[@]}" \
    --dataset-config-file configs/datasets/dg/${DATASET_YAML}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir ${OUT_DIR} \
    --arcface_s 5\
    --arcface_m 0.5\
    --num_styles 80\
    --txts_path dassl/txts\
    MODEL.BACKBONE.NAME ${NET} \
    DATALOADER.TRAIN_X.SAMPLER RandomSampler \
    DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
    OPTIM.MAX_EPOCH ${I2_EPOCHS} \
    DATASET.NAME ${DATASET_NAME} \
    OPTIM.LR 0.002 \
    DATASET.ROOT ${DATASET_ROOT} \
    TRAINER.GEOSTYLER.WEIGHT_DIR_PATH \
    "${OUT_DIR}"/checkpoint \
    TRAINER.GEOSTYLER.TRAIN_STYLE True


    I2_EPOCHS=50
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python train.py \
    --root ${DATASET_ROOT} \
    --seed ${SEED} \
    --trainer WOPA_clip \
    --source-domains none\
    --target-domains "${DOMAINS[@]}" \
    --dataset-config-file configs/datasets/dg/${DATASET_YAML}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/${DATASET}/${NET}/PS_re_train_cls/seed${SEED} \
    --arcface_s 5\
    --arcface_m 0.5\
    --num_styles 80\
    --txts_path dassl/txts\
    MODEL.BACKBONE.NAME ${NET} \
    DATALOADER.TRAIN_X.SAMPLER RandomSampler \
    DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
    OPTIM.MAX_EPOCH ${I2_EPOCHS} \
    DATASET.NAME ${DATASET_NAME} \
    OPTIM.LR 0.005 \
    DATASET.ROOT ${DATASET_ROOT} \
    STYLE_GENERATOR.NAME "GeoStylerGenerator" \
    TRAINER.GEOSTYLER.WEIGHT_DIR_PATH \
    "${OUT_DIR}"/checkpoint

done
