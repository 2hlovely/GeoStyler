#!/usr/bin/env bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/.." || exit 1

DATASET_ROOT=/mnt/a/cs_ai_wys/Marui/Data4DG
DATASET=pacs
NET=resnet50_clip
DATASET_YAML=pacs_source_free
DEVICE=0
ND=0
BATCH=128

DYNA_TOPK=3

DOMAINS=(art_painting cartoon photo sketch)
DATASET_NAME=PACS_SF

for SEED in 1 2 3; do
  OUT_DIR=output/${DATASET}/${NET}/PS_re_train_style/seed${SEED}


  I2_EPOCHS=50
  CUDA_VISIBLE_DEVICES=$DEVICE \
    python train.py \
    --root ${DATASET_ROOT} \
    --seed ${SEED} \
    --eval-only \
    --trainer WOPA_clip \
    --model-dir outputTTA/${DATASET}/${NET}/PS_re_train_cls/seed${SEED} \
    --source-domains none --target-domains "${DOMAINS[@]}" \
    --dataset-config-file configs/datasets/dg/${DATASET_YAML}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir outputTTA/${DATASET}/${NET}/PS_re_eval_cls_tta/seed${SEED} \
    --arcface_s 5 --arcface_m 0.5 --num_styles 80 --txts_path dassl/txts MODEL.BACKBONE.NAME ${NET} \
    DATALOADER.TRAIN_X.SAMPLER RandomSampler \
    DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
    OPTIM.MAX_EPOCH ${I2_EPOCHS} \
    DATASET.NAME ${DATASET_NAME} \
    OPTIM.LR 0.005 \
    DATASET.ROOT ${DATASET_ROOT} \
    STYLE_GENERATOR.NAME "GeoStylerGenerator" \
    TRAINER.GEOSTYLER.WEIGHT_DIR_PATH \
    "${OUT_DIR}"/checkpoint \
    DYNA_STYLER.TOPK_STYLES ${DYNA_TOPK}

done
