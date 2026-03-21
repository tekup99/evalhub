#!/bin/bash

# Config dosyasını yükle
CONFIG_FILE="scripts/configs/config_v1.env"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config dosyasi bulunamadi: $CONFIG_FILE"
    exit 1
fi
source "$CONFIG_FILE"

# Config içindeki her bir model için ayrı bir Slurm işi başlat
for MODEL in $MODELS; do
    echo "Slurm isine gönderiliyor: $MODEL"
    # Modeli TARGET_MODEL ortam değişkeni olarak Slurm betiğine aktarıyoruz
    sbatch --export=ALL,TARGET_MODEL="$MODEL" scripts/evaluation/pass_k/my_pass_k.sh
done