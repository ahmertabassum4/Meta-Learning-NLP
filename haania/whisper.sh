#!/bin/bash
#SBATCH --job-name=whisper_eval
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/haania.siddiqui/Meta-Learning-NLP/haania/runs/slurm-%j.out
#SBATCH --error=/home/haania.siddiqui/Meta-Learning-NLP/haania/runs/slurm-%j.err

set -eo pipefail
set -x

source /apps/local/anaconda3/conda_init.sh
conda activate meta

cd /home/haania.siddiqui/Meta-Learning-NLP || exit 1
mkdir -p /home/haania.siddiqui/Meta-Learning-NLP/haania/runs

which python3
python3 --version
echo "${CONDA_DEFAULT_ENV:-NOT_SET}"
hostname
pwd
nvidia-smi

python3 haania/evaluate_hin_eng_whisper.py \
  --model_path /home/haania.siddiqui/Meta-Learning-NLP/haania/runs/whisper_csfleurs_hin_eng_30 \
  --lang_pair hin-eng \
  --csfleurs_subset xtts_test1 \
  --real_wav_root /home/haania.siddiqui/OpenSLR104/English_Hindi_test/test \
  --real_wav_scp /home/haania.siddiqui/OpenSLR104/English_Hindi_test/test/transcripts/wav.scp \
  --real_segments /home/haania.siddiqui/OpenSLR104/English_Hindi_test/test/transcripts/segments \
  --real_text /home/haania.siddiqui/OpenSLR104/English_Hindi_test/test/transcripts/text \
  --device cuda \
  --save_json /home/haania.siddiqui/Meta-Learning-NLP/haania/runs/whisper_csfleurs_hin_eng_30/eval_results.json

python3 haania/evaluate_ben_eng_whisper.py \
  --model_path /home/haania.siddiqui/Meta-Learning-NLP/haania/runs/whisper_repo_ben_eng_medium \
  --repo_test_dir /home/haania.siddiqui/Meta-Learning-NLP/data/cs_fleurs_ben_eng_300_49/test \
  --real_wav_root /home/haania.siddiqui/Bengali/test \
  --real_wav_scp /home/haania.siddiqui/Bengali/test/transcripts/wav.scp \
  --real_segments /home/haania.siddiqui/Bengali/test/transcripts/segments \
  --real_text /home/haania.siddiqui/Bengali/test/transcripts/text \
  --device cuda \
  --save_json /home/haania.siddiqui/Meta-Learning-NLP/haania/runs/whisper_repo_ben_eng_medium/eval_results.json