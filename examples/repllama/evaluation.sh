set -e

model_name=$1
model_path=$2
dataset=$3
cuda=$4

echo "==========================="
echo "model_name: $model_name"
echo "model_path: $model_path"
echo "dataset: $dataset"
echo "cuda: $cuda"
echo "==========================="

experiment_saving_path=beir_embedding_${dataset}_${model_name}

mkdir -p beir_embedding_${dataset}_${model_name}
doc_embedding=0
for s in 0 1 2 3;
do
  if ! [[ -f $experiment_saving_path/corpus_${dataset}.${s}.pkl ]]; then
  doc_embedding=1
  fi
done

if [[ "$doc_embedding" -eq "1" ]]; then
  echo "==========================="
  echo "Generating Document pickles"
  echo "==========================="
  for s in 0 1 2 3;
  do
  CUDA_VISIBLE_DEVICES=${cuda} python encode.py \
    --output_dir=temp \
    --model_name_or_path $model_path \
    --fp16 \
    --per_device_eval_batch_size 16 \
    --p_max_len 512 \
    --dataset_name Tevatron/beir-corpus:${dataset} \
    --encoded_save_path $experiment_saving_path/corpus_${dataset}.${s}.pkl \
    --encode_num_shard 4 \
    --encode_shard_index ${s}
  done
fi

if ! [[ -f $experiment_saving_path/queries_${dataset}.pkl ]]; then
  echo "==========================="
  echo "Generating Query pickles"
  echo "==========================="
  CUDA_VISIBLE_DEVICES=${cuda} python encode.py \
    --output_dir=temp \
    --model_name_or_path ${model_path} \
    --fp16 \
    --per_device_eval_batch_size 16 \
    --q_max_len 512 \
    --dataset_name Tevatron/beir:${dataset}/test \
    --encoded_save_path $experiment_saving_path/queries_${dataset}.pkl \
    --encode_is_qry
fi



if ! [[ -f $experiment_saving_path/rank.${dataset}.txt ]]; then
  echo "==========================="
  echo "FAISS Retrieving"
  echo "==========================="
  python -m tevatron.faiss_retriever \
    --query_reps "$experiment_saving_path/queries_${dataset}.pkl" \
    --passage_reps "$experiment_saving_path/corpus_${dataset}.*.pkl" \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to $experiment_saving_path/rank.${dataset}.txt
fi

if ! [[ -f $experiment_saving_path/rank.${dataset}.trec ]]; then
  echo "==========================="
  echo "Converting to TREC"
  echo "==========================="
  python -m tevatron.utils.format.convert_result_to_trec \
    --input $experiment_saving_path/rank.${dataset}.txt \
    --output $experiment_saving_path/rank.${dataset}.trec \
    --remove_query
fi

python -m pyserini.eval.trec_eval -c \
    -mrecall.100 \
    -mndcg_cut.10 \
    beir-v1.0.0-${dataset}-test \
    $experiment_saving_path/rank.${dataset}.trec

