#4개로 병렬화
#parallel=4
#device=0
#for ((i = $device; i <= 59; i += $parallel)); do
#  formatted_number=$(printf "%02d" $i)
#  sh run_trecrag_doc_encode.sh $device 16 \
#  /workspace/trecrag/data/corpus/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_${formatted_number}.json \
#  /workspace/trecrag/data/corpus/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_${formatted_number}.pkl
#done

set -e;


cuda=$1
batch_size=$2
document_path=$3
output_path=$4
echo "cuda: ${cuda}"
echo "document_path: ${document_path}"
echo "output_path: ${output_path}"

# encode documents
CUDA_VISIBLE_DEVICES=${cuda} python encode_trec_rag.py \
          --document_path ${document_path} \
          --output_dir=temp \
          --model_name_or_path castorini/repllama-v1-7b-lora-passage \
          --tokenizer_name meta-llama/Llama-2-7b-hf \
          --fp16 \
          --p_max_len 512 \
          --dataset_name Tevatron/longeval-corpus:${dataset} \
          --encoded_save_path ${output_path} \
          --encode_num_shard 1 \
          --encode_shard_index 0 \
          --per_device_eval_batch_size ${batch_size:-16}
