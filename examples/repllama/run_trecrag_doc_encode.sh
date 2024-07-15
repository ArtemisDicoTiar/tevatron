#4개로 병렬화
#parallel=4
#device=0
#for ((i = $device; i <= 59; i += $parallel)); do
#  formatted_number=$(printf "%02d" $i)
#  sh run_trecrag_doc_encode.sh $device /workspace/trecrag/data/corpus/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_${formatted_number}.json
#done

set -e;


cuda=$1
document_path=$2
output_path=$3
file_name="${document_file%.json}"
document_dir=/workspace/trecrag/data/corpus/msmarco_v2.1_doc_segmented
index_dir=/workspace/trecrag/indexs/msmarco_v2.1_doc_segmented.repllama
echo "cuda: ${cuda}"
echo "document_file: ${document_file}"
echo "file_name: ${file_name}"

# encode documents
CUDA_VISIBLE_DEVICES=${cuda} python encode_trec_rag.py \
          --document_path ${document_dir}/${document_file} \
          --output_dir=temp \
          --model_name_or_path castorini/repllama-v1-7b-lora-passage \
          --tokenizer_name meta-llama/Llama-2-7b-hf \
          --fp16 \
          --per_device_eval_batch_size 16 \
          --p_max_len 512 \
          --dataset_name Tevatron/longeval-corpus:${dataset} \
          --encoded_save_path ${index_dir}/${file_name}.pkl \
          --encode_num_shard 1 \
          --encode_shard_index 0 \
          --per_device_eval_batch_size 16
