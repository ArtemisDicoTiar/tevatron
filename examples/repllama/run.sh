set -e;

dataset=$1;
cuda=$2

mkdir -p beir_embedding_${dataset}

# encode documents
for s in 0 1 2 3;
do
    if [ -f ./beir_embedding_${dataset}/corpus_${dataset}.${s}.pkl ]; then
        echo "document pkl exists"
    else  
        CUDA_VISIBLE_DEVICES=${cuda} python encode.py \
          --output_dir=temp \
          --model_name_or_path castorini/repllama-v1-7b-lora-passage \
          --tokenizer_name meta-llama/Llama-2-7b-hf \
          --fp16 \
          --per_device_eval_batch_size 16 \
          --p_max_len 512 \
          --dataset_name Tevatron/beir-corpus:${dataset} \
          --encoded_save_path beir_embedding_${dataset}/corpus_${dataset}.${s}.pkl \
          --encode_num_shard 4 \
          --encode_shard_index ${s}
    fi
done


# encode queries
if [ -f ./beir_embedding_${dataset}/queries_${dataset}.pkl ]; then
    echo "query pkl exists"
else
    CUDA_VISIBLE_DEVICES=${cuda} python encode.py \
      --output_dir=temp \
      --model_name_or_path castorini/repllama-v1-7b-lora-passage \
      --tokenizer_name meta-llama/Llama-2-7b-hf \
      --fp16 \
      --per_device_eval_batch_size 16 \
      --q_max_len 512 \
      --dataset_name Tevatron/beir:${dataset}/test \
      --encoded_save_path beir_embedding_${dataset}/queries_${dataset}.pkl \
      --encode_is_qry
fi


# search / indexing
if [ -f ./beir_embedding_${dataset}/rank.${dataset}.txt ]; then
    echo "ranking result exists"
else
    echo "ranking result missing"
    if [ -f ./beir_embedding_${dataset}/index.faiss ]; then
        echo "faiss index exists"
    else
        python -m tevatron.faiss_retriever \
            --query_reps beir_embedding_${dataset}/queries_${dataset}.pkl \
            --passage_reps "beir_embedding_${dataset}/corpus_${dataset}.*.pkl" \
            --depth 100 \
            --batch_size 64 \
            --save_text \
            --save_ranking_to beir_embedding_${dataset}/rank.${dataset}.txt \
            --faiss beir_embedding_${dataset}/index.faiss
    fi
fi



# convert to TREC format
python -m tevatron.utils.format.convert_result_to_trec \
    --input beir_embedding_${dataset}/rank.${dataset}.txt \
    --output beir_embedding_${dataset}/rank.${dataset}.trec \
    --remove_query

# Evaluate
python -m pyserini.eval.trec_eval -c \
    -mrecall.100 \
    -mndcg_cut.10 \
    beir-v1.0.0-${dataset}-test \
    beir_embedding_${dataset}/rank.${dataset}.trec
