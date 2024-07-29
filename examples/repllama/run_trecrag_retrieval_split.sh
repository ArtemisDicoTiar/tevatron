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
query_pkl=$3
document_pkls=$4
output_dir=$5
echo "cuda: ${cuda}"
echo "query_path: ${query_pkl}"
echo "document_path: ${document_pkls}"
echo "output_dir: ${output_dir}"

# encode queries
for document_pkl in ${document_pkls}; do
  echo "Processing ${document_pkl}"
  echo "Saving to ${output_dir}/$(basename ${document_pkl})_ranking.txt"
  mkdir -p ${output_dir}
  python -m tevatron.faiss_retriever \
            --query_reps ${query_pkl} \
            --passage_reps "${document_pkl}" \
            --depth 1000 \
            --batch_size ${batch_size} \
            --save_text \
            --save_ranking_to ${output_dir}/$(basename ${document_pkl})_ranking.txt
done
