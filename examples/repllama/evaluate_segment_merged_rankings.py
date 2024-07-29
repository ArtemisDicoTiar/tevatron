import argparse
import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import pandas as pd
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the SHARDED ranking results')
    parser.add_argument('--input_file', type=str, required=True, help='The input file containing the ranking results')
    parser.add_argument('--qrels_file', type=str, required=True, help='The qrels file to evaluate the ranking results')
    parser.add_argument('--output_file', type=str, required=True, help='The output file to store the evaluation results')

    args = parser.parse_args()

    input_file = Path(args.input_file)
    assert input_file.suffix == '.jsonl', 'The input file must be a jsonl file'
    output_file_path = Path(args.output_file)
    output_total_file_path = output_file_path.with_name(output_file_path.name + ".total")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    qrels_file = Path(args.qrels_file)

    rankings = defaultdict(dict)
    with input_file.open('r') as lines:
        for line in lines:
            line_dict = json.loads(line)
            qid = line_dict['query']['qid']
            qid = str(qid)
            candidates = line_dict['candidates']
            for candidate in candidates:
                did = candidate['docid']
                did = str(did)
                score = candidate['dist']
                score = float(score)
                rankings[qid][did] = score

    print(f'Loading qrels from {qrels_file}')
    qrels = defaultdict(dict)
    with qrels_file.open('r') as f:
        for line in f:
            qid, _, did, rel = line.strip().split()
            qid = str(qid)
            did = str(did)
            rel = int(rel)
            qrels[qid][did] = int(rel)

    qrels_segmented = defaultdict(dict)
    for qid in rankings:
        qid_rankings = rankings[qid]
        for did in qid_rankings:
            # did: msmarco_v2.1_doc_00_1545808368#1_2707234808
            non_segmented_did = did.split("#")[0]
            qrels_segmented[qid][did] = qrels[qid].get(non_segmented_did, 0)

    eval_ks = [1, 3, 5, 10, 100, 1000]
    evaluator = EvaluateRetrieval()

    with output_file_path.open('w') as f:
        for qid in tqdm(qrels.keys(), desc='Evaluating each query'):
            qid_qrels = qrels_segmented[qid]
            qid_rankings = rankings[qid]
            results = evaluator.evaluate({qid: qid_qrels}, {qid: qid_rankings}, eval_ks)

            results = [{"qid": qid}] + list(results)
            f.write(json.dumps(results, indent=4))
            f.write('\n')

    total_results = evaluator.evaluate(qrels_segmented, rankings, eval_ks)
    with output_total_file_path.open('w') as f:
        for total_result in total_results:
            f.write(json.dumps(total_result, indent=4))
            f.write('\n')

    pprint(total_results)
