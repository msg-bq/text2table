# 使用wrong_format和f_score两个分别计算

import argparse
from typing import List

import bert_score
import numpy as np
import tqdm
from sacrebleu import sentence_chrf

from table_utils import extract_table_by_name, parse_text_to_table, is_empty_table


bert_scorer = None
metric_cache = dict()  # cache some comparison operations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp')
    parser.add_argument('tgt')
    parser.add_argument('--row-header', default=False, action="store_true")  # 像是是否考虑header匹配与否
    parser.add_argument('--col-header', default=False, action="store_true")
    parser.add_argument('--metric', default='E', choices=['E', 'c', 'BS-scaled', ],
                        help="E: exact match\nc: chrf\nBS-scaled: re-scaled BERTScore")
    args = parser.parse_args()
    assert args.row_header or args.col_header
    print("Args", args)
    return args


def _parse_table_element_to_relation(table, i, j, row_header: bool, col_header: bool):
    assert row_header or col_header
    relation = []
    if row_header:
        assert j > 0
        relation.append(table[i][0])
    if col_header:
        assert i > 0
        relation.append(table[0][j])
    relation.append(table[i][j])
    return tuple(relation)


def _parse_table_to_data(table, row_header: bool, col_header: bool):  # ret: row_headers, col_headers, relation tuples
    if is_empty_table(table, row_header, col_header):
        return set(), set(), set()

    assert row_header or col_header
    row_headers = list(table[:, 0]) if row_header else []
    col_headers = list(table[0, :]) if col_header else []
    if row_header and col_headers:
        row_headers = row_headers[1:]
        col_headers = col_headers[1:]

    row, col = table.shape
    relations = []
    for i in range(1 if col_header else 0, row):
        for j in range(1 if row_header else 0, col):
            if table[i][j] != "":
                relations.append(_parse_table_element_to_relation(table, i, j, row_header, col_header))
    return set(row_headers), set(col_headers), set(relations)


def _calc_similarity_matrix(tgt_data, pred_data, metric):
    def calc_data_similarity(tgt, pred):
        if isinstance(tgt, tuple):
            ret = 1.0
            for tt, pp in zip(tgt, pred):
                ret *= calc_data_similarity(tt, pp)
            return ret

        if (tgt, pred) in metric_cache:
            return metric_cache[(tgt, pred)]

        if metric == 'E':
            ret = int(tgt == pred)
        elif metric == 'c':
            ret = sentence_chrf(pred, [tgt, ]).score / 100
        elif metric == 'BS-scaled':
            global bert_scorer
            if bert_scorer is None:
                bert_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
            ret = bert_scorer.score([pred, ], [tgt, ])[2].item()
            ret = max(ret, 0)
            ret = min(ret, 1)
        else:
            raise ValueError(f"Metric cannot be {metric}")

        metric_cache[(tgt, pred)] = ret
        return ret
    # if metric == 'BS-scaled':
    #     global bert_scorer
    #     if bert_scorer is None:
    #         bert_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
    #     matrix = [(tgt, pred) for tgt in tgt_data for pred in pred_data]
    #     ret = bert_scorer.score([i[0] for i in matrix], [i[1] for i in matrix])[2]
    #     return ret.reshape([len(pred_data), len(tgt_data)]).numpy()
    # else:
    return np.array([[calc_data_similarity(tgt, pred) for pred in pred_data] for tgt in tgt_data], dtype=float)
#  我下意识会认为这里是笛卡尔积，但实际上是等价于zip


def _metrics_by_sim(tgt_data, pred_data, metric):
    sim = _calc_similarity_matrix(tgt_data, pred_data, metric)  # (n_tgt, n_pred) matrix
    prec = np.mean(np.max(sim, axis=0))
    recall = np.mean(np.max(sim, axis=1))
    if prec + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1, sim.shape[0], sim.shape[1]


def _read_tables(path: str) -> List[str]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            line = extract_table_by_name(line)
            data.append(parse_text_to_table(line))

    return data


def calc_wrong_format(hyp_data, tgt_data):
    empty_tgt = 0
    wrong_format = 0
    for hyp_table, tgt_table in zip(hyp_data, tgt_data):
        if is_empty_table(tgt_table, args.row_header, args.col_header):
            empty_tgt += 1
        elif hyp_table is None:
            wrong_format += 1

    valid_tgt = len(hyp_data) - empty_tgt
    print("Wrong format: %d / %d (%.2f%%)" % (wrong_format, valid_tgt, wrong_format / valid_tgt * 100))


def calc_f_score(hyp_data, tgt_data):
    row_header_precision = []
    row_header_recall = []
    row_header_f1 = []
    row_header_n_tgt = []
    row_header_n_pred = []
    col_header_precision = []
    col_header_recall = []
    col_header_f1 = []
    col_header_n_tgt = []
    col_header_n_pred = []
    relation_precision = []
    relation_recall = []
    relation_f1 = []
    relation_n_tgt = []
    relation_n_pred = []

    for hyp_table, tgt_table in tqdm.tqdm(zip(hyp_data, tgt_data), total=len(hyp_data)):
        if is_empty_table(tgt_table, args.row_header, args.col_header):
            pass
        elif hyp_table is None or is_empty_table(hyp_table, args.row_header, args.col_header):  # 为什么hyp为空就会判错，
            # 不过可能不存在为空的tgt
            if args.row_header:  # 确实下面这些数组只在row_header为True的时候生效
                row_header_precision.append(0)
                row_header_recall.append(0)
                row_header_f1.append(0)
            if args.col_header:
                col_header_precision.append(0)
                col_header_recall.append(0)
                col_header_f1.append(0)
            relation_precision.append(0)
            relation_recall.append(0)
            relation_f1.append(0)
        else:
            hyp_row_headers, hyp_col_headers, hyp_relations = _parse_table_to_data(hyp_table, args.row_header,
                                                                                   args.col_header)
            """
                hyp: array([['', 'Losses', 'Total points', 'Points in 4th quarter', 'Wins'],
           ['Hawks', '12', '95', '', '46'],
           ['Magic', '41', '88', '21', '19']], dtype='<U21')
                其中row_headers = {'', 'Hawks', 'Magic'}
                col_header = {'Losses', 'Wins', 'Points in 4th quarter', 'Total points'}
                relations = {('Magic', 'Losses', '41'), ('Magic', 'Wins', '19'), ('Hawks', 'Total points', '95'), ('Hawks', 'Wins', '46'), ('Magic', 'Points in 4th quarter', '21'), ('Hawks', 'Losses', '12'), ('Magic', 'Total points', '88')}
                (如果col或row缺失，那里面就会是2维tuple。如果还缺估计就是1维了。总之是一个个cell叭)
                """
            tgt_row_headers, tgt_col_headers, tgt_relations = _parse_table_to_data(tgt_table, args.row_header,
                                                                                   args.col_header)
            if args.row_header:
                p, r, f, n_tgt, n_pred = _metrics_by_sim(tgt_row_headers, hyp_row_headers, args.metric)  # 基本就是一个个值比对
                #  不同的metric影响两个值的相似度计算方式，比如exact还是n-gram什么的。元组的相似度是元素分数的累乘
                #
                row_header_precision.append(p)
                row_header_recall.append(r)
                row_header_f1.append(f)
                row_header_n_tgt.append(n_tgt)
                row_header_n_pred.append(n_pred)
            if args.col_header:
                p, r, f, n_tgt, n_pred = _metrics_by_sim(tgt_col_headers, hyp_col_headers, args.metric)
                col_header_precision.append(p)
                col_header_recall.append(r)
                col_header_f1.append(f)
                col_header_n_tgt.append(n_tgt)
                col_header_n_pred.append(n_pred)
            if len(hyp_relations) == 0:
                relation_precision.append(0.0)
                relation_recall.append(0.0)
                relation_f1.append(0.0)
                relation_n_tgt.append(0)
                relation_n_pred.append(0)
            else:
                p, r, f, n_tgt, n_pred = _metrics_by_sim(tgt_relations, hyp_relations, args.metric)
                relation_precision.append(p)
                relation_recall.append(r)
                relation_f1.append(f)
                relation_n_tgt.append(n_tgt)
                relation_n_pred.append(n_pred)

# print('Macro-averaged results:')
    result_dict = {}
    if args.row_header:
        result_dict['row_header'] = row_header_f1
        print("Row header: precision = %.2f; recall = %.2f; f1 = %.2f" % (
            np.mean(row_header_precision) * 100, np.mean(row_header_recall) * 100, np.mean(row_header_f1) * 100))
    if args.col_header:
        result_dict['col_header'] = col_header_f1
        print("Col header: precision = %.2f; recall = %.2f; f1 = %.2f" % (
            np.mean(col_header_precision) * 100, np.mean(col_header_recall) * 100, np.mean(col_header_f1) * 100))
    result_dict['relation'] = relation_f1
    print("Non-header cell: precision = %.2f; recall = %.2f; f1 = %.2f" % (
        np.mean(relation_precision) * 100, np.mean(relation_recall) * 100, np.mean(relation_f1) * 100))
    np.save(f'{args.hyp}.{args.metric}', result_dict)
    # print('Micro-averaged results:')
    # if args.row_header:
    #     p = np.average(row_header_precision, weights=row_header_n_pred)
    #     r = np.average(row_header_recall, weights=row_header_n_tgt)
    #     f = 2 * p * r / (p + r)
    #     print("Row header: precision = %.2f; recall = %.2f; f1 = %.2f" % (p * 100, r * 100, f * 100))
    # if args.col_header:
    #     p = np.average(col_header_precision, weights=col_header_n_pred)
    #     r = np.average(col_header_recall, weights=col_header_n_tgt)
    #     f = 2 * p * r / (p + r)
    #     print("Col header: precision = %.2f; recall = %.2f; f1 = %.2f" % (p * 100, r * 100, f * 100))
    # p = np.average(relation_precision, weights=relation_n_pred)
    # r = np.average(relation_recall, weights=relation_n_tgt)
    # f = 2 * p * r / (p + r)
    # print("Non-header cell: precision = %.2f; recall = %.2f; f1 = %.2f" % (p * 100, r * 100, f * 100))


if __name__ == '__main__':
    args = parse_args()
    hyp_data_list = _read_tables(args.hyp)
    tgt_data_list = _read_tables(args.tgt)
    calc_wrong_format(hyp_data_list, tgt_data_list)
    calc_f_score(hyp_data_list, tgt_data_list)