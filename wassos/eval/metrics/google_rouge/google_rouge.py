# eval_utils.py

"""
The original code was adapted from MeanSum.
Calculate and keep track of ROUGE statistics
"""

from collections import defaultdict
import logging
import os
import pdb
from copy import copy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

from wassos.eval.metrics.google_rouge.rouge_scorer import RougeScorer
from wassos.eval.metrics.google_rouge.utils import update_moving_avg


class GoogleRouge(object):
    def __init__(self, remove_stopwords=False, use_stemmer=True,
                 store_all=True):
        """

        Args:
            remove_stopwords: boolean (remove stop words before calculating rouge)
            use_stemmer: boolean (stem words before calculating rouge)
            store_all: boolean
                - whether to store the 4 rouge stats for every summary.csv. This could be used to plot the
                distribution of the stats instead of just looking at the average.
        """
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        self.use_stemmer = use_stemmer
        self.store_all = store_all

        # python implementation of ROUGE
        self.rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                        use_stemmer=use_stemmer)

        # Every time update_avg_rouge() is called, the rouges are calculated between a summary.csv and n_docs reviews.
        # Using those rouge scores, four statistics are computed: the mean, max, min, and std.
        # The following dictionaries are then updated using those statistics, i.e. they will be a mean of
        # each of those four stats.
        self._updates = 0
        self.avg_avg_rouges = self.get_rouge_defaultdict()
        self.avg_min_rouges = self.get_rouge_defaultdict()
        self.avg_max_rouges = self.get_rouge_defaultdict()
        self.avg_std_rouges = self.get_rouge_defaultdict()

        if self.store_all:
            self.avg_rouges = self.get_rouge_defaultdict(list)
            self.min_rouges = self.get_rouge_defaultdict(list)
            self.max_rouges = self.get_rouge_defaultdict(list)
            self.std_rouges = self.get_rouge_defaultdict(list)

    #########################################
    #
    # General Utils
    #
    #########################################
    def get_rouge_defaultdict(self, default_type=float):
        """
        Return dict of default dicts.
        """
        dict = {'rouge1': defaultdict(default_type),
                'rouge2': defaultdict(default_type),
                'rougeL': defaultdict(default_type)}
        return dict

    def get_avg_stats_dicts(self):
        return {'avg': self.avg_avg_rouges,
                'min': self.avg_min_rouges,
                'max': self.avg_max_rouges,
                'std': self.avg_std_rouges}

    def get_list_stats_dicts(self):
        return {'avg': self.avg_rouges,
                'min': self.min_rouges,
                'max': self.max_rouges,
                'std': self.std_rouges}

    def aggr(self, avg=True):
        """
        Sums all average ROUGEs and divides by the number of updates to compute
        the macro scores.

        :param avg: whether to use average ROUGE between a hypothesis and multiple
                    references. Otherwise, will use maximum scores.
        """
        assert self.store_all
        res = self.get_rouge_defaultdict()
        rouge_scores = self.avg_rouges if avg else self.max_rouges
        for rname in res.keys():
            for mname, mvals in rouge_scores[rname].items():
                res[rname][mname] = np.mean(mvals)
        return res

    def update_with_evaluator(self, evaluator):
        """
        Use another GoogleRouge object to update the self.* rouge dicts. This is used
        by best_review_baseline() in run_evaluations.

        Args:
            evaluator: GoogleRouge instance
        """
        self._updates += 1  # global count

        # Update moving averages
        for stat, rouge_dict in self.get_avg_stats_dicts().items():
            src_rouge_dict = getattr(evaluator, 'avg_{}_rouges'.format(stat))
            for rouge_name, d in src_rouge_dict.items():
                for metric, score in d.items():
                    cur_score = rouge_dict[rouge_name][metric]
                    rouge_dict[rouge_name][metric] = update_moving_avg(
                        cur_score, score, self._updates)

        # Add to lists
        for stat, rouge_dict in self.get_list_stats_dicts().items():
            src_rouge_dict = getattr(evaluator, '{}_rouges'.format(stat))
            for rouge_name, d in src_rouge_dict.items():
                for metric, scores in d.items():
                    rouge_dict[rouge_name][metric].extend(scores)

    #########################################
    #
    # Main methods
    #
    #########################################
    def calc_rouges(self, source, summary):
        """
        Wrapper around the rouge_scorer. Removes stop words potentially,

        Args:
            source: str
            summary: str

        Returns:
            dict: keys are strs, values are rouge objects with ('precision', 'recall', and 'fmeasure' fields)
        """
        if self.remove_stopwords:
            source = ' '.join(
                [w for w in word_tokenize(source) if w not in self.stopwords])
            summary = ' '.join(
                [w for w in word_tokenize(summary) if w not in self.stopwords])

        rouges = self.rouge_scorer.score(source, summary)

        return rouges

    def accum(self, hypotheses, references):
        """
        Args:
            hypotheses: list of strs
            references: list of lists of strs
        Returns: 4 (avg, min, max, std) rouge dicts for this batch
        """
        # Store average of the four statistics for this batch
        batch_avg_avg_rouges = self.get_rouge_defaultdict()
        batch_avg_min_rouges = self.get_rouge_defaultdict()
        batch_avg_max_rouges = self.get_rouge_defaultdict()
        batch_avg_std_rouges = self.get_rouge_defaultdict()

        for i, hyp in enumerate(hypotheses):
            refs = references[i]

            # Compute rouges between summary.csv and each reference
            rouges = self.get_rouge_defaultdict(list)
            for ref in refs:
                scores = self.calc_rouges(ref, hyp)
                for rouge_name, rouge_obj in scores.items():  # rouge_name = rouge1, rouge2, rougeL
                    for metric in ['precision', 'recall', 'fmeasure']:
                        score = getattr(rouge_obj, metric)
                        rouges[rouge_name][metric[0]].append(
                            score)  # [0] for first letter

            # Compute statistics and update batch and global averages
            avg_rouges = self.get_rouge_defaultdict()
            min_rouges = self.get_rouge_defaultdict()
            max_rouges = self.get_rouge_defaultdict()
            std_rouges = self.get_rouge_defaultdict()
            self._updates += 1  # global count
            for rouge_name, rouge_obj in rouges.items():
                for metric in ['precision', 'recall', 'fmeasure']:
                    scores = rouges[rouge_name][metric[0]]

                    avg, min, max, std = np.mean(scores), np.min(
                        scores), np.max(scores), np.std(scores)
                    avg_rouges[rouge_name][metric[0]] = avg
                    min_rouges[rouge_name][metric[0]] = min
                    max_rouges[rouge_name][metric[0]] = max
                    std_rouges[rouge_name][metric[0]] = std

                    # update batch averages
                    cur_avg_avg = batch_avg_avg_rouges[rouge_name][metric[0]]
                    cur_avg_min = batch_avg_min_rouges[rouge_name][metric[0]]
                    cur_avg_max = batch_avg_max_rouges[rouge_name][metric[0]]
                    cur_avg_std = batch_avg_std_rouges[rouge_name][metric[0]]
                    batch_avg_avg_rouges[rouge_name][
                        metric[0]] = update_moving_avg(cur_avg_avg, avg, i + 1)
                    batch_avg_min_rouges[rouge_name][
                        metric[0]] = update_moving_avg(cur_avg_min, min, i + 1)
                    batch_avg_max_rouges[rouge_name][
                        metric[0]] = update_moving_avg(cur_avg_max, max, i + 1)
                    batch_avg_std_rouges[rouge_name][
                        metric[0]] = update_moving_avg(cur_avg_std, std, i + 1)

                    # update global averages
                    cur_avg_avg = self.avg_avg_rouges[rouge_name][metric[0]]
                    cur_avg_min = self.avg_min_rouges[rouge_name][metric[0]]
                    cur_avg_max = self.avg_max_rouges[rouge_name][metric[0]]
                    cur_avg_std = self.avg_std_rouges[rouge_name][metric[0]]
                    self.avg_avg_rouges[rouge_name][
                        metric[0]] = update_moving_avg(cur_avg_avg, avg,
                                                       self._updates)
                    self.avg_min_rouges[rouge_name][
                        metric[0]] = update_moving_avg(cur_avg_min, min,
                                                       self._updates)
                    self.avg_max_rouges[rouge_name][
                        metric[0]] = update_moving_avg(cur_avg_max, max,
                                                       self._updates)
                    self.avg_std_rouges[rouge_name][
                        metric[0]] = update_moving_avg(cur_avg_std, std,
                                                       self._updates)

                    # Add to dictionary storing all stats
                    if self.store_all:
                        self.avg_rouges[rouge_name][metric[0]].append(avg)
                        self.min_rouges[rouge_name][metric[0]].append(min)
                        self.max_rouges[rouge_name][metric[0]].append(max)
                        self.std_rouges[rouge_name][metric[0]].append(std)

        return batch_avg_avg_rouges, batch_avg_min_rouges, batch_avg_max_rouges, batch_avg_std_rouges

    #########################################
    #
    # Data munging utils
    #
    #########################################
    def to_str(self, rouge_dict):
        """
        Convert dict of dicts of rouge scores to a readable string

        Example output:
        rouge1-f=0.1576, rouge1-p=0.1143, rouge1-r=0.1925, \
        rouge2-f=0.0000, rouge2-p=0.0000, rouge2-r=0.0000, \
        rougeL-f=0.0950, rougeL-p=0.0714, rougeL-r=0.1021
        """
        strs = []
        for rouge_name, d in sorted(rouge_dict.items()):
            for metric, score in sorted(d.items()):
                # strs.append('{}-{}={:.4f}'.format(rouge_name, metric, score))
                strs.append('%s-%s=%.4f' % (rouge_name, metric, score))
        str = ', '.join(strs)
        return str

    def to_csv(self, rouge_dict, out_fp):
        """
        rouge: dict of dicts
        out_fp: str

        Output:
            Rouge,  F, precision, recall
            rouge1
            rouge2
            rougeL
        """
        with open(out_fp, 'w') as f:
            f.write('Rouge,F,Precision,Recall\n')
            for rouge_name, scores in sorted(rouge_dict.items()):
                f.write(
                    '{},{},{},{}\n'.format(rouge_name, scores['f'], scores['p'],
                                           scores['r']))
