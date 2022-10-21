# contains computational functions that are used for modelling
import torch as T
from torch.autograd import Variable
from torch import Tensor
from torch.nn.functional import softmax
import numpy as np
from torch.nn import  Parameter, init, functional


def init_wassers_weight(batch_size, max_rev_per_group):
    # [batch_size, max_rev_per_group]
    w = T.empty((batch_size, max_rev_per_group))
    w = init.uniform_(w, a=0, b=1)
    # w = init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    w = Parameter(w, requires_grad=True)
    w = functional.softmax(w, dim=-1)
    return w

def init_rev_weight(batch_size, rev_num_per_group, dim):

    # initialize the weight of each review in a group
    w_rev = T.empty((batch_size, rev_num_per_group, dim))
    w_rev = init.kaiming_normal_(w_rev, mode='fan_out', nonlinearity='relu')
    w_rev = Parameter(w_rev, requires_grad=True)
    # w = functional.softmax(w, dim=-1)
    return w_rev

def masked_softmax(scores, mask=None, dim=-1):
    """
    Normalizes scores using masked softmax operation.

    :param scores: [batch_size, seq_len, *, 1]
    :param mask: [batch_size, seq_len, *]
    :param dim: over which dimension to perform normalization.
    :return: normalized scores
    """
    scores[mask == 0.] = np.float('-inf')
    probs = softmax(scores, dim=dim)
    return probs


def kld_gauss(mu_q, sigma_q, mu_p, sigma_p, dim=1, eps=0.):
    """
    Computes the Kullback-Leibler divergence between two Gaussian distributions.
    Namely, KL[N(z; mu_q, sigma_q) || N(mu_p, sigma_p)]. It's assumed that both
    sigmas are diagonal matrices.

    :param mu_q: [batch_size, *]
    :param sigma_q: [batch_size, *]
    :param mu_p: [batch_size, *]
    :param sigma_p: [batch_size, *]
    :param eps: constant to avoid log of zero.
    :return [batch_size, *]
    """
    d = mu_q.shape[dim]
    sigma_p_inv = (sigma_p + eps) ** -1
    tra = (sigma_p_inv * sigma_q).sum(dim=dim)
    quadr = T.sum(sigma_p_inv * ((mu_p - mu_q) ** 2), dim=dim)
    log_det_p = T.sum(T.log(sigma_p + eps), dim=dim)
    log_det_q = T.sum(T.log(sigma_q + eps), dim=dim)
    log_det = log_det_p - log_det_q
    return 0.5 * (tra + quadr - d + log_det)

def wasserstein_gauss(mu_r, sigma_r, mu_s, sigma_s, group_index, max_rev_per_group, dim= -1 ):
    """
    compute the wasserstein distance between two Gaussian dis
    It's assumed that both sigmas are diagonal matrices.
    they are the dis of review and the dis of summary

    :param mu_r: [batch_size, *]
    :param sigma_r: [batch_size, *]
    :param mu_s: [1, *]
    :param sigma_s: [1,*]
    :param dim:
    :param group_index: review index that belong to a group
    :return: [batch_size, *]
    """

    # get the reviews that from the same group

    group_mu_r = mu_r[group_index]
    group_sigma_r = sigma_r[group_index]


    mu_s = T.unsqueeze(mu_s, dim = 1)
    mu_s = mu_s.repeat(1,max_rev_per_group,1)
    # mu_s = mu_s.repeat(1, 8, 1)

    sigma_s = T.unsqueeze(sigma_s, dim = 1)
    sigma_s = sigma_s.repeat(1,max_rev_per_group,1)
    # sigma_s = sigma_s.repeat(1, 8, 1)


    mu = T.sum((group_mu_r - mu_s) ** 2, dim=dim)
    # The trace of a matrix (sigma)
    tra = T.sum(group_sigma_r + sigma_s - 2 * (group_sigma_r * sigma_s).sqrt(), dim=dim)


    return mu + tra




def kld_normal(mu, sigma, dim=1):
    """KLD from a normal to a Gaussian assuming both have diag. covariances."""
    return -0.5 * T.sum(1 + T.log(sigma) - mu.pow(2) - sigma, dim=dim)


def comp_seq_log_prob(log_probs, seqs, seqs_mask):
    """
    Computes the summed log probabilities over seq_lens.

    :param log_probs: [batch_size, seq_len, vocab_size]
    :param seqs: [batch_size, seq_len, vocab_size]
    :param seqs_mask: [batch_size, seq_len]
    :return: word_scores [batch_size, seq_len]
    """
    bs, sl = log_probs.shape[:2]
    # selecting log probs for specific (true) tokens
    log_probs = log_probs[T.arange(bs, dtype=T.int64).reshape((-1, 1)),
                          T.arange(sl, dtype=T.int64),
                          seqs]
    log_probs = (log_probs * seqs_mask).sum(dim=1)
    return log_probs


def re_parameterize(mu, sigma):
    """Performs the re-parametrization based sampling from a Gaussian distr."""
    std = sigma ** 0.5
    eps = T.randn_like(std)
    return mu + eps * std


def entropy(q, dim=-1, eps=1e-6):
    """
    Computes entropy of categorical distribution.

    :param q: [batch_size, len]
    :param eps: self-explanatory.
    """
    log_q = T.log(q + eps)
    entr = - T.sum(q * log_q, dim=dim)
    return entr


def compute_kl_approx(q_word_log_probs, p_word_log_probs, mask):
    """
    Computes an approximate KLD over summaries, where history is sampled
    but the actual support is summed over. Distributions are assumed to be
    categorical.

    :param q_word_log_probs: [summs_nr, seq_len, vocab_size]
                             log probabilities over all words (except first <S>)
                             assigned by the summarizer (q) distribution.
    :param p_word_log_probs [summs_nr, seq_len, vocab_size]
                            log probabilities assigned by the prior
    :param mask: [summs_nr, seq_len]
    """
    q_word_probs = q_word_log_probs.exp()

    kld = q_word_probs * (q_word_log_probs - p_word_log_probs)

    kld = kld.sum(-1)  # summing over vocabulary
    kld = (kld * mask).sum(-1)  # summing over sequences

    return kld


def sample_dropout_mask(shape, device, dropout_prob, training=True):
    if training:
        mask = T.empty(shape, device=device).bernoulli(1. - dropout_prob)
    else:
        mask = T.full(shape, 1. - dropout_prob, device=device)
    return mask


def get_curr_budget(len_budget, curr_step):
    """
    Returns current budget (floats) a model has to finish generation of seqs.

    :param curr_step: scalar or [seq_len] scalars
    :param len_budget: tensor [batch_size] defining how many
                       tokens a model should generate.
    """
    if isinstance(curr_step, Tensor):
        assert len(curr_step.shape) == 1
        curr_step = curr_step.unsqueeze(0)
        len_budget = len_budget.unsqueeze(-1)
    curr_budget = len_budget - curr_step
    curr_budget = curr_budget.float()
    return curr_budget


def get_summ_budget(revs_mask, summ_rev_indxs, summ_rev_indxs_mask):
    """
    Returns the summary length budget which is the average of reviews
    associated with it. Budget is in integers.
    """
    total_len = (revs_mask[summ_rev_indxs].sum(-1) * summ_rev_indxs_mask).sum(
        -1)
    summ_budget = (total_len / summ_rev_indxs_mask.sum(-1)).long()
    return summ_budget


def normalize_att_mask(att_mask, inp_word_hots):
    """Computes a normalized mask by dividing it by frequency of soft words."""
    masked_inp_hots = inp_word_hots * att_mask.unsqueeze(-1)
    # summing over seq_len to count occurrences of words
    soft_weights = masked_inp_hots.sum(dim=1)
    soft_weights[soft_weights > 0] = 1. / soft_weights[soft_weights > 0]
    new_att_mask = (masked_inp_hots * soft_weights.unsqueeze(1)).sum(-1)
    return new_att_mask
