from mltoolkit.mlmo.interfaces import ITorchModel
from wassos.utils.fields import ModelF
from logging import getLogger
from mltoolkit.mlmo.utils.tools import DecState
import torch as T
import os
from wassos.utils.helpers.modelling import group_att_over_input, group_att_over_input_for_decode
from mltoolkit.mlmo.utils.helpers.pytorch.computation import re_parameterize
import random

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class IWassOS(ITorchModel):
    """Model interface. Contains a custom method for summary generation."""

    def __init__(self, beamer, min_gen_seq_len=None, **kwargs):
        """
        :param beamer: beam search object for sequence generation.
        :param min_gen_seq_len: minimum length of generated reviews and summaries.
        """
        super(IWassOS, self).__init__(**kwargs)
        self.beamer = beamer
        self.min_sen_seq_len = min_gen_seq_len

    def predict(self, batch, **kwargs):
        """Predicts summaries and reviews."""

        self.model.eval()
        revs = batch[ModelF.REV].to(self.device)
        rev_lens = batch[ModelF.REV_LEN].to(self.device)
        revs_mask = batch[ModelF.REV_MASK].to(self.device)
        summ_rev_indxs = batch[ModelF.GROUP_REV_INDXS].to(self.device)
        summ_rev_indxs_mask = batch[ModelF.GROUP_REV_INDXS_MASK].to(self.device)
        other_revs = batch[ModelF.OTHER_REV_INDXS].to(self.device)
        other_revs_mask = batch[ModelF.OTHER_REV_INDXS_MASK].to(self.device)
        rev_to_group_indx = batch[ModelF.REV_TO_GROUP_INDX].to(self.device)

        bs = revs.size(0)
        max_rev_len = revs.size(1)
        summs_nr = summ_rev_indxs.size(0)

        with T.no_grad():
            rev_embds = self.model._embds(revs)
            rev_encs, rev_hiddens = self.model.encode(rev_embds, rev_lens)

            att_keys = self.model.create_att_keys(rev_hiddens)
            # contxt_states = rev_hiddens
            contxt_states = self.model.get_contxt_states(rev_hiddens, rev_embds)

            summ_att_keys, \
            summ_att_vals, \
            summ_att_mask = group_att_over_input_for_decode(inp_att_keys=att_keys,
                                                 inp_att_vals=rev_hiddens,
                                                 inp_att_mask=revs_mask,
                                                 att_indxs=summ_rev_indxs,
                                                 att_indxs_mask=summ_rev_indxs_mask)

            summ_att_word_ids = revs[summ_rev_indxs].view(summs_nr, -1)

            c_mu_q_sem, \
            c_sigma_q_sem, \
            c_mu_q_syn, \
            c_sigma_q_syn, \
            scor_wts= self.model.get_q_c_mu_sigma(contxt_states,
                                                               revs_mask,
                                                               summ_rev_indxs,
                                                               summ_rev_indxs_mask)


            # DECODING OF REVIEWS #

            rev_enc_hid = rev_encs.view(bs, 2, -1)
            rev_enc_sem = rev_enc_hid[:, 0, :].squeeze(1)
            rev_enc_syn = rev_enc_hid[:, 1, :].squeeze(1)

            z_mu_q_sem, z_sigma_q_sem, z_mu_q_syn, z_sigma_q_syn = self.model.get_q_z_mu_sigma(rev_enc_sem, rev_enc_syn)
            z_mu_q = T.cat([z_mu_q_syn, z_mu_q_sem], dim=-1)

            # creating attention values for the reviewer
            rev_att_keys, \
            rev_att_vals, \
            rev_att_vals_mask = group_att_over_input_for_decode(inp_att_keys=att_keys,
                                                     inp_att_vals=rev_hiddens,
                                                     inp_att_mask=revs_mask,
                                                     att_indxs=other_revs,
                                                     att_indxs_mask=other_revs_mask)

            rev_att_word_ids = revs[other_revs].view(bs, -1)
            if self.min_sen_seq_len is not None:
                min_lens = [self.min_sen_seq_len] * bs
            else:
                min_lens = None

            # z = re_parameterize(z_mu_q, z_sigma_q)

            init_rev_dec_state = DecState(rec_vals={"hidden": z_mu_q})
            # init_rev_dec_state = DecState(rec_vals={"hidden": z})

            rev_word_ids, \
            rev_coll_vals = self.beamer(init_rev_dec_state,
                                        min_lens=min_lens,
                                        max_steps=max_rev_len,
                                        att_keys=rev_att_keys,
                                        att_values=rev_att_vals,
                                        att_mask=rev_att_vals_mask,
                                        att_word_ids=rev_att_word_ids,
                                        minimum=1, **kwargs)

            # DECODING OF SUMMARIES #
            if self.min_sen_seq_len is not None:
                min_lens = [self.min_sen_seq_len] * summs_nr
            else:
                min_lens = None

            c_mu_q = T.cat((c_mu_q_syn, c_mu_q_sem), dim=-1)


            z_mu_p, z_sigma_p = self.model.get_p_z_mu_sigma(c_mu_q)

            init_summ_dec_state = DecState(rec_vals={"hidden": z_mu_p})
            summ_word_ids, \
            summ_coll_vals = self.beamer(init_summ_dec_state,
                                         min_lens=min_lens,
                                         max_steps=max_rev_len,
                                         att_keys=summ_att_keys,
                                         att_values=summ_att_vals,
                                         att_mask=summ_att_mask,
                                         att_word_ids=summ_att_word_ids,
                                         minimum=1, **kwargs)

            return rev_word_ids, rev_coll_vals, summ_word_ids, summ_coll_vals


