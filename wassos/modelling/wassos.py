from mltoolkit.mlmo.layers import Ffnn, MuSigmaFfnn, Attention, OutEmbds
from mltoolkit.mlmo.layers.encoders import GruEncoder
from mltoolkit.mlmo.layers.decoders import PointerGenNetwork, GruPointerDecoder
from torch.nn import Module, Embedding, Tanh, Sequential, Softmax, \
    Sigmoid, Parameter, Linear, init, functional, ReLU, LayerNorm, Dropout
from mltoolkit.mlmo.utils.helpers.pytorch.computation import comp_seq_log_prob, \
    re_parameterize, kld_gauss, kld_normal, masked_softmax, wasserstein_gauss, init_wassers_weight, init_rev_weight
from collections import OrderedDict
import torch as T
from wassos.utils.helpers.modelling import group_att_over_input, group_att_over_input_for_decode
from mltoolkit.mlmo.utils.tools import DecState
from mltoolkit.mlmo.layers.wass import Wass, Rev,Weight, Linear_logit
from transformer.Models import Decoder
from pre_train_transformer import load_parameter
from wassos.utils.hparams import ModelHP, RunHP
from .tools import BridgeRNN, BridgeMLP

run_hp = RunHP()

EPS = 1e-12


class WassOS(Module):
    """
    CopyCat summarizer variational summarizer based on hierarchical latent
    representations of data. Specifically, it represents both review groups
    and reviews as separate latent code types.
    """

    def __init__(self, args,vocab, tag_vocab, max_rev_per_group, vocab_size, ext_vocab_size, ext_tag_size,tag_size, emb_dim, enc_hidden_dim,
                 c_dim, z_dim, att_hidden_dim, states_sc_hidden=150,
                 cgate_hidden_dim=None, word_padding_idx=0, dec_layers =1, dec_hidden_size= 200, heads =8,
                 ff_size=3072, strategy='T_center'):
        """
        :param vocab_size: the number of words by the generator.
        :param ext_vocab_size: the extended number of words that is accessible
            by the copy mechanism.
        :param emb_dim: the number of dimensions in word embeddings.
        :param enc_hidden_dim: GRU encoder's hidden dimension.
        :param c_dim: dimension of the group representation.
        :param z_dim: dimension of the review representation.
        :param att_hidden_dim: hidden dimension of hidden dimension of
            feed-forward networks.
        :param states_sc_hidden: hidden dimension of the score function used
            for production of the group representation c.
        :param cgate_hidden_dim: copy gate hidden dimension.
        """
        assert vocab_size <= ext_vocab_size

        super(WassOS, self).__init__()

        self.strategy = strategy

        self.max_rev_per_group = max_rev_per_group

        self.args = args

        self.z_dim = z_dim

        self._embds = Embedding(ext_vocab_size, emb_dim)

        self._embds_tag = Embedding(ext_tag_size, emb_dim)

        self._encoder = GruEncoder(input_dim=emb_dim, hidden_dim=enc_hidden_dim)

        self.enc_hidden_dim = enc_hidden_dim

        # POINTER-GENERATOR NETWORK #

        #   generation network - computes generation distribution over words
        dec_inp_dim = z_dim + enc_hidden_dim
        gen_network = Sequential()
        gen_network.add_module("lin_proj", Linear(dec_inp_dim, emb_dim))
        gen_network.add_module("out_embds", OutEmbds(self._embds, vocab_size))
        gen_network.add_module("softmax", Softmax(dim=-1))

        #   copy gate - computes probability of copying a word
        c_ffnn = Ffnn(z_dim + enc_hidden_dim + emb_dim,
                      hidden_dim=cgate_hidden_dim,
                      non_linearity=Tanh(), output_dim=1)
        copy_gate = Sequential()
        copy_gate.add_module("ffnn", c_ffnn)
        copy_gate.add_module("sigmoid", Sigmoid())

        pgn = PointerGenNetwork(gen=gen_network, copy_gate=copy_gate,
                                ext_vocab_size=ext_vocab_size)

        # COMPLETE DECODER (GRU + ATTENTION + COPY-GEN) #

        #   attention for decoder over encoder's hidden states
        dec_att = Attention(query_dim=z_dim, hidden_dim=att_hidden_dim,
                            value_dim=enc_hidden_dim,
                            non_linearity=Tanh())
        self._keys_creator = dec_att.create_keys

        self.tgt_embedding = Embedding(ext_vocab_size, 768, word_padding_idx)

        self.load_decoder = Decoder(self.tgt_embedding, dec_layers, heads, d_k=96, d_v=96,
                                    d_model=dec_hidden_size, d_inner=ff_size, n_position=200, dropout=0.1,
                                    scale_emb=False)
        self.transformer_decoder = load_parameter(self.load_decoder)
        # self.transformer_decoder = self.load_decoder

        self.logit = Linear_logit(int(z_dim/2), ext_vocab_size)

        self._decoder = GruPointerDecoder(input_dim=emb_dim + z_dim,
                                          hidden_dim=z_dim,
                                          contxt_dim=enc_hidden_dim,
                                          att_module=dec_att,
                                          pointer_gen_module=pgn,
                                          transformer_decoder=self.transformer_decoder,
                                          word_padding_idx=0,
                                          cat_contx_to_inp=True,
                                          pass_extra_feat_to_pg=False,
                                          logit_fun=self.logit)

        # NETWORKS THAT PRODUCES LATENT VARIABLES' MUS AND SIGMAS #

        # z inference network
        self._z_inf_network_sem = MuSigmaFfnn(
                                          # input_dim=enc_hidden_dim,
                                          input_dim=int(enc_hidden_dim / 2),
                                          non_linearity=Tanh(),
                                          output_dim=int(z_dim/2))
        self._z_inf_network_syn = MuSigmaFfnn(
                                          # input_dim=enc_hidden_dim,
                                          input_dim=int(enc_hidden_dim / 2),
                                          non_linearity=Tanh(),
                                          output_dim=int(z_dim / 2))


        # z prior network
        self._z_prior_network = MuSigmaFfnn(input_dim=c_dim, output_dim=z_dim)

        # c inference network
        #   scores each state to obtain group context vector
        if self.strategy == 'T_center':
            states_dim = enc_hidden_dim + emb_dim
        else:
            states_dim = enc_hidden_dim
        self._c_states_scoring = Ffnn(input_dim=states_dim,
                                      hidden_dim=states_sc_hidden,
                                      output_dim=1, non_linearity=Tanh())

        self._c_inf_network_sem = MuSigmaFfnn(input_dim=int(states_dim/2),
                                          output_dim=int(c_dim/2))


        self._c_inf_network_syn = MuSigmaFfnn(input_dim=int(states_dim / 2),
                                          output_dim=int(c_dim / 2))

        self.context_linear = Linear((enc_hidden_dim + emb_dim), states_dim)


        self.wassers_weight_sem = Wass(int(z_dim/2), 1)
        self.wassers_weight_syn = Wass(int(z_dim/2), 1)

        if self.strategy == 'T_center':
            dim = emb_dim + enc_hidden_dim
        else:
            dim = enc_hidden_dim
        self.rev_weight = Rev(dim, 1)

        # encode of multi-task
        self.task_enc_dim = int(enc_hidden_dim/2)
        self.task_dec_dim = int(enc_hidden_dim/2)
        self.args = args
        self.vocab = vocab
        self.tag_vocab = tag_vocab

        # multi-task loss =====================
        self.sup_syn = GruPointerDecoder(input_dim=emb_dim + int(z_dim/2),
                                          hidden_dim=int(z_dim/2),
                                          contxt_dim=enc_hidden_dim,
                                          att_module=dec_att,
                                          pointer_gen_module=pgn,
                                          transformer_decoder=self.transformer_decoder,
                                          word_padding_idx=0,
                                          cat_contx_to_inp=True,
                                          pass_extra_feat_to_pg=False,
                                          logit_fun=self.logit)

        self.sup_sem = BridgeMLP(
            args=self.args,
            vocab=self.vocab,
            enc_dim=self.task_enc_dim,
            dec_hidden=self.task_dec_dim
        )

        # z_syn predicts the sentence
        self.syn_infer = GruPointerDecoder(input_dim=emb_dim + int(z_dim/2),
                                          hidden_dim=int(z_dim/2),
                                          contxt_dim=enc_hidden_dim,
                                          att_module=dec_att,
                                          pointer_gen_module=pgn,
                                          transformer_decoder=self.transformer_decoder,
                                          word_padding_idx=0,
                                          cat_contx_to_inp=True,
                                          pass_extra_feat_to_pg=False,
                                          logit_fun=self.logit)

        # z_sem predicts the sentence
        self.sem_infer = GruPointerDecoder(input_dim=emb_dim + int(z_dim/2),
                                          hidden_dim=int(z_dim/2),
                                          contxt_dim=enc_hidden_dim,
                                          att_module=dec_att,
                                          pointer_gen_module=pgn,
                                          transformer_decoder=self.transformer_decoder,
                                          word_padding_idx=0,
                                          cat_contx_to_inp=True,
                                          pass_extra_feat_to_pg=False,
                                          logit_fun=self.logit)

        self.syn_adv = GruPointerDecoder(input_dim=emb_dim + int(z_dim/2),
                                          hidden_dim=int(z_dim/2),
                                          contxt_dim=enc_hidden_dim,
                                          att_module=dec_att,
                                          pointer_gen_module=pgn,
                                          transformer_decoder=self.transformer_decoder,
                                          word_padding_idx=0,
                                          cat_contx_to_inp=True,
                                          pass_extra_feat_to_pg=False,
                                          logit_fun=self.logit)
        # syn predicts bag of words
        self.sem_adv = BridgeMLP(
            args=self.args,
            vocab=self.vocab,
            enc_dim=self.task_enc_dim,
            dec_hidden=self.task_dec_dim
        )



    def forward(self, rev, rev_len,rev_tag, rev_mask,rev_tag_mask,
                group_rev_indxs, group_rev_indxs_mask,
                rev_to_group_indx, other_rev_indxs, other_rev_indxs_mask,
                other_rev_comp_states, other_rev_comp_states_mask,
                c_lambd=0., z_lambd=0.):
        """
        :param rev: review word ids.
            [batch_size, rev_seq_len]
        :param rev_len: review lengths.
            [batch_size]
        :param rev_mask: float mask where 0. is set to padded words.
            [batch_size, rev_seq_len]
        :param rev_to_group_indx: mapping from reviews to their corresponding
            groups.
            [batch_size]
        :param group_rev_indxs: indxs of reviews that belong to same groups.
            [group_count, max_rev_count]
        :param group_rev_indxs_mask: float mask where 0. is set to padded
            review indxs.
            [group_count, max_rev_count]
        :param other_rev_indxs: indxs of leave-one-out reviews.
            [batch_size, max_rev_count]
        :param other_rev_indxs_mask: float mask for leave-one-out reviews.
        :param other_rev_comp_states: indxs of (hidden) states of leave-one-out
            reviews. Used as an optimization to avoid attending over padded
            positions.
            [batch_size, cat_rev_len]
        :param other_rev_comp_states_mask: masking of states for leave-one-out
            reviews.
            [batch_size, cat_rev_len]
        :param c_lambd: annealing constant for c representations.
        :param z_lambd: annealing constant for z representations.

        :return loss: scalar loss corresponding to the mean ELBO over batches.
        :return metrs: additional statistics that are used for analytics and
            debugging.
        """
        bs = rev.size(0)
        device = rev.device
        group_count = group_rev_indxs.size(0)
        loss = 0.
        metrs = OrderedDict()

        # used for cross_entropy loss
        target = T.zeros_like(rev)
        target[:-1] = rev[1:]
        target[-1] = rev[0]
        # target = rev
        rev_length = rev.size()[1]
        target_list = list(T.chunk(target, rev_length, 1))

        # used for syntactic parse
        target_tag = T.zeros_like(rev_tag)
        target_tag[:-1] = rev_tag[1:]
        target_tag[-1] = rev_tag[0]
        rev_tag_length = rev_tag.size()[1]
        target_tag_list = list(T.chunk(target_tag, rev_tag_length, 1))



        rev_word_embds = self._embds(rev)
        rev_tag_embds = self._embds_tag(rev_tag)

        rev_encs, rev_hiddens = self.encode(rev_word_embds, rev_len)

        att_keys = self.create_att_keys(rev_hiddens)
        contxt_states = self.get_contxt_states(rev_hiddens, rev_word_embds)



        # running the c inference network for the whole group
        # calculate w1*c1 + w2*c2 + ....

        c_mu_q_sem, \
        c_sigma_q_sem, \
        c_mu_q_syn, \
        c_sigma_q_syn, \
        scor_wts = self.get_q_c_mu_sigma(contxt_states, rev_mask,
                                         group_rev_indxs, group_rev_indxs_mask)




        # running the z inference network for each review,
        # separate rev_enc into half
        rev_enc_hid = rev_encs.view(bs, 2, -1)
        rev_enc_sem = rev_enc_hid[:, 0, :].squeeze(1)
        rev_enc_syn = rev_enc_hid[:, 1, :].squeeze(1)

        z_mu_q_sem, z_sigma_q_sem, z_mu_q_syn, z_sigma_q_syn = self.get_q_z_mu_sigma(rev_enc_sem, rev_enc_syn)
        z_sem = re_parameterize(z_mu_q_sem, z_sigma_q_sem)
        z_syn = re_parameterize(z_mu_q_syn, z_sigma_q_sem)
        z = T.cat((z_syn, z_sem), dim=-1)



        # PERFORMING REVIEWS RECONSTRUCTION ================================================#

        rev_att_keys, \
        rev_att_vals, \
        rev_att_mask = group_att_over_input_for_decode(inp_att_keys=att_keys,
                                            inp_att_vals=rev_hiddens,
                                            inp_att_mask=rev_mask,
                                            att_indxs=other_rev_indxs,
                                            att_indxs_mask=other_rev_indxs_mask)



        rev_att_word_ids = rev[other_rev_indxs].view(bs, -1)

        # optimizing the attention targets by making more compact tensors
        # with less padded entries
        sel = T.arange(bs, device=device).unsqueeze(-1)
        rev_att_keys = rev_att_keys[sel, other_rev_comp_states]
        rev_att_vals = rev_att_vals[sel, other_rev_comp_states]
        rev_att_mask = other_rev_comp_states_mask
        rev_att_word_ids = rev_att_word_ids[sel, other_rev_comp_states]

        # creating an extra feature that is passe
        extra_feat = z.unsqueeze(1).repeat(1, rev_word_embds.size(1), 1)


        log_probs, rev_att_wts, \
        hidden, cont, \
        copy_probs = self._decode(embds=rev_word_embds, mask=rev_mask,
                                          extra_feat=extra_feat,
                                          review_word_id=rev,
                                          hidden=z, att_keys=rev_att_keys,
                                          att_values=rev_att_vals,
                                          att_mask=rev_att_mask,
                                          att_word_ids=rev_att_word_ids)

        rec_term = comp_seq_log_prob(log_probs[:, :-1], seqs=rev[:, 1:],
                                     seqs_mask=rev_mask[:, 1:])

        avg_rec_term = rec_term.mean(dim=0)

        # loss for E

        loss += -avg_rec_term


        syntax_hidden, semantic_hidden = self.latent_for_init(z_sem, z_syn)


        syn_loss, sem_loss = self.get_mul_loss(z_syn, semantic_hidden, target_tag_list, rev, rev_tag_embds, rev_tag_mask)
        mul_loss = self.args.mul_syn * syn_loss + self.args.mul_sem * sem_loss
        loss += mul_loss

        if self.training:
            with T.no_grad():
                dis_syn_loss, dis_sem_loss = self.get_dis_loss(z_syn,z_sem, target_list, rev_word_embds, rev_mask)
                syn_adv, sem_adv = self.get_adv(syntax_hidden, z_sem, target_tag_list, rev, rev_tag_embds, rev_tag_mask)
                if self.args.infer_weight > 0.:
                    adv_syn = self.args.adv_syn * syn_adv + self.args.infer_weight * self.args.inf_sem * dis_sem_loss
                    adv_sem = self.args.adv_sem * sem_adv + self.args.infer_weight * self.args.inf_syn * dis_syn_loss
                else:
                    adv_syn = self.args.adv_syn * syn_adv
                    adv_sem = self.args.adv_sem * sem_adv
        else:
            dis_syn_loss, dis_sem_loss = self.get_dis_loss(z_syn, z_sem, target_list, rev_word_embds, rev_mask)
            syn_adv, sem_adv = self.get_adv(syntax_hidden, z_sem, target_tag_list, rev, rev_tag_embds, rev_tag_mask)
            if self.args.infer_weight > 0.:
                adv_syn = self.args.adv_syn * syn_adv + self.args.infer_weight * self.args.inf_sem * dis_sem_loss
                adv_sem = self.args.adv_sem * sem_adv + self.args.infer_weight * self.args.inf_syn * dis_syn_loss
            else:
                adv_syn = self.args.adv_syn * syn_adv
                adv_sem = self.args.adv_sem * sem_adv

        loss += -(adv_syn + adv_sem)



        # WASSERSTEIN DISTANCE #

        # wassers_distance size [batch_size, max_rev_per_group]

        if self.strategy == 'T_center':
            wassers_distance_sem = wasserstein_gauss(z_mu_q_sem, z_sigma_q_sem, c_mu_q_sem, c_sigma_q_sem,
                                                     group_rev_indxs, self.max_rev_per_group)
            wassers_distance_syn = wasserstein_gauss(z_mu_q_syn, z_sigma_q_syn, c_mu_q_syn, c_sigma_q_syn, group_rev_indxs, self.max_rev_per_group)

            w_sem = self.wassers_weight_sem(z_sem, group_rev_indxs)
            w_syn = self.wassers_weight_syn(z_syn, group_rev_indxs)

            wassers_sem = wassers_distance_sem.unsqueeze(-1)
            inf_sem = T.sum(T.sum(w_sem * wassers_sem, dim=-1), dim=-1)
            loss_inf_sem = inf_sem.mean() * 2.0

            wassers_syn = wassers_distance_syn.unsqueeze(-1)
            inf_syn = T.sum(T.sum(w_syn * wassers_syn, dim=-1), dim=-1)
            loss_inf_syn = inf_syn.mean() * 2.0
            loss += (loss_inf_sem + loss_inf_syn) / 4.0
        else:
            wassers_distance_sem = wasserstein_gauss(z_mu_q_sem, z_sigma_q_sem, c_mu_q_sem, c_sigma_q_sem,
                                                     group_rev_indxs, self.max_rev_per_group)

            w_sem = self.wassers_weight_sem(z_sem, group_rev_indxs)

            wassers_sem = wassers_distance_sem.unsqueeze(-1)
            inf_sem = T.sum(T.sum(w_sem * wassers_sem, dim=-1), dim=-1)
            loss_inf_sem = inf_sem.mean() * 2.0
            loss += loss_inf_sem




        # KULLBACK-LEIBLER TERMS #


        # z kld
        z_kl_term_sem = kld_normal(z_mu_q_sem, z_sigma_q_sem)
        avg_z_kl_term_sem = z_kl_term_sem.mean(dim=0)
        kl_zem = z_lambd * avg_z_kl_term_sem * self.args.sem_weight

        z_kl_term_syn = kld_normal(z_mu_q_syn, z_sigma_q_syn)
        avg_z_kl_term_syn = z_kl_term_syn.mean(dim=0)
        kl_syn = z_lambd * avg_z_kl_term_syn * self.args.syn_weight
        kl_loss = (kl_syn + kl_zem)/(self.args.sem_weight + self.args.syn_weight)
        loss += kl_loss


        # log_var_p_z = T.log(z_sigma_p).sum(-1)

        rev_att_wts = rev_att_wts[:, :-1] * rev_mask[:, 1:].unsqueeze(
            -1)
        copy_probs = copy_probs[:, :-1] * rev_mask[:, 1:]

        # computing maximums over attention weights
        rev_att_max = rev_att_wts.max(-1)[0]

        # computing maximums over steps of copy probs
        max_copy_probs = copy_probs.max(-1)[0]

        # averaging over batches different statistics for logging
        avg_att_max = rev_att_max.mean()
        avg_max_copy_prob = max_copy_probs.mean(0)
        avg_copy_prob = (copy_probs.sum(-1) / rev_len.float()).mean(0)

        metrs['avg_rec'] = avg_rec_term.item()
        metrs['avg_z_kl_term_sem'] = avg_z_kl_term_sem.item()
        metrs['avg_z_kl_term_syn'] = avg_z_kl_term_syn.item()
        metrs['wass_sem'] = loss_inf_sem.item()

        metrs['loss'] = loss.item()
        metrs['mul_loss'] = mul_loss.item()

        metrs['avg_att_max'] = avg_att_max.item()
        metrs['avg_max_copy_prob'] = avg_max_copy_prob.item()
        metrs['avg_copy_prob'] = avg_copy_prob.item()

        metrs['c_lambd'] = c_lambd
        metrs['z_lambd'] = z_lambd

        return loss, metrs

    def get_p_z_mu_sigma(self, c):
        """Computes z prior's parameters (mu, sigma)."""
        mu_p, sigma_p = self._z_prior_network(c)
        return mu_p, sigma_p

    def get_q_z_mu_sigma(self, encs_sem, encs_syn):
        """
        Runs the inference network and computes mu and sigmas for review latent
        codes (z).
        """
        mu_q_sem, sigma_q_sem = self._z_inf_network_sem(encs_sem)
        mu_q_syn, sigma_q_syn = self._z_inf_network_syn(encs_syn)

        return mu_q_sem, sigma_q_sem, mu_q_syn, sigma_q_syn

    def get_q_c_mu_sigma(self, states, states_mask, group_indxs,
                         group_indxs_mask):
        """Computes c approximate posterior's parameters (mu, sigma).

        :param states: [batch_size, seq_len1, dim]
                        representations of review steps, e.g. hidden + embd.
        :param states_mask: [batch_size, seq_len1]
        :param group_indxs: [batch_size2, seq_len2]
                      indxs of reviews belonging to the same product
                      [batch_size2, seq_len2(max_rev_per_group)]
        :param group_indxs_mask: [batch_size2, seq_len2]
        """




        grouped_states, \
        grouped_mask = group_att_over_input(inp_att_vals=states,
                                            inp_att_mask=states_mask,
                                            att_indxs=group_indxs,
                                            att_indxs_mask=group_indxs_mask)


        ws_state, \
        score_weights = self._compute_ws_state(states=grouped_states,
                                               states_mask=grouped_mask)

        group = ws_state.size()[0]
        state_hid = ws_state.view(group, 2, -1)
        state_sem = state_hid[:, 0, :].squeeze(1)
        state_syn = state_hid[:, 1, :].squeeze(1)

        mu_sem, sigma_sem = self._c_inf_network_sem(state_sem)
        if self.strategy == 'T_center':
            mu_syn, sigma_syn = self._c_inf_network_syn(state_syn)
        else:
            mu_syn, sigma_syn = self._z_inf_network_syn(state_syn)

        return mu_sem, sigma_sem, mu_syn, sigma_syn, score_weights

    def encode(self, embds, mask_or_lens):
        return self._encoder(embds, mask_or_lens)

    def decode_beam(self, seqs, hidden, att_word_ids, init_z=None, **kwargs):
        """Function to be used in the beam search process.

        :param seqs: [batch_size, 1]
        :param hidden: [batch_size, hidden_dim]
        :param att_word_ids: [batch_size, cat_rev_len]
        :param init_z: [batch_size, z_dim]
        """
        embds = self._embds(seqs)
        mask = T.ones_like(seqs, dtype=T.float32)

        if init_z is None:
            init_z = hidden

        word_log_probs, att_wts, \
        hidden, cont, ptr_probs= self._decode(embds=embds, mask=mask,
                                               extra_feat=init_z.unsqueeze(1),
                                               hidden=hidden,
                                               att_word_ids=att_word_ids,
                                               # review_word_id=seqs,
                                               **kwargs)
        out = DecState(word_scores=word_log_probs,
                       rec_vals={"hidden": hidden, "cont": cont,
                                 "init_z": init_z},
                       coll_vals={'copy_probs': ptr_probs.squeeze(-1),
                                  'att_wts': att_wts.squeeze(1),
                                  "att_word_ids": att_word_ids})
        return out

    def _decode(self, embds, mask, hidden, cont=None, review_word_id=None, eps=EPS,
                **dec_kwargs):

        """Teacher forcing decoding of sequences.

        :param embds: [batch_size, seq_len, dim]
        :param mask: [batch_size, seq_len]
        :param att_word_ids: [batch_size, inp_seq_len]
        :param copy_prob: a fixed probability of copying words.
        :param hidden: [batch_size, hidden_dim]
        """
        word_probs, copy_probs, \
        att_weights, prev_hidden, \
        prev_cont= self._decoder(embds, mask, init_hidden=hidden,
                                          init_cont=cont, review_word_id=review_word_id, **dec_kwargs)
        #     self._decoder(embds, mask, init_hidden=hidden,
        #                           init_cont=cont, **dec_kwargs)



        word_log_probs = T.log(word_probs + eps)

        return word_log_probs, att_weights, prev_hidden, prev_cont, copy_probs

    def _compute_ws_state(self, states, states_mask):
        """Computes weighted state by scoring each state."""

        state_scores = self._c_states_scoring(states)
        grouped_state_weights = masked_softmax(state_scores, states_mask, dim=2)

        # compute the weight of word in each reviews, and put them together
        # the size of group_context is[batch_size, rev_num_per_group, dim]
        group_context = (states * grouped_state_weights).sum(dim=2)
        review_weight = self.rev_weight(group_context)
        goup_rev = T.sum(review_weight * group_context, dim=1)

        return goup_rev, grouped_state_weights


    def create_att_keys(self, hiddens):
        """
        Creates the attention keys that are used in the decoder.
        Performs projection that is required by the attention mechanism, allows
        for a speed-up by not performing this operation every time attention is
         called (at each decoding step).

        :param hiddens: [batch_size, seq_len, hidden_dim]
        """
        att_keys = self._keys_creator(hiddens)
        return att_keys

    def get_contxt_states(self, hiddens, embds):
        context = T.cat((hiddens, embds), dim=-1)
        new_context = self.context_linear(context)
        return new_context
        # return T.cat((hiddens, embds), dim=-1)

    def get_mul_loss(self, syntax_hidden, semantic_hidden, syn_tgt, sem_tgt, tag_emb, tag_mask, cont=None, **dec_kwargs):

        extra_feat = syntax_hidden.unsqueeze(1).repeat(1, tag_emb.size(1), 1)

        logits = self.sup_syn(tag_emb, tag_mask, init_hidden=syntax_hidden,
                                           init_cont=cont, extra_feat=extra_feat,mode='syn', **dec_kwargs)

        rev_tag_length = len(syn_tgt)
        syn_loss = 0

        for targ, logit in zip(syn_tgt, logits):
            # print(type(targ), type(logit), "type is what")
            targ = T.squeeze(targ)

            ptloss = T.nn.CrossEntropyLoss()
            sub_loss = ptloss(logit, targ)
            syn_loss += sub_loss
        syn_loss = T.div(syn_loss, rev_tag_length)

        sem_loss = self.sup_sem.forward(hidden=semantic_hidden, tgt_var=sem_tgt)

        return syn_loss, sem_loss

    # z_sem and z_syn  predict sentence
    def get_dis_loss(self, syntax_hidden, semantic_hidden, sem_tgt, sem_emb, sem_mask, cont=None, **dec_kwargs):
        extra_feat_sem = semantic_hidden.unsqueeze(1).repeat(1, sem_emb.size(1), 1)
        extra_feat_syn = syntax_hidden.unsqueeze(1).repeat(1, sem_emb.size(1), 1)

        logits_syn = self.syn_infer(sem_emb, sem_mask, init_hidden=syntax_hidden,
                                          init_cont=cont, extra_feat=extra_feat_syn,mode='sentence', **dec_kwargs)
        logits_sem = self.sem_infer(sem_emb, sem_mask, init_hidden=semantic_hidden,
                                           init_cont=cont, extra_feat=extra_feat_sem, mode='sentence', **dec_kwargs)


        rev_length = len(sem_tgt)
        dis_syn_inf = 0
        dis_sem_inf = 0

        for targ, logit in zip(sem_tgt, logits_syn):
            # print(type(targ), type(logit), "type is what")
            targ = T.squeeze(targ)

            ptloss = T.nn.CrossEntropyLoss()
            dis_syn_inf = ptloss(logit, targ)
            dis_syn_inf += dis_syn_inf
        dis_syn_loss = T.div(dis_syn_inf, rev_length)

        for targ, logit in zip(sem_tgt, logits_sem):
            # print(type(targ), type(logit), "type is what")
            targ = T.squeeze(targ)

            ptloss = T.nn.CrossEntropyLoss()
            dis_sem_inf = ptloss(logit, targ)
            dis_sem_inf += dis_sem_inf
        dis_sem_loss = T.div(dis_sem_inf, rev_length)

        return dis_syn_loss, dis_sem_loss

    def get_adv(self, syntax_hidden, semantic_hidden, syn_tgt, sem_tgt, tag_emb, tag_mask, cont=None, **dec_kwargs):

        extra_feat = semantic_hidden.unsqueeze(1).repeat(1, tag_emb.size(1), 1)
        logits = self.syn_adv(tag_emb, tag_mask, init_hidden=semantic_hidden,
                                          init_cont=cont, extra_feat=extra_feat, mode='syn', **dec_kwargs)

        rev_tag_length = len(syn_tgt)
        syn_adv = 0

        for targ, logit in zip(syn_tgt, logits):
            # print(type(targ), type(logit), "type is what")
            targ = T.squeeze(targ)

            ptloss = T.nn.CrossEntropyLoss()
            syn_adv = ptloss(logit, targ)
            syn_adv += syn_adv
        syn_adv = T.div(syn_adv, rev_tag_length)

        sem_adv = self.sem_adv.forward(hidden=syntax_hidden, tgt_var=sem_tgt)

        return syn_adv, sem_adv

    def latent_for_init(self, z_sem, z_syn):
        '''
        this method for the shape of z, because DSS-VAE needs it
        :param z_separate:
        :return:
        '''
        def reshape(xx_hidden):
            xx_hidden = xx_hidden.view(batch_size, 1, int(self.enc_hidden_dim / 2))
            xx_hidden = xx_hidden.permute(1, 0, 2)
            return xx_hidden

        # syntax_latent = z_syn
        semantic_latent = z_sem
        batch_size = semantic_latent.size(0)

        syntax_hidden = reshape(z_syn)
        semantic_hidden = reshape(z_sem)

        return syntax_hidden, semantic_hidden




