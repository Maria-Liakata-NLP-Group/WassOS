import torch
import torch.nn as nn
from .networks.bridger import MLPBridger
from .networks.rnn_decoder import RNNDecoder
from .networks.loss_funcs import bag_of_word_loss

class BridgeRNN(nn.Module):
    def __init__(self, args, vocab, enc_hidden_dim, dec_hidden_dim, embed):
        super().__init__()
        self.bridger = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,

            encoder_dim=enc_hidden_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=dec_hidden_dim,
            decoder_layer=args.dec_num_layers,
        )

        self.decoder = RNNDecoder(
            vocab=len(vocab),
            max_len=args.src_max_time_step,
            input_size=args.dec_embed_dim,
            hidden_size=dec_hidden_dim,
            embed_droprate=args.dec_ed,
            rnn_droprate=args.dec_rd,
            n_layers=args.dec_num_layers,
            rnn_cell=args.rnn_type,
            use_attention=args.use_attention,
            embedding=embed,

            eos_id=vocab['S'].id,
            sos_id = vocab['E'].id

            # eos_id=vocab.src.eos_id,
            # sos_id=vocab.src.sos_id,
        )

    def forward(self, hidden, tgt_var):
        decode_init = self.bridger.forward(input_tensor=hidden)

        _loss = -torch.sum(self.decoder.score(
            inputs=tgt_var,
            encoder_outputs=None,
            encoder_hidden=decode_init,
        ))
        return _loss

    def predict(self, hidden):
        decode_init = self.bridger.forward(input_tensor=hidden)

        decoder_outputs, decoder_hidden, ret_dict, enc_states = self.decoder.forward(
            inputs=None,
            encoder_outputs=None,
            encoder_hidden=decode_init,
        )
        result = torch.stack(ret_dict['sequence']).squeeze()
        if result.dim() < 2:
            result = result.unsqueeze(1)
        return result

class BridgeMLP(nn.Module):
    def __init__(self, args, vocab, enc_dim, dec_hidden):
        super().__init__()
        self.bridger = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=enc_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=dec_hidden,
            decoder_layer=args.dec_num_layers,
        )
        if "stack_mlp" not in args:
            self.scorer = nn.Sequential(
                nn.Dropout(args.dec_rd),
                nn.Linear(
                    in_features=dec_hidden * args.dec_num_layers,
                    # out_features=len(vocab.src),
                    out_features=len(vocab),
                    bias=True
                ),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.scorer = nn.Sequential(
                nn.Dropout(args.dec_rd),
                nn.Linear(
                    in_features=dec_hidden * args.dec_num_layers,
                    out_features=dec_hidden,
                    bias=True
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=dec_hidden,
                    # out_features=len(vocab.src),
                    out_features=len(vocab),
                    bias=True
                ),
                nn.LogSoftmax(dim=-1)
            )

            # word_vocab[PAD].id
        # self.semantic_nll = nn.NLLLoss(ignore_index=vocab.src.pad_id)
        self.semantic_nll = nn.NLLLoss(ignore_index=vocab["<PAD>"].id)

    def forward(self, hidden, tgt_var):
        batch_size = tgt_var.size(0)
        semantic_decode_init = self.bridger.forward(input_tensor=hidden)
        xx_hidden = semantic_decode_init.permute(1, 0, 2).contiguous().view(batch_size, -1)
        score = self.scorer.forward(input=xx_hidden)
        sem_loss = bag_of_word_loss(score, tgt_var, self.semantic_nll)
        return sem_loss
