from transformer.Models import Decoder
from torch.nn import Module, Embedding
from transformers import EncoderDecoderModel
from collections import OrderedDict
import torch
import torch.nn as nn

def load_weights_sequential(target, source_state, extra_chan=1):
    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if not 'num_batches_tracked' in k1:
            if k1 in source_state:
                tar_v = source_state[k1]

                if v1.shape != tar_v.shape:
                    # Init the new segmentation channel with zeros
                    # print(v1.shape, tar_v.shape)
                    c, _, w, h = v1.shape
                    pads = torch.zeros((c, extra_chan, w, h), device=tar_v.device)
                    nn.init.orthogonal_(pads)
                    tar_v = torch.cat([tar_v, pads], 1)

                new_dict[k1] = tar_v

    target.load_state_dict(new_dict, strict=False)

def load_weights_sequential1(target, source_state, extra_chan=1):
    '''

    :param target:  my model, get parameter from original model
    :param source_state:  original model
    :param extra_chan:
    :return:
    '''

    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        # if not 'num_batches_tracked' in k1:
        #
        #     print(k1)

        split_name = k1.split('.')


        if len(split_name) ==2:
            if split_name[0] == 'layer_norm':
                judge_name = "decoder.cls.predictions.transform.LayerNorm." + split_name[1]

                if judge_name in source_state.keys():

                    tar_v = source_state[judge_name]

                    new_dict[k1] = tar_v


        if len(split_name) > 3:

            final_name = split_name[-2] + '.' + split_name[-1]
            middle_name = split_name[-3]

            if 'self_attn' == middle_name:

                judge_name = "decoder.bert.encoder.layer.6.attention.self."+ final_name
                layer_name = "decoder.bert.encoder.layer.6.attention.output." + final_name

                if judge_name in source_state.keys():

                    tar_v = source_state[judge_name]

                    new_dict[k1] = tar_v

                if layer_name in source_state.keys():

                    tar_v = source_state[layer_name]

                    new_dict[k1] = tar_v

            if middle_name == 'enc_attn' or middle_name == 'hidden_attn':

                judge_name = "decoder.bert.encoder.layer.6.crossattention.self." + final_name
                layer_name = "decoder.bert.encoder.layer.6.crossattention.output." + final_name

                if judge_name in source_state:

                    tar_v = source_state[judge_name]
                    new_dict[k1] = tar_v

                if layer_name in source_state:

                    tar_v = source_state[layer_name]

                    new_dict[k1] = tar_v

            if split_name[-2] == 'w_1':

                judge_name = "decoder.bert.encoder.layer.6.intermediate.dense." + split_name[-1]

                if judge_name in source_state:

                    tar_v = source_state[judge_name]
                    new_dict[k1] = tar_v

            if split_name[-2] == 'w_2':

                judge_name = "decoder.bert.encoder.layer.6.output.dense." + split_name[-1]

                if judge_name in source_state:

                    tar_v = source_state[judge_name]

                    new_dict[k1] = tar_v

            if split_name[-2] == 'layer_norm':

                judge_name = "decoder.bert.encoder.layer.6.output.LayerNorm." + split_name[-1]

                if judge_name in source_state:

                    tar_v = source_state[judge_name]

                    new_dict[k1] = tar_v

    target.load_state_dict(new_dict, strict=False)

def load_parameter(model, pretrained=True, extra_chan=0):


    # model = Decoder(embeddings= embedding, n_layers=1, n_head = 8, d_k=96, d_v=96,
    #         d_model=768, d_inner=3072, n_position=200, dropout=0.1,
    #         scale_emb=False)

    # model = TransformerDecoder(num_layers=1, d_model=768, heads=8, d_ff=3072, dropout=0.1 ,embeddings= embedding)
    original_model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

    if pretrained:
        load_weights_sequential1(model, original_model.state_dict(), extra_chan)
    return model

if __name__ == '__main__':

    embedding = Embedding(5,6)

    my_model = Decoder(embeddings=embedding, n_layers=1, n_head=8, d_k=96, d_v=96,
                    d_model=768, d_inner=3072, n_position=200, dropout=0.1,
                    scale_emb=False)

    # my_model = TransformerDecoder(num_layers=1, d_model=768, heads=8, d_ff=1, dropout=0.1 ,embeddings= embedding)
    original_model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")


    # test_model = pre_train()

    for name, param in my_model.named_parameters():

        print(name ,'             ', param.size())

        # split_name = name.split-tweet('.')
        # final_name = split_name[-3] + '.' + split_name[-1]


        # print(final_name ,'       ',name, '      ', param.size())
