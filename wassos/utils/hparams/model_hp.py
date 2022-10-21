from mltoolkit.mlmo.utils.tools import BaseHP


class ModelHP(BaseHP):
    """
    Contains hyper-parameters of the actual model.
    Please see `CopyCat` for more details on hyper-parameters.
    """

    def __init__(self):
        super(ModelHP, self).__init__()

        self.vocab_size = 50000
        self.ext_vocab_size = 80000
        self.ext_tag_size = 50
        self.tag_size = 50
        self.emb_dim = 256
        self.enc_hidden_dim = 768
        self.c_dim = 768
        self.z_dim = 768
        self.states_sc_hidden = 512
        self.att_hidden_dim = 256
        self.cgate_hidden_dim = 128
        self.dec_layers = 1
        self.dec_hidden_size = 768
        self.heads = 8
        self.ff_size = 3072
        self.strategy = 'T_center'

