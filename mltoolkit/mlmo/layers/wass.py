from torch.nn import Module, Sequential, Linear, Parameter, functional
from mltoolkit.mlutils.helpers.general import listify
import torch
from torch.nn import init

class Wass(Module):

    def __init__(self, emb_dim, out_score):
        super(Wass, self).__init__()

        # self.wassers_weight = Parameter(torch.Tensor(batch_size, max_rev_per_group))
            # Parameter(init.normal_(T.empty((batch_size, max_rev_per_group)), mean=0, std=1))

        # self.wassers_weight = Parameter(data=torch.Tensor(batch_size, max_rev_per_group), requires_grad=True)
        # self.wassers_weight.data.uniform_(0, 1)

        self.wass_weight = Linear(emb_dim, out_score)

        # self.weight.data.uniform_(-1, 1)

    def forward(self, x, group_index):

        # wass = init.normal_(self.wassers_weight, mean=0, std=1)

        wass = self.wass_weight(x)
        wass_index = wass[group_index]
        # print(wass_index.size(), wass_index,'wass_index.size()============')
        w_weight = functional.softmax(wass_index, dim=-2)
        # print(w_weight.size(), w_weight,"w_weight.size()======")
        return w_weight

class Rev(Module):

    '''
    这一组是review的权重
    '''

    def __init__(self, emb_dim, out_score):

        super(Rev, self).__init__()

        self.rev_weight = Linear(emb_dim, out_score)


    def forward(self, x):


        rev = self.rev_weight(x)
        # print(wass_index.size(), wass_index,'wass_index.size()============')
        w_weight = functional.softmax(rev, dim=-2)
        # print(w_weight.size(), w_weight,"w_weight.size()======")

        return w_weight

class Weight(Module):

    def __init__(self, emb_dim):

        super(Weight, self).__init__()
        self.weight = Parameter(data=torch.Tensor(emb_dim))


        # self.weight.data.uniform_(-1, 1)

    def forward(self):

       lam = torch.exp(init.uniform_(self.weight, a=-5, b=0))
       lam = functional.softmax(lam, dim=0)

       return lam

class Linear_logit(Module):

    def __init__(self, input_size, output_size):

        super(Linear_logit, self).__init__()
        self.matrix = Parameter(torch.randn(output_size, input_size))
        self.bias_term = Parameter(torch.randn(output_size))

    def forward(self, input_):

        return torch.matmul(input_, torch.transpose(self.matrix, dim0=0, dim1=1)) + self.bias_term
