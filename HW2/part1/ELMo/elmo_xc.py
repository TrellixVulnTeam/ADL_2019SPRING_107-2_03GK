import torch
import numpy as np
import torch.nn.AdaptiveLogSoftmaxWithLoss
from .char_token import ConvTokenEmbedder
from .classify_layer import SampledSoftmaxLayer
class Elmo_Net(torch.nn.Module):
    """

    Args:

    """
    def __init__(self,  word_emb_layer, char_emb_layer, n_class, use_cuda=False):
        super(Elmo_Net, self).__init__()
        self.token_embedder = ConvTokenEmbedder(word_emb_layer, char_emb_layer, use_cuda)
        self.classify_layer = SampledSoftmaxLayer(self.output_dim, n_class, n_samples = 8192, use_cuda)
        self.forward_lstm1 = torch.nn.GRU(
            input_size=1024,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
        )

        self.forward_lstm2 = torch.nn.GRU(
            input_size=512,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
        )

        self.backfard_lstm1 = torch.nn.GRU(
            input_size=1024,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
        )

        self.backfard_lstm2 = torch.nn.GRU(
            input_size=512,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
        )


        self.forward_middle = torch.nn.Linear(1024, 512)
        self.forward_last = torch.nn.Linear(1024, 130000)

        self.backfard_middle = torch.nn.Linear(1024, 512)
        self.backfard_last = torch.nn.Linear(1024, 130000)

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def forward(self, word_inp, chars_inp, mask_package):

        self.classify_layer.update_negative_samples(word_inp, chars_inp, mask_package[0])
        self.classify_layer.update_embedding_matrix()

        # x = CharEmbedding(sentence)#x is (batch_size, sentence_len, projection_size) (x, 64 ,1024)
        xf = self.forward_lstm1(x)
        xf = self.forward_middle(xf)
        xf = self.forward_lstm2(xf)
        xf = self.forward_last(xf)

        xb = self.backfard_lstm1(x)
        xb = self.backfard_middle(xb)
        xb = self.backfard_lstm2(xb)
        xb = self.backfard_last(xb)
