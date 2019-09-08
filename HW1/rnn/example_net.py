import torch
import torch.nn.functional as F


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
class ExampleNet(torch.nn.Module):

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(ExampleNet, self).__init__()

        self.relu =torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.gru = torch.nn.GRU(input_size=dim_embeddings,
                                hidden_size=512,
                                num_layers=1,
                                bidirectional = False,
                                batch_first = True,
                                )
        self.lstm = torch.nn.LSTM(input_size=dim_embeddings,
                                hidden_size=512,
                                num_layers=1,
                                bidirectional = False,
                                batch_first = True,
                                )
        self.out = torch.nn.Linear(512, 512)
        self.out_op_1 = torch.nn.Linear(1024, 512)
        self.out_op_2 = torch.nn.Linear(512, 1)


    def forward(self, context, context_lens, options, option_lens):
        context, context_state = self.gru(context)
        context = self.out(context)
        context_out = self.relu(context[:, -1, :])

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option, option_state = self.gru(option)
            option = self.out(option)
            option_out = self.relu(option[:, -1, :])
            output_combine = torch.cat((context_out, option_out), 1)
            option_new = self.out_op_1(output_combine)
            option_new = self.relu(option_new)
            option_new = self.out_op_2(option_new)
            option_new = option_new.view(-1)
            logits.append(option_new)
        logits = torch.stack(logits, 1)

        return logits