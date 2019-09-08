import torch
import torch.nn.functional as F
import json#instance
import numpy as np
import pandas as pd
class ExampleNet(torch.nn.Module):
    """

        Args:

     """
#################  RNN GRU  ############################
    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(ExampleNet, self).__init__()
        #self.dict = json.loads("/home/xiec/ADL/hw1/adl-hw1-example-code/shengyang_src/wordict.json")#instance
        self.context_rnn = torch.nn.GRU(input_size=dim_embeddings,
                                        num_layers=1,
                                        hidden_size=512,
                                        batch_first=True,
                                        bidirectional=True)

        self.options_rnn = torch.nn.GRU(input_size=dim_embeddings,
                                        num_layers=1,
                                        hidden_size=512,

                                        batch_first=True,
                                        bidirectional=True)

        self.gru_attn = torch.nn.GRU(input_size=1024,
                                     num_layers=1,
                                     hidden_size=512,
                                     batch_first=True,
                                     bidirectional=True)#dropout=0.2,


        self.relu =torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.cout = torch.nn.Linear(1024, 512)
        self.fina_out1 = torch.nn.Linear(1024, 512)
        self.ins_out2 = torch.nn.Linear(512, 256)
        self.fina_out2 = torch.nn.Linear(256, 1)
        # self.count = 0

    def forward(self, context, context_lens, options, option_lens):
        context, context_state = self.context_rnn(context)  #context Size([18, 218, 1024])
        context_out = self.cout(context)
        context_out = torch.mean(context_out, 1, True)

        batch_size = context.size(0)
        input_size = context.size(1)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option, option_state = self.options_rnn(option)#Size([18, 50, 1024])

            attn = torch.bmm(context, option.transpose(1, 2))#Size([18, 218, 50])
            attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)#Size([18, 50, 218])

            attn_weight_cpu = attn_weight_option.cpu()
            attn_weight_cpu = np.array(attn_weight_cpu.detach())
            print('attnwei',attn_weight_cpu.shape)

            # print('numpy_attn_weight', np.array(torch.tensor(attn_weight)))

            # print('print dic', self.dictionary)
            # print('attn_wei', attn_weight_cpu)
            # print('attn_wei_size', len(attn_weight_cpu))
            cont_var = [list(self.dictionary.keys())[index] for index in batch_context]
            # for index in batch_context:
            #     cont_var += list(self.dictionary.keys())[index]
            print('contvar',cont_var)

            opt_var = [ list(self.dictionary.keys())[index] for index in correct_ans]
            # for index in batch_option:
            #     opt_var += list(self.dictionary.keys())[index]
            print('optvar', opt_var)

            # df = pd.DataFrame(attn_weight_cpu[0], columns=cont_var, index=opt_var)    context option
            df = pd.DataFrame(attn_weight_cpu[0], columns=opt_var, index=opt_var)

            fig = plt.figure()

            ax = fig.add_subplot(111)

            cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
            fig.colorbar(cax)

            tick_spacing = 1
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

            ax.set_xticklabels([''] + list(df.columns))
            ax.set_yticklabels([''] + list(df.index))

            plt.show()

            attn_out = torch.bmm(attn, context)#attn torch.Size([32, 50, 269]) *  context torch.Size([32, 269, 1024]) =  Size([32, 50, 1024])
            # if self.count == 0:
            #     numpysave = attn_out.cpu().detach()
            #     np.savez('a.npz', numpysave)
            attn_out_combine = torch.mul(attn_out, option)
            # attn_out_combine = torch.cat(attn_out, option)
            final_context, final_state = self.gru_attn(attn_out_combine)##
            final_context = self.fina_out1(final_context)
            final_context = torch.mean(final_context, 1, True)
            final_context = torch.mul(context_out, final_context)
            # final_context = torch.cat((context_out, final_context),dim=2)

            final_context = self.relu(final_context)#[:,-1,:]
            final_context = self.ins_out2(final_context)
            final_context = self.relu(final_context)
            final_context = self.fina_out2(final_context)
            # self.count = self.count + 1


            logits.append(final_context.view(-1))
        logits = torch.stack(logits, 1)  # logits為list形式 把第1維壓縮疊加在一起，把7個（18,1）合在一起變成（18,
        return logits















