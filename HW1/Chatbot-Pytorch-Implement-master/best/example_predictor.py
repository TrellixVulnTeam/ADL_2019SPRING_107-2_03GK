import torch
from base_predictor import BasePredictor
from   import ExampleNet


class ExamplePredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self, embedding,
                 dropout_rate=0.3, loss='BCELoss', margin=0, threshold=None,
                 similarity='inner_product', **kwargs):
        super(ExamplePredictor, self).__init__(**kwargs)
        self.model = ExampleNet(embedding.size(1), similarity=similarity)# num_layers=2, hidden_size=128, num_directions=2, dropout_rate= 0.2
        self.embedding = torch.nn.Embedding(embedding.size(0),
                                            embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)

        # use cuda
        self.model = self.model.to(self.device)
        #self.model.cuda()
        self.embedding = self.embedding.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.8)

        self.loss = {
            'BCELoss': torch.nn.BCEWithLogitsLoss()
        }[loss]
        # self.loss = {
        #     'MSELoss': torch.nn.MSELoss(reduction='mean')
        # }[loss]

    def _run_iter(self, batch, training):
        with torch.no_grad():
            # print(0))
            # print(batch['context'].size())#options[32,10,50] context[32,300]
            context = self.embedding(batch['context'].to(self.device))
            options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        loss = self.loss(logits, batch['labels'].float().to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        return logits
