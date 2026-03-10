import torch
import torch.nn as nn

class SISAEnsemble(nn.Module):

    def __init__(self, shard_models):
        super(SISAEnsemble, self).__init__()

        self.shard_models = nn.ModuleList(shard_models)

    def forward(self, x):

        outputs_sum = None

        for model in self.shard_models:

            outputs = model(x)

            if outputs_sum is None:
                outputs_sum = outputs
            else:
                outputs_sum += outputs

        outputs_avg = outputs_sum / len(self.shard_models)

        return outputs_avg