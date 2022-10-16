import torch
from utils import tensor_op


class DetectorHead(torch.nn.Module):
    def __init__(self, input_channel, cell_size):
        super(DetectorHead, self).__init__()
        self.cell_size = cell_size
        ##
        self.act = torch.nn.ReLU(inplace=True)
        self.dense = torch.nn.Linear(input_channel, pow(cell_size, 2)+1)
        self.norm = torch.nn.BatchNorm2d(pow(cell_size, 2)+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = None
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)
        x = self.dense(x)
        x = x.permute(0, 3, 1, 2)
        out = self.norm(x)

        prob = self.softmax(out)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        # Reshape to get full resolution heatmap.
        prob = tensor_op.pixel_shuffle(prob, self.cell_size)  # [B,1,H*8,W*8]
        prob = prob.squeeze(dim=1)#[B,H,W]

        return {'logits':out, 'prob':prob}