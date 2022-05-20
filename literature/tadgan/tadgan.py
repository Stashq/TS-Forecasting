import torch.nn as nn
import torch.nn.functional as F

from models import Encoder, Decoder


class CriticX(nn.Module):
    def __init__(
        self, x_size: int, h_size: int
    ):
        super(CriticX, self).__init__()
        self.dense1 = nn.Linear(in_features=x_size, out_features=h_size)
        self.dense2 = nn.Linear(in_features=h_size, out_features=1)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return (x)


class CriticZ(nn.Module):
    def __init__(self, z_size: int, h_size: int = 10):
        super(CriticZ, self).__init__()
        self.dense1 = nn.Linear(in_features=z_size, out_features=h_size)
        self.dense2 = nn.Linear(in_features=h_size, out_features=1)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return (x)


class TADGAN(nn.Module):
    def __init__(
        self, c_in: int, h_size: int,  n_layers: int, z_size: int,
        pretrained_encoder: nn.Module = None,
        pretrained_decoder: nn.Module = None
    ):
        self.z_size = z_size
        super(TADGAN, self).__init__()
        self.encoder = Encoder(
            x_size=c_in, h_size=h_size, n_layers=n_layers, emb_size=z_size)
        self.decoder = Decoder(
            z_size=z_size, h_size=h_size, n_layers=n_layers, x_size=c_in)
        self.critic_x = CriticX(x_size=c_in, h_size=20)
        self.critic_z = CriticZ(z_size=z_size, h_size=10)

        if pretrained_encoder is not None:
            self.encoder.load_state_dict(pretrained_encoder.state_dict())
        if pretrained_decoder is not None:
            self.decoder.load_state_dict(pretrained_decoder.state_dict())

    def forward(self, x):
        seq_len = x.size(1)
        z = self.encoder(x)
        x_hat = self.decoder(z, seq_len=seq_len)
        return x_hat
