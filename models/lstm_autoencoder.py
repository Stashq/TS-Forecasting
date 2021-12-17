from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        h_size: int,
        n_layers: int
    ):
        super().__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.h_size = h_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, x):
        embedding, (_, _) = self.lstm(x)
        return embedding


class Decoder(nn.Module):
    def __init__(
        self,
        emb_size: int,
        h_size: int,
        n_layers: int
    ):
        super().__init__()
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.h_size = h_size
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, emb):
        x_tilda, (_, _) = self.lstm(emb)
        return x_tilda


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int
    ):
        super().__init__()
        self.c_in = c_in
        self.n_layers = n_layers
        self.h_size = h_size
        self.encoder = Encoder(c_in, h_size, n_layers)
        self.decoder = Encoder(h_size, c_in, n_layers)

    def forward(self, x):
        emb = self.encoder(x)
        x_tilda = self.decoder(emb)
        return x_tilda
