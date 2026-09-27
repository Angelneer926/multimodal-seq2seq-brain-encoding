import torch
import torch.nn as nn
    

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.embedding(pos)
class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.max_len = max_len
        self.rel_embedding = nn.Embedding(2 * max_len - 1, dim)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.size()
        device = x.device
        pos = torch.arange(T, device=device)
        rel_pos = pos[None, :] - pos[:, None]  # (T, T)
        rel_pos = rel_pos.clamp(-self.max_len + 1, self.max_len - 1)
        rel_pos += self.max_len - 1

        rel_emb = self.rel_embedding(rel_pos)  # (T, T, D)
        rel_sum = rel_emb.sum(dim=1)  # (T, D)
        rel_sum = rel_sum.unsqueeze(0).expand(B, -1, -1)  # (B, T, D)
        return x + rel_sum

    
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nhead, num_parcels, num_subjects=4, max_len=1024, use_learnable_bos=True, predict_residual=False, activation="relu",   
             dropout=0.1):
        super().__init__()
        self.num_subjects = num_subjects
        self.num_parcels = num_parcels
        self.use_learnable_bos = use_learnable_bos
        self.predict_residual = predict_residual
        self.activation = activation
        self.dropout = dropout

        self.encoder_input_proj = nn.Linear(input_dim, hidden_dim)
        self.decoder_input_proj = nn.Linear(num_parcels, hidden_dim)

        self.encoder_pos_enc = RelativePositionalEncoding(hidden_dim, max_len)
        self.decoder_pos_enc = RelativePositionalEncoding(hidden_dim, max_len)

        if use_learnable_bos:
            self.learnable_bos = nn.Parameter(torch.zeros(1, 1, num_parcels))
        else:
            self.register_buffer("bos_token", torch.zeros(1, 1, num_parcels))

        self.subject_embeddings = nn.Embedding(num_subjects, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=self.dropout,
    activation=self.activation),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=self.dropout,
    activation=self.activation),
            num_layers=num_layers
        )

        self.subject_output_layers_mse = nn.ModuleList([
            nn.Linear(hidden_dim, num_parcels) for _ in range(num_subjects)
        ])
        self.subject_output_layers_pearson = nn.ModuleList([
            nn.Linear(hidden_dim, num_parcels) for _ in range(num_subjects)
        ])

    def forward_autoregressive(self, src, tgt, subject_ids, teacher_forcing_ratio=1.0):
        if src.ndim == 4:
            B, T, W, D = src.shape
            src = src.view(B, T, W * D)

        B, T_tgt, _ = tgt.shape
        device = tgt.device

        src_proj = self.encoder_input_proj(src)
        src_proj = self.encoder_pos_enc(src_proj)
        src_mask = nn.Transformer.generate_square_subsequent_mask(src_proj.size(1)).to(device)
        memory = self.encoder(src_proj.transpose(0, 1), mask=src_mask)

        outputs_mse = []
        outputs_pearson = []
        projected_inputs = []

        if self.use_learnable_bos:
            input_token = self.learnable_bos.expand(B, -1, -1)
        else:
            input_token = self.bos_token.expand(B, -1, -1)

        input_token = self.decoder_input_proj(input_token)
        subject_embed = self.subject_embeddings(subject_ids).unsqueeze(1)
        input_token = input_token + subject_embed
        projected_inputs.append(input_token)

        for t in range(T_tgt):
            tgt_seq = torch.cat(projected_inputs, dim=1)
            tgt_proj = self.decoder_pos_enc(tgt_seq).transpose(0, 1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_proj.size(0)).to(device)
            out = self.decoder(tgt_proj, memory, tgt_mask=tgt_mask)
            dec_output = out[-1] 

            mse_pred = torch.stack([
                self.subject_output_layers_mse[subject_ids[i]](dec_output[i])
                for i in range(B)
            ], dim=0).unsqueeze(1)

            pearson_pred = torch.stack([
                self.subject_output_layers_pearson[subject_ids[i]](dec_output[i])
                for i in range(B)
            ], dim=0).unsqueeze(1)

            outputs_mse.append(mse_pred)
            outputs_pearson.append(pearson_pred)

            if self.training and torch.rand(1).item() < teacher_forcing_ratio:
                next_token = self.decoder_input_proj(tgt[:, t:t+1, :])
            else:
                next_token = self.decoder_input_proj(mse_pred.detach())

            next_token = next_token + subject_embed
            projected_inputs.append(next_token)

        return torch.cat(outputs_mse, dim=1), torch.cat(outputs_pearson, dim=1)

    def generate(self, src, tgt_len, subject_ids):
        if src.ndim == 4:
            B, T, W, D = src.shape
            src = src.view(B, T, W * D)
        else:
            B = src.size(0)

        src_proj = self.encoder_input_proj(src)
        src_proj = self.encoder_pos_enc(src_proj)
        src_mask = nn.Transformer.generate_square_subsequent_mask(src_proj.size(1)).to(src.device)
        memory = self.encoder(src_proj.transpose(0, 1), mask=src_mask)

        input_token = self.learnable_bos.expand(B, -1, -1).to(src.device) if self.use_learnable_bos else self.bos_token.expand(B, -1, -1).to(src.device)
        input_token = self.decoder_input_proj(input_token)
        subject_embed = self.subject_embeddings(subject_ids).unsqueeze(1)
        input_token = input_token + subject_embed

        projected_inputs = [input_token]
        outputs = []

        for _ in range(tgt_len):
            tgt_seq = torch.cat(projected_inputs, dim=1)
            tgt_proj = self.decoder_pos_enc(tgt_seq).transpose(0, 1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_proj.size(0)).to(src.device)
            out = self.decoder(tgt_proj, memory, tgt_mask=tgt_mask)
            dec_output = out[-1]                                

            pred = torch.stack([
                self.subject_output_layers_mse[subject_ids[i]](dec_output[i])
                for i in range(B)
            ], dim=0).unsqueeze(1)

            outputs.append(pred)

            input_token = self.decoder_input_proj(pred)
            input_token = input_token + subject_embed
            projected_inputs.append(input_token)

        return torch.cat(outputs, dim=1)

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

