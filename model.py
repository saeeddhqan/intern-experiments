
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import math, random
from typing import Optional, Union, Iterable
import numpy as np
from torch.distributions import Categorical

seed = 1244
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

class LayerNorm(nn.LayerNorm):
	def forward(self, x: Tensor) -> Tensor:
		return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
	def forward(self, x: Tensor) -> Tensor:
		return F.linear(
			x,
			self.weight.to(x.dtype),
			None if self.bias is None else self.bias.to(x.dtype),
		)


class Conv1d(nn.Conv1d):
	def _conv_forward(
		self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
	) -> Tensor:
		return super()._conv_forward(
			x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
		)


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-5):
		super().__init__()
		self.eps = eps


	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x):
		return self._norm(x.float()).type_as(x)


class MultiHeadAttention(nn.Module):
	def __init__(self, n_state: int, n_head: int, causality: str):
		super().__init__()
		self.n_head = n_head
		self.query = Linear(n_state, n_state)
		self.key = Linear(n_state, n_state, bias=False)
		self.value = Linear(n_state, n_state)
		self.out = Linear(n_state, n_state)
		self.causality = causality
		if causality == 'semi-causal':
			# for 30sec
			# self.nblocks = 15 # 15 * 100 == 1500 for 30 seconds of audio
			# self.bsize = 100 # n of samples for each second
			# for 7sec
			self.nblocks = 7
			self.bsize = 50

	def forward(
		self,
		x: Tensor,
		xa: Optional[Tensor] = None,
		mask: Optional[Tensor] = None,
	):
		q = self.query(x)
		k = self.key(x if xa is None else xa)
		v = self.value(x if xa is None else xa)

		x = self.qkv_attention(q, k, v, mask=mask)
		return self.out(x)

	def qkv_attention(
		self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
	):
		B, T, C = q.shape
		scale = (C // self.n_head) ** -0.25
		if self.causality != 'semi-causal':
			q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
			k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
			v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
			qk = (q @ k)
			if mask is not None and self.causality in ('causal', 'bw-semi-causal'):
				qk = qk + mask[:T, :T]
			w = F.softmax(qk.float(), dim=-1).to(q.dtype)
			return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
		else:
			q = q.view(B, self.nblocks, self.bsize, self.n_head, -1).permute(0, 3, 1, 2, 4) * scale
			k = k.view(B, self.nblocks, self.bsize, self.n_head, -1).permute(0, 3, 1, 4, 2) * scale
			v = v.view(B, self.nblocks, self.bsize, self.n_head, -1).permute(0, 3, 1, 2, 4)
			w = (q @ k).float().softmax(dim=-1).to(q.dtype)
			return (w @ v).permute(0, 2, 3, 1, 4).flatten(start_dim=3).view(B, T, C)


class ResidualAttentionBlock(nn.Module):
	def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, causality: str = 'causal'):
		super().__init__()
		self.attn = MultiHeadAttention(n_state, n_head, causality=causality)
		self.attn_ln = LayerNorm(n_state, eps=1e-8)

		self.cross_attn = (
			MultiHeadAttention(n_state, n_head, causality='non-causal') if cross_attention else None
		)
		self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

		n_mlp = n_state * 4
		self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )

		self.mlp_ln = LayerNorm(n_state)

	def forward(
		self,
		x: Tensor,
		xa: Optional[Tensor] = None,
		mask: Optional[Tensor] = None,
	):
		x = x + self.attn(self.attn_ln(x), mask=mask)

		if self.cross_attn:
			x = x + self.cross_attn(self.cross_attn_ln(x), xa)
		x = x + self.mlp(self.mlp_ln(x))
		return x


def sinusoids(length, channels, max_timescale=10000):
	"""Returns sinusoids for positional embedding"""
	assert channels % 2 == 0
	log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
	inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
	scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
	return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class AudioEncoder(nn.Module):
	def __init__(
		self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layers: int, n_frames: int,	causality: str,
	):
		super().__init__()

		self.n_layers = n_layers
		self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
		self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
		self.register_buffer('positional_embedding', sinusoids(n_ctx, n_state))

		self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
			[ResidualAttentionBlock(n_state,
				n_head,
				causality=causality,
				) for idx in range(n_layers)]
		)

		self.ln_post = LayerNorm(n_state)

		if causality == 'causal':
			mask = torch.empty(n_ctx, n_ctx).fill_(float('-inf')).triu_(1)
			self.register_buffer('mask', mask, persistent=False)
		elif causality == 'bw-semi-causal':
			# for 30sec datasets
			# num_blocks = 15 # 1500 / 100
			# block_size = 100
			# for 7sec datasets
			num_blocks = 7 # ((7 * 100) / 2) / 50 => (duration * samples)..
			block_size = 50
			mask = torch.tril(torch.ones(num_blocks, num_blocks), diagonal=0).repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
			mask[mask == 0] = float('-inf')
			mask[mask == 1] = 0
			self.register_buffer('mask', mask, persistent=False)
		else:
			self.mask = None


	def forward(self, x: Tensor):
		"""
		x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
			the mel spectrogram of the audio
		"""
		x = F.gelu(self.conv1(x))
		x = F.gelu(self.conv2(x))
		x = x.permute(0, 2, 1)

		x = (x + self.positional_embedding).to(x.dtype)

		assert x.shape[1:] == self.positional_embedding.shape, 'incorrect audio shape'

		for i, block in enumerate(self.blocks):
			x = block(x, mask=self.mask)
		x = self.ln_post(x)
		return x


class TextDecoder(nn.Module):
	def __init__(
		self, n_vocab: int, n_ctx: int, n_state: int,
		n_head: int, n_layers: int, one_shot: bool
	):
		super().__init__()
		self.one_shot = one_shot
		self.tok_embs = nn.Embedding(n_vocab, n_state)
		self.n_layers = n_layers
		# NOTE: be careful with the positional embedding
		self.pos_embs = nn.Parameter(torch.empty(n_ctx, n_state).fill_(0.001))

		self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
			[
				ResidualAttentionBlock(n_state, n_head, cross_attention=True, causality='causal')
				for _ in range(n_layers)
			]
		)

		self.ln = LayerNorm(n_state)

		mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
		self.register_buffer('mask', mask, persistent=False)


	def forward(self, x: Tensor, xa: Tensor):
		'''
			Given the text tokens and the encoded audio features, predict the next token.
			Parameters
			----------
			x: torch.LongTensor, shape = (batch_size, <= n_ctx)
				the text tokens
			xa: torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
				the encoded audio features to be attended on
			Returns
			-------
			logits: torch.FloatTensor, shape = (batch_size, n_vocab)
		'''
		B, T = x.shape
		x = (
			self.tok_embs(x) if not self.one_shot else self.tok_embs(torch.zeros_like(x).to(torch.int).to(x.device))
			+ self.pos_embs[:T].view(1, T, -1)
		)
		x = x.to(xa.dtype)

		for i, block in enumerate(self.blocks):
			x = block(x, xa, mask=self.mask)
		x = self.ln(x)

		logits = (
			x @ torch.transpose(self.tok_embs.weight.to(x.dtype), 0, 1)
		).float()
		return logits


class Whisper(nn.Module):
	def __init__(self, params: dict):
		super().__init__()
		self.params = params
		self.vocab_size = params.n_vocab
		self.encoder = AudioEncoder(
			params.n_mels,
			params.n_audio_ctx,
			params.dim,
			params.nheads,
			params.nlayers,
			params.n_frames,
			params.causal_mode,
		)

		self.decoder = TextDecoder(
			params.n_vocab,
			params.n_text_ctx,
			params.dim,
			params.nheads,
			params.nlayers,
			params.one_shot,
		)
		self.apply(self._init_weights)
		print("number of parameters: %.2fM" % (self.num_params()/1e6,))


	def num_params(self):
		n_params = sum(p.numel() for p in self.parameters())
		n_params -= self.decoder.pos_embs.numel()
		n_params -= self.decoder.tok_embs.weight.numel()
		return n_params


	def _init_weights(self, module, std: float = 0.02):
		if isinstance(module, (nn.Linear, nn.Conv1d)):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()
		elif isinstance(module, nn.LayerNorm):
			torch.nn.init.normal_(module.bias, mean=0.0, std=std)
			torch.nn.init.ones_(module.weight)


	def embed_audio(self, mel: torch.Tensor):
		return self.encoder(mel)


	@torch.no_grad()
	def inference(self, mel: torch.Tensor, seq_len: int, sampling_method: str = 'multinomial'):
		'''
			Run inference on the model.
			parameters
			----------
			mel: torch.Tensor
				The mel spectrogram
			seq_len: int
				The length of the sequence
			return
			------
			logits: torch.Tensor
				The logits
		'''
		audio_features = self.embed_audio(mel)

		if self.params.one_shot:
			seq = torch.zeros(mel.size(0), self.params.seq_len).to(self.params.device)
			logits = self.decoder(seq, audio_features)
			probs = F.softmax(logits, dim=-1)
			preds = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(-1)
			return logits, preds.view(mel.size(0), -1)
			
		tokens = [self.params.text_process.sot_i]
		seq = torch.tensor(tokens).view(1, 1).expand(mel.size(0), -1).to(self.params.device)

		for x in range(seq_len - 1):
			logits = self.decoder(seq, audio_features)
			if sampling_method == 'greedy':
				preds = torch.argmax(logits[:, -1:], dim=-1)
			elif sampling_method == 'multinomial':
				probs = F.softmax(logits[:, -1], dim=-1)
				preds = torch.multinomial(probs, num_samples=1)
			elif sampling_method == 'top-k':
				top_k_logits, top_k_indices = torch.topk(logits[:, -1:], k=10, dim=-1)
				top_k_probs = F.softmax(top_k_logits, dim=-1)
				preds = torch.multinomial(top_k_probs, num_samples=1)
				preds = top_k_indices.gather(-1, preds)
			seq = torch.cat((seq, preds), dim=1)

		return logits, seq


	def forward(self, 
			tokens: torch.Tensor, mel: torch.Tensor,
			targets: Optional[torch.Tensor] = None
	):
		'''
			Given the text tokens and the encoded audio features, predict the next token.

			Parameters
			----------
			tokens: torch.LongTensor, shape = (batch_size, <= n_ctx)
				the text tokens
			mel: torch.Tensor, shape = (batch_size, n_mels, n_frames)
				the mel spectrogram
			targets: torch.LongTensor, shape = (batch_size, <= n_ctx)
				the target text tokens
			Returns
			-------
			logits: torch.FloatTensor, shape = (batch_size, n_vocab)
				the logits
			loss: torch.FloatTensor, shape = (1,)
				the loss if targets is not None
		'''

		audio_features = self.embed_audio(mel)
		logits = self.decoder(tokens, audio_features)
		if targets is None:
			return logits
		else:
			loss = F.cross_entropy(
				logits.view(-1, self.vocab_size),
				targets.flatten() if not self.params.one_shot else tokens.flatten(),
			)

		return logits, loss


	def forward_inference(self,
			tokens: torch.Tensor, audio_features: torch.Tensor
	):
		'''
			Given the text tokens and the encoded audio features, predict the next token.

			Parameters
			----------
			tokens: torch.LongTensor, shape = (batch_size, <= n_ctx)
				the text tokens
			audio_features: torch.Tensor, shape = (batch_size, n_mels, n_frames)
				audio features
			Returns
			-------
			logits: torch.FloatTensor, shape = (batch_size, n_vocab)
				the logits
			loss: torch.FloatTensor, shape = (1,)
				the loss if targets is not None
		'''

		logits = self.decoder(tokens, audio_features)
		probs = torch.softmax(logits[:, -1], dim=-1)
		# print(probs)
		# dist = Categorical(probs)
		# token = dist.sample()
		return torch.argmax(probs)
