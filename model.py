
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
	def __init__(self, n_state: int, n_head: int, causalty: bool):
		super().__init__()
		self.n_head = n_head
		self.query = Linear(n_state, n_state, bias=False)
		self.key = Linear(n_state, n_state, bias=False)
		self.value = Linear(n_state, n_state, bias=True)
		self.out = Linear(n_state, n_state, bias=True)
		self.dropout = 0.0
		self.causalty = causalty

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
		q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
		k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
		v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

		# x = F.scaled_dot_product_attention(q, k, v,
		# 	dropout_p=self.dropout if self.training else 0,
		# 	is_causal=self.causalty,
		# 	scale=1.0)
		qk = (q @ k)
		if mask is not None and self.causalty:
			qk = qk + mask[:T, :T]
		qk = qk.float()
		w = F.softmax(qk.float(), dim=-1).to(q.dtype)
		return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class NonLinear(nn.Module):
	def __init__(self, n_state):
		super().__init__()
		self.dim = n_state
		self.w1 = nn.Linear(self.dim, 4 * self.dim, bias=True)
		self.w2 = nn.Linear(4 * self.dim, self.dim, bias=True)
		# self.w3 = nn.Linear(self.dim, 4 * self.dim, bias=False)

	def forward(self, x: Tensor):
		'''
			mlp forward
		'''
		# return self.w2(F.silu(self.w1(x)) * self.w3(x))
		return self.w2(F.gelu(self.w1(x)))


class ResidualAttentionBlock(nn.Module):
	def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, causalty: bool = True):
		super().__init__()
		self.attn = MultiHeadAttention(n_state, n_head, causalty=causalty)
		self.attn_ln = LayerNorm(n_state, eps=1e-8)

		self.cross_attn = (
			MultiHeadAttention(n_state, n_head, causalty=False) if cross_attention else None
		)
		self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

		n_mlp = n_state * 4
		self.mlp = NonLinear(n_state)
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
		self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layers: int, n_frames: int,	causalty: str,
	):
		super().__init__()
		causalty = causalty == 'causal'

		self.n_layers = n_layers
		self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
		self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
		self.register_buffer('positional_embedding', sinusoids(n_ctx, n_state))

		self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
			[ResidualAttentionBlock(n_state, n_head, causalty=causalty, idx=idx) for idx in range(n_layers)]
		)
		self.ln_encode = LayerNorm(n_state)
		self.ln_post = LayerNorm(n_state)
		self.mask = None
		if causalty:
			mask = torch.empty(n_ctx, n_ctx).fill_(float('-inf')).triu_(1)
			self.register_buffer('mask', mask, persistent=False)


	def forward(self, x: Tensor):
		"""
		x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
			the mel spectrogram of the audio
		"""

		x = F.gelu(self.conv1(x))
		x = F.gelu(self.conv2(x))
		x = x.permute(0, 2, 1)

		x = self.ln_encode(x) # accelerates training

		x = (x + self.positional_embedding).to(x.dtype)

		assert x.shape[1:] == self.positional_embedding.shape, 'incorrect audio shape'

		for i, block in enumerate(self.blocks):
			x = block(x, mask=self.mask)
		x = self.ln_post(x)
		return x


class TextDecoder(nn.Module):
	def __init__(
		self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layers: int,
	):
		super().__init__()
		self.tok_embs = nn.Embedding(n_vocab, n_state)
		self.n_layers = n_layers
		# NOTE: be careful with the positional embedding
		self.pos_embs = nn.Parameter(torch.empty(n_ctx, n_state).fill_(0.001))

		# self.dropout = nn.Dropout(0.1)
		self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
			[
				ResidualAttentionBlock(n_state, n_head, cross_attention=True, causalty=True)
				for _ in range(n_layers)
			]
		)
		self.ln = LayerNorm(n_state)
		# self.ln = LayerNorm(n_state)
		# self.class_head = Linear(n_state + lang_dim, n_state)
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
			self.tok_embs(x)
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
			params.n_audio_state,
			params.n_audio_head,
			params.n_audio_layer,
			params.n_frames,
			params.causal_mode,
		)

		self.decoder = TextDecoder(
			params.n_vocab,
			params.n_text_ctx,
			params.n_text_state,
			params.n_text_head,
			params.n_text_layer,
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
		tokens = [self.params.text_process.sot_i]
		seq = torch.tensor(tokens).view(1, 1).expand(mel.size(0), -1).to(self.params.device)
		audio_features = self.embed_audio(mel)
		for x in range(seq_len):
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
				targets.flatten(),
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
