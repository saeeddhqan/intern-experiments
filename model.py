
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import math, random
from typing import Optional, Union, Iterable, NoReturn, Dict, Tuple
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


def sinusoids(
	length: int,
	channels: int,
	max_timescale: int = 10000,
) -> Tensor:

	assert channels % 2 == 0
	log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
	inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
	scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
	return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


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
		self,
		x: Tensor,
		weight: Tensor,
		bias: Optional[Tensor]
	) -> Tensor:
		return super()._conv_forward(
			x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
		)


class MultiHeadAttention(nn.Module):
	def __init__(self,
		n_state: int,
		n_head: int,
		causality: str,
		nblocks: int = None,
		bsize: int = None,
	) -> NoReturn:
		super().__init__()

		self.n_head = n_head
		self.query = Linear(n_state, n_state)
		self.key = Linear(n_state, n_state, bias=False)
		self.value = Linear(n_state, n_state)
		self.out = Linear(n_state, n_state)
		self.causality = causality
		if causality == 'grouped-causal':
			self.nblocks = nblocks
			self.bsize = bsize

	def forward(
		self,
		x: Tensor,
		xa: Optional[Tensor] = None,
		mask: Optional[Tensor] = None,
	) -> NoReturn:

		q = self.query(x)
		k = self.key(x if xa is None else xa)
		v = self.value(x if xa is None else xa)

		x = self.qkv_attention(q, k, v, mask=mask)
		return self.out(x)

	def qkv_attention(self,
		q: Tensor,
		k: Tensor,
		v: Tensor,
		mask: Optional[Tensor] = None,
	) -> NoReturn:

		B, T, C = q.shape
		scale = (C // self.n_head) ** -0.25
		if self.causality != 'grouped-causal':
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
	def __init__(self,
		n_state: int, n_head: int, cross_attention: bool = False,
		causality: str = 'causal', nblocks: int = None, bsize: int = None,
	) -> NoReturn:
		super().__init__()
		self.attn = MultiHeadAttention(n_state, n_head, causality=causality, nblocks=nblocks, bsize=bsize)
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

	def forward(self,
		x: Tensor,
		xa: Optional[Tensor] = None,
		mask: Optional[Tensor] = None,
	) -> Tensor:

		x = x + self.attn(self.attn_ln(x), mask=mask)
		if self.cross_attn:
			x = x + self.cross_attn(self.cross_attn_ln(x), xa)
		x = x + self.mlp(self.mlp_ln(x))
		return x


class AudioEncoder(nn.Module):
	def __init__(
		self, n_mels: int, n_ctx: int, n_state: int,
		n_head: int, n_layers: int, n_frames: int,
		causality: str, dataset: str,
	) -> NoReturn:
		super().__init__()

		self.n_layers = n_layers
		self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
		self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
		self.register_buffer('positional_embedding', sinusoids(n_ctx, n_state))
		
		nblocks = None
		bsize = None

		if causality == 'causal':
			mask = torch.empty(n_ctx, n_ctx).fill_(float('-inf')).triu_(1)
			self.register_buffer('mask', mask, persistent=False)
		elif causality == 'bw-semi-causal':
			nblocks = 15 if dataset == 'boolq' else 7
			bsize = 100 if dataset == 'boolq' else 50
			mask = torch.tril(torch.ones(nblocks, nblocks), diagonal=0).repeat_interleave(bsize, dim=0).repeat_interleave(bsize, dim=1)
			mask[mask == 0] = float('-inf')
			mask[mask == 1] = 0
			self.register_buffer('mask', mask, persistent=False)
		elif causality == 'grouped-causal':
			nblocks = 15 if dataset == 'boolq' else 7
			bsize = 100 if dataset == 'boolq' else 50
			self.mask = None
		else:
			self.mask = None


		self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
			[ResidualAttentionBlock(n_state,
				n_head,
				causality=causality,
				nblocks=nblocks,
				bsize=bsize,
				) for idx in range(n_layers)]
		)

		self.ln_post = LayerNorm(n_state)


	def forward(self, x: Tensor) -> Tensor:
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
	) -> NoReturn:
		super().__init__()
		self.one_shot = one_shot
		self.token_embedding = nn.Embedding(n_vocab, n_state)
		self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state)) # warn
		self.n_layers = n_layers

		self.blocks = nn.ModuleList(
			[
				ResidualAttentionBlock(
					n_state, n_head,
					cross_attention=True,
					causality='causal',
				)
				for _ in range(n_layers)
			]
		)

		self.ln = LayerNorm(n_state)

		mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
		self.register_buffer('mask', mask, persistent=False)


	def forward(self, x: Tensor, xa: Tensor) -> Tensor:
		B, T = x.shape

		if self.one_shot:
			x = torch.zeros_like(x).to(torch.int).to(x.device)

		x = self.token_embedding(x) + self.positional_embedding[:T].view(1, T, -1)
		x = x.to(xa.dtype)

		for i, block in enumerate(self.blocks):
			x = block(x, xa, mask=self.mask)
		x = self.ln(x)

		logits = (
			x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
		).float()
		return logits


class Whisper(nn.Module):
	def __init__(self, params: Dict) -> NoReturn:
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
			params.dataset_name,
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
		print("number of parameters: %.2fM" % (self.num_params() / 1e6,))


	def num_params(self) -> int:
		n_params = sum(p.numel() for p in self.parameters())
		n_params -= self.decoder.positional_embedding.numel()
		n_params -= self.decoder.token_embedding.weight.numel()
		return n_params


	def _init_weights(self, module, std: float = 0.02) -> NoReturn:
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


	def embed_audio(self, mel: Tensor) -> Tensor:
		return self.encoder(mel)


	@torch.no_grad()
	def inference(self,
		audio_features: Tensor,
		seq_len: int,
		eot: int,
		batch_process: bool = True,
	) -> Tuple:

		audio_features = self.embed_audio(audio_features)
		sampling_method = 'greedy'
		B = audio_features.size(0)
		if self.params.one_shot:
			seq = torch.zeros(B, self.params.seq_len).to(self.params.device)
			logits = self.decoder(seq, audio_features)
			probs = F.softmax(logits, dim=-1)
			preds = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(-1)
			return logits, preds.view(B, -1)
			
		tokens = [self.params.text_process.sot_i]
		seq = torch.tensor(tokens).flatten().view(1, -1).expand(B, -1).to(self.params.device)
		seq_len -= seq.size(1)

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
			if batch_process is False:
				if preds[0, 0] == eot:
					break
		return seq


	def forward(self, 
		tokens: Tensor, mel: Tensor,
		targets: Optional[Tensor] = None
	) -> Tuple:

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
