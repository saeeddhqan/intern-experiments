
# Disable warnings
import warnings
warnings.filterwarnings('ignore')

import os, pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys, wandb, random, json, math, hashlib
from datetime import datetime
import util, model, time, argparse
import torch
from torch import Tensor

from concurrent.futures import ThreadPoolExecutor
from util import config
from torch.utils.data import DataLoader
import numpy as np
from typing import Union, Optional, Iterable, NoReturn, Dict, Tuple


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
set_seed(1244)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
wandb.require('core')


def get_timestamp():
	now = datetime.now()
	formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
	return formatted_time



class Manager:
	def __init__(self, model: model.Whisper, mode: str) -> NoReturn:
		self.model = model
		self.model.to(config.device)
		self.mode = mode
		match config.dataset_name:
			case 'boolq':
				datawrapper = util.DataBoolq
			case 'librispeech10h':
				datawrapper = util.DataLibSpeech10h
			case 'digits':
				datawrapper = util.DataDigits
			case _:
				raise NotImplementedError('provide a supported dataset name')

		self.dataset_train = DataLoader(
			dataset=datawrapper('train', config.device),
			batch_size=config.batch_size,
			shuffle=True,
		)
		self.dataset_test = DataLoader(
			dataset=datawrapper('test', config.device),
			batch_size=config.batch_size,
			shuffle=False,
		)
		self.checkpoints = {}
		self.checkpoints_path = None
		self.model_path_format = None
		self.metrics = {'instances': [], 'time_per_sample': []}
		self.steps = 0
		self.init_config()
		config.logger.info(f"Dataset train len: {len(self.dataset_train)}")
		config.logger.info(f"Dataset test len: {len(self.dataset_test)}")


	def weighted_average_model_weights(
		self,
		checkpoint_dict: Dict,
		n: int = 5,
		alpha: float = 1.0,
		method: str = 'ensemble'
	) -> Dict:
		sorted_checkpoints = sorted(checkpoint_dict.items(), key=lambda x: x[1]['wer'])[:n]

		losses = [checkpoint[1]['wer'] for checkpoint in sorted_checkpoints]
		paths = [checkpoint[1]['path'] for checkpoint in sorted_checkpoints]

		test_losses_neg = torch.tensor([loss for loss in losses])
		test_losses_neg = -(test_losses_neg / test_losses_neg.max())
		softmax_weights = F.softmax(torch.tensor(test_losses_neg) * alpha, dim=0) # Softmax of negative losses
		averaged_weights = None
		for i, path in enumerate(paths):
			model_weights = torch.load(path)['model']
			if averaged_weights is None:
				averaged_weights = {key: torch.zeros_like(val) for key, val in model_weights.items()}
			for key in averaged_weights:
				if method == 'ensemble':
					averaged_weights[key] += softmax_weights[i] * model_weights[key]
				else:
					averaged_weights[key] += model_weights[key] / n

		return averaged_weights


	def get_lr(self, step: int = 10, warmup_steps: int = 10) -> float:
		peak_lr = 0.05 / math.sqrt(config.dim)
		return peak_lr * min(step ** -0.5, step * (warmup_steps ** -1.5))


	def init_config(self) -> NoReturn:
		if config.freeze_encoder or config.freeze_decoder:
			attr = 'encoder' if config.freeze_encoder else 'decoder'
			for param in getattr(self.model, attr).parameters():
				param.requires_grad = False


		self.optimizer = torch.optim.Adam(self.model.parameters(),
			lr=1e-4, betas=(0.9, 0.98), fused=True)

		if self.mode == 'train':
			made_checkpoints_dir = False
			if not os.path.exists(config.checkpoint_dir):
				os.mkdir(config.checkpoint_dir)
				made_checkpoints_dir = True

			os.makedirs(config.log_dir, exist_ok=True)
			os.makedirs('results', exist_ok=True)
			if not os.path.exists('results/results.csv'):
				with open('results/results.csv', 'w') as fp:
					fp.write('time,train_id,wer,train_loss,test_loss,nlayers,nheads,dim,batch_size,causal_mode,nar,mtps,test_ratio,freeze_encoder,freeze_decoder,dataset,rtf\n')

			var = config.variation.replace(' ', '_').lower()
			var += '__' + config.causal_mode
			var += f"__{config.dim}dim__{config.nlayers}nlayers__{config.batch_size}batch__{config.epoch}epochs"
			var += f"__{config.specaug_rate}sa__{config.accumulation_steps}as"
			var += f"__{config.nheads}nh"
			var += f"__{config.nar}_nar"
			var += f"__{config.freeze_encoder}_fencoder"
			var += f"__{config.freeze_decoder}_fdecoder"
			var += f"__{config.dataset_name}_ds"

			self.var = var
			if config.train_id == '':
				config.train_id = hashlib.sha1(var.encode()).hexdigest()[:8]
				self.train_id = config.train_id

			if config.wandb:
				self.wandb_init = wandb.init(
					project='intern-experiments',
					name=var,
				)

			self.model_path_format = os.path.join(
				config.checkpoint_dir,
				f"{self.train_id}",
			)
			self.create_model_path = lambda epoch: self.model_path_format + f"_{epoch}.pt"
			reload_checkpoints = self.load_checkpoints()

			if reload_checkpoints:
				self.checkpoints = {'checkpoints': {}}
				self.save_checkpoints()

			if config.wandb:
				self.wandb_init.watch(self.model, log='all')

		config.logger = util.get_logger(
			config.log_dir, config.train_id,
			config.train_id, not config.no_footprint)
		
		if self.mode == 'train':
			config.logger.info(self.var)


	def load_checkpoints(self) -> bool:
		'''
			Load checkpoints object.
		'''
		self.checkpoints_path = self.model_path_format + '.json'
		reload_checkpoints = False
		if os.path.exists(self.checkpoints_path):
			with open(self.checkpoints_path) as f:
				tmp = json.load(f)
				if tmp.get('checkpoints', None) is not None:
					self.checkpoints = tmp
				else:
					reload_checkpoints = True
		else:
			reload_checkpoints = True
		return reload_checkpoints


	def save_checkpoints(self) -> NoReturn:
		'''
			Save checkpoints object.
		'''
		if config.no_footprint:
			return
		with open(self.checkpoints_path, 'w') as f:
			json.dump(self.checkpoints, f)


	def checkpointing(self,
		step: int,
		train_loss: float,
		test_loss: float,
		wer: float,
	) -> NoReturn:
		'''
			Save a model.
		'''
		if config.no_footprint:
			return
		path = self.create_model_path(step)

		torch.save({
			'optimizer': self.optimizer.state_dict(),
			'model': self.model.state_dict(),
			'model_name': config.model_name,
			'test_loss': test_loss,
			'train_loss': train_loss,
			'wer': wer,
			'step': step,
			'path': path,
			'steps': self.steps,
			'var': self.var,
			}, path)
		self.checkpoints['checkpoints'][step] = {
			'model_name': config.model_name,
			'test_loss': test_loss,
			'train_loss': train_loss,
			'wer': wer,
			'step': step,
			'path': path,
			'steps': self.steps,
			'var': self.var,
		}
		self.save_checkpoints()


	def resume(self, path: str) -> NoReturn:
		'''
			Load a model.
		'''
		if path in ('tiny.en', 'small', 'small.en'):
			path = util.download_whisper(util.whisper_models[path], 'whisper_models')
			checkpoint = torch.load(path)['model_state_dict']
			# loading weights
			encoder = {}
			decoder = {}
			for x in checkpoint:
				if x.startswith('encoder'):
					encoder[x.replace('encoder.', '')] = checkpoint[x]
				elif x.startswith('decoder'):
					if 'positional_embedding' in x:
						decoder[x.replace('decoder.', '')] = checkpoint[x][:config.n_text_ctx]
					else:
						decoder[x.replace('decoder.', '')] = checkpoint[x]
			self.model.encoder.load_state_dict(encoder)
			self.model.decoder.load_state_dict(decoder)
			return
		try:
			checkpoint = torch.load(path)
			self.model.load_state_dict(checkpoint['model'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.steps = checkpoint['steps']
		except FileNotFoundError:
			print(f"File {path} not found.")
			sys.exit()
		except RuntimeError:
			print(f"Error loading {path}.")
			sys.exit()


	def before_train(self) -> NoReturn:
		'''
			Set train mode and load a model if any.
		'''

		if config.model_mode in ('train',):
			self.model.train()

		if config.model_path != '':
			if config.model_mode == 'train':
				config.logger.info(f"Resuming training on {config.model_path} with {config.epoch} epochs...")
				self.resume(config.model_path)

		self.model.to(config.device)


	def before_test(self) -> NoReturn:
		'''
			Set test mode in case of live and test
		'''
		if config.model_mode in ('live', 'test'):
			self.model.eval()


	def after_train(self) -> NoReturn:
		'''
			Save checkpoints, plot metrics, add results to a csv file.
		'''
		best_loss = self.calculate_loss(steps=config.test_steps * 2)
		best_res = self.comprehensive_test(mode='main')

		ensemble_weights = self.weighted_average_model_weights(self.checkpoints['checkpoints'], method='ensemble')
		self.model.load_state_dict(ensemble_weights)

		ensemble_loss = self.calculate_loss(steps=config.test_steps * 2)
		ensemble_res = self.comprehensive_test(mode='main')

		mean_weights = self.weighted_average_model_weights(self.checkpoints['checkpoints'], method='mean')
		self.model.load_state_dict(ensemble_weights)

		mean_loss = self.calculate_loss(steps=config.test_steps * 2)
		mean_res = self.comprehensive_test(mode='main')
		config.logger.info(f"Best model [test loss, train loss, wer]: {best_loss['test']}, {best_loss['train']}, {best_res['wer']}")
		config.logger.info(f"Ensemble model [test loss, train loss, wer]: {ensemble_loss['test']}, {ensemble_loss['train']}, {ensemble_res['wer']}")
		config.logger.info(f"Mean model [test loss, train loss, wer]: {mean_loss['test']}, {mean_loss['train']}, {mean_res['wer']}")

		if config.model_mode == 'train':
			if config.no_footprint:
				return
			self.save_checkpoints()
			wer, train_loss, test_loss = util.plot_metrics(self.metrics, self.train_id)
			with open('results/results.csv', 'a') as fp:
				mtps = sum(self.metrics['time_per_sample']) / len(self.metrics['time_per_sample'])
				rtf = mtps / config.chunk_length
				test_ratio = self.metrics['instances'][0][5]
				config.logger.info(f"Average time per sample: {mtps}")
				config.logger.info(f"Real Time Factor: {rtf}")
				fp.write(
					f"{get_timestamp()},{self.train_id},{wer},{train_loss},{test_loss},{config.nlayers},{config.nheads},"\
					f"{config.dim},{config.batch_size},{config.causal_mode},{config.nar},{mtps},{test_ratio},{config.freeze_encoder},"\
					f"{config.freeze_decoder},{config.dataset_name},{rtf},"\
					f"{best_loss['train']},{best_loss['test']},{best_res['wer']},"\
					f"{ensemble_loss['train']},{ensemble_loss['test']},{ensemble_res['wer']},"\
					f"{mean_loss['train']},{mean_loss['test']},{mean_res['wer']}\n"\
				)


	@torch.no_grad()
	def calculate_loss(self, steps: int) -> dict[str, int]:
		'''
			Returns train/test loss
		'''
		out = {}
		for split in (('train', self.dataset_train), ('test', self.dataset_test)):
			nsteps = steps if steps > 0 else len(split[1])
			losses = torch.zeros(nsteps)
			for k, chunk in enumerate(split[1]):
				mel, sequence, labels = chunk
				seq_len = sequence.size(1)

				with config.autocast:
					_, loss = self.model(
						sequence,
						mel,
						labels,
					)
				losses[k] = loss.item()
				if k + 1 == nsteps:
					break
			out[split[0]] = losses.mean()

		return out


	@torch.no_grad()
	def test(self,
		epoch: int,
		step: int,
		test_cond: int,
	) -> NoReturn:
		'''
			Capture metrics.
		'''
		self.model.eval()
		loss = self.calculate_loss(steps=config.test_steps)

		config.logger.info(f"[{epoch}][{step}/{len(self.dataset_train)}] > train: {round(loss['train'].item(), 4)}, test: {round(loss['test'].item(), 4)}")

		if config.wandb:
			wandb.log({
				f"test/loss": loss['test'],
				f"test/perplexity": round(torch.exp(loss['test']).item(), 5),
				f"train/loss": loss['train'],
				f"train/perplexity": round(torch.exp(loss['train']).item(), 5),
			})

		res = self.comprehensive_test()
		if config.save_checkpoints:
			self.checkpointing(
				self.steps // test_cond, loss['train'].item(), loss['test'].item(), res['wer'])

		self.metrics['instances'].append((
			epoch, step,
			loss['train'], loss['test'],
			res['wer'],
			res['test_ratio'],
		))
		self.metrics['time_per_sample'].append(res['time_per_sample'])
		if config.wandb and self.mode == 'train':
			wandb.log({
				f"test/wer": res['wer'],
			})
		self.model.train()


	def train_loop(self, epoch: int) -> NoReturn:
		'''
			Train for one epoch.
		'''
		epoch_loss = 0
		test_cond = len(self.dataset_train) // 4
		for step, chunk in enumerate(self.dataset_train):
			lr = self.get_lr(self.steps + 1) if not config.fine_tune else 1e-4

			for param_group in self.optimizer.param_groups:
				param_group['lr'] = lr

			mel, sequence, labels = chunk
			with config.autocast:
				_, loss = self.model(
					sequence,
					mel,
					labels,
				)

			loss = loss / config.accumulation_steps
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
			if (step + 1) % config.accumulation_steps == 0:
				self.optimizer.step()
				self.optimizer.zero_grad(set_to_none=True)
				if config.device == 'cuda':
					torch.cuda.synchronize()

			epoch_loss += loss.detach()
			print(step, end='\r')
			if step % test_cond == test_cond - 1:
				self.test(epoch=epoch, step=step, test_cond=test_cond)
			self.steps += 1


	def train(self) -> NoReturn:
		'''
			Training loop.
		'''

		self.before_train()

		for epoch in range(config.epoch):
			try:
				self.train_loop(epoch)
			except KeyboardInterrupt:
				print(f"Keyboard interrupt at epoch {epoch}.")
				break
		self.after_train()


	@torch.no_grad()
	def comprehensive_test(self,
		mode: str = 'micro',
	) -> Dict:
		'''
			Calculate accuracy.
		'''
		self.before_test()
		datalen = len(self.dataset_test)
		n_samples = min(datalen, 32) if config.partial_test else datalen
		n_samples = n_samples if mode == 'micro' else datalen # for final test

		test_ratio = n_samples / len(self.dataset_test)
		results = {}
		all_wer = []
		total_time = 0
		print_steps = 6
		took_tests = 0
		for step, chunk in enumerate(self.dataset_test):
			mel, sequence, labels = chunk
			start = time.perf_counter()
			seqx = self.model.inference(
				mel,
				config.n_text_ctx,
				config.text_process.eot_i,
			)
			end = time.perf_counter()
			total_time += end - start

			seqx = seqx.cpu().tolist()
			labels = labels.cpu().tolist()

			for i in range(mel.size(0)):
				wer, groundtruth, text = self.decode_single_item(seqx[i], labels[i])
				all_wer.append(wer)

			if step < print_steps:
				config.logger.info(f"real: {groundtruth}")
				config.logger.info(f"got: {text}")
			took_tests += 1
			print(step, end='\r')
			if step == n_samples:
				break

		time_per_batch = total_time / took_tests
		time_per_sample = time_per_batch / mel.size(0)
		wer = sum(all_wer) / len(all_wer)

		results['total_time'] = total_time
		results['wer'] = wer
		results['time_per_batch'] = time_per_batch
		results['time_per_sample'] = time_per_sample
		results['n_samples'] = n_samples
		results['test_ratio'] = test_ratio

		config.logger.info(f"\t\twer({mode}): {round(wer, 4)}")
		config.logger.info(f"\t\ttotal time({mode}): {round(total_time, 4)}")
		config.logger.info(f"\t\ttime per sample({mode}): {round(time_per_sample, 8)}")
		config.logger.info(f"\t\ttest ratio({mode}): {round(test_ratio, 4)}")

		return results


	def decode_single_item(self, seq: Tensor, labels: Tensor) -> Tuple:
		if config.text_process.eot_i in seq: # slice seq up to the eot.
			seq = seq[:seq.index(config.text_process.eot_i)]
		if config.text_process.eot_i in labels:
			labels = labels[:labels.index(config.text_process.eot_i)]

		text = util.normalizer(config.text_process.decoder(seq, remove_special_chars=True))
		groundtruth = config.text_process.decoder(labels, remove_special_chars=True)
		if config.dataset_name == 'digits':
			acc = util.calculate_cer(text, groundtruth)
		else:
			acc = util.calculate_wer(text, groundtruth)
		return acc.item(), groundtruth, text


	@torch.no_grad()
	def live_demo(self) -> Dict:
			'''
				Live demo.
			'''
			self.model.eval()

			while True:
				voice_path = input('Voice path: ')
				mel = util.prepare_audio(voice_path, 'cuda').unsqueeze(0).to(config.dtype)
				print('Using "greedy" for sampling')
				preds = self.model.inference(
					mel,
					config.n_text_ctx,
					config.text_process.eot_i,
					batch_process=False,
				)
				text = config.text_process.decoder(preds[0].tolist(), remove_special_chars=True)
				print('\t', text)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--action', '-a', type=str, help='train, live, and test', required=True)
	parser.add_argument('--epoch', type=int, default=config.epoch, help='epoch size')
	parser.add_argument('--checkpoint_dir', type=str, default=config.checkpoint_dir, help='directory to save checkpoints')
	parser.add_argument('--variation', type=str, default=config.variation, help='variation')
	parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch')
	parser.add_argument('--nlayers', '-nl', type=int, default=config.nlayers, help='num layers')
	parser.add_argument('--nheads', '-nh', type=int, default=config.nheads, help='num heads')
	parser.add_argument('--dim', '-d', type=int, default=config.dim, help='dim')
	parser.add_argument('--causal_mode', type=str, default=config.causal_mode, help="causality mode ('causal', 'non-causal', 'grouped-causal', 'bw-semi-causal')")
	parser.add_argument('--wandb', action='store_true', default=config.wandb, help='use wandb')
	parser.add_argument('--save_checkpoints', action='store_true', default=config.save_checkpoints, help='save checkpoints')
	parser.add_argument('--model_path', type=str, default=config.model_path, help='which model do you want to test')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help="which dataset do you want to train/test('librispeech10h', 'boolq', 'digits')")
	parser.add_argument('--accumulation_steps', '-as', type=int, default=config.accumulation_steps, help='accumulation steps')
	parser.add_argument('--nar', action='store_true', default=config.nar, help='use non auto regressive method')
	parser.add_argument('--fine_tune', action='store_true', default=config.fine_tune, help='for fine tune')
	parser.add_argument('--no_footprint', action='store_true', default=config.no_footprint, help='for fine tune')
	parser.add_argument('--freeze_encoder', action='store_true', default=config.freeze_encoder, help='freezing encoder during fine tuning')
	parser.add_argument('--freeze_decoder', action='store_true', default=config.freeze_decoder, help='freezing decoder during fine tuning')
	parser.add_argument('--partial_test', action='store_true', default=config.partial_test, help='use if you do not want to test the entire test set after each epoch')

	args = parser.parse_args()

	config.set_args(args)
	util.set_dataset_specifiers()

	if config.fine_tune:
		if config.model_path == '':
			print('provide a model with --model_path')
			exit()
		config.save_checkpoints = False
		config.variation += 'finetune'
		config.action = 'train'

	if config.action in ('test', 'live'):
		config.no_footprint = True
		if config.model_path in ('', None):
			print('provide a model with --model_path')
			exit()

	whisper = model.Whisper(config)
	manager = Manager(whisper, mode=args.action)
	config.logger.info(whisper)

	match config.action:
		case 'train':
			config.mode = 'train'
			manager.train()
		case 'live':
			manager.resume(config.model_path)
			manager.live_demo()
		case 'test':
			config.mode = 'test'
			manager.resume(config.model_path)
			results = manager.comprehensive_test()
			for key in results:
				key_title = key.replace('_', ' ').title()
				print(f"{key_title}:\n\t{results[key]}")
		case _ :
			print('Invalid action.')
