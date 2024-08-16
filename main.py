
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

from util import config
from torch.utils.data import DataLoader
import numpy as np
from typing import Union, Optional, Iterable

def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
wandb.require('core')
set_seed(1244)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_timestamp():
	now = datetime.now()
	formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
	return formatted_time


def get_logger(log_dir: str, name: str, log_filename: str ='log'):
	os.makedirs(log_dir, exist_ok=True)

	class CustomLogger:
		def __init__(self, name):
			self.name = name
			self.log_file = open(os.path.join(log_dir, log_filename + '.log'), 'a')

		def info(self, message):
			log_entry = f"{datetime.now()} - {self.name} - INFO - {message}"
			self.log_file.write(log_entry + '\n')
			print(log_entry)

		def __del__(self):
			self.log_file.close()

	print('Log directory: ', log_dir)
	return CustomLogger(name)


class Manager:
	def __init__(self, model: model.Whisper):
		self.model = model
		self.model.to(config.device)
		self.optimizer = torch.optim.Adam(model.parameters(),
			lr=1e-7, betas=(0.9, 0.98), fused=True)
		self.preprocess_encoder = util.prepare_audio
		self.preprocess_decoder = util.prepare_text
		self.dataset_train = DataLoader(
			dataset=util.Data(config.train_path, config.device),
			batch_size=config.batch_size,
			shuffle=True,
			num_workers=2,
		)
		self.dataset_test = DataLoader(
			dataset=util.Data(config.test_path, config.device),
			batch_size=config.batch_size,
			shuffle=False,
			num_workers=2,
		)
		self.dataset_ctest = DataLoader(
			dataset=util.Data(config.test_path, config.device),
			batch_size=1,
			shuffle=False,
		)
		self.checkpoints = {}
		self.checkpoints_path = None
		self.writer = None
		self.model_path_format = None
		self.loss = 0.0
		self.metrics = []
		self.init_config()


	def get_lr(self, step, warmup_steps=10):
		peak_lr = 0.05 / np.sqrt(config.dim)
		return peak_lr * min(step ** -0.5, step * (warmup_steps ** -1.5))


	def capture_metrics(self, epoch: int, step: int, train_loss: float, test_loss: float, wer: float, accuracy: float):
		self.metrics.append((epoch, step, train_loss, test_loss, wer, accuracy))


	def init_config(self):
		made_checkpoints_dir = False
		if not os.path.exists(config.checkpoint_dir):
			os.mkdir(config.checkpoint_dir)
			made_checkpoints_dir = True

		os.makedirs(config.log_dir, exist_ok=True)
		os.makedirs('results', exist_ok=True)
		if not os.path.exists('results/results.csv'):
			with open('results/results.csv', 'w') as fp:
				fp.write('time,train_id,accuracy,wer,train_loss,test_loss,nlayers,nheads,dim,batch_size,causal_mode\n')

		var = config.variation.replace(' ', '_').lower()
		var += '__' + config.causal_mode
		var += f"__{config.dim}dim__{config.nlayers}nlayers__{config.batch_size}batch__{config.epochs}epochs"
		var += f"__{config.specaug_rate}sa__{config.accumulation_steps}as"
		var += f"__{config.nheads}nh"

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
		if config.log_anything:
			config.logger = get_logger(config.log_dir, config.train_id, config.train_id)


	def load_checkpoints(self):
		'''
			Load checkpoints object.
		'''


		self.checkpoints_path = self.model_path_format + '.json'
		reload_checkpoints = False
		if os.path.exists(self.checkpoints_path):
			with open(self.checkpoints_path) as f:
				tmp = json.load(f)
				if 'checkpoints' in tmp and 'best_model' in tmp:
					if (isinstance(tmp['checkpoints'], dict)):
						self.checkpoints = tmp
					else:
						reload_checkpoints = True
				else:
					reload_checkpoints = True
		else:
			reload_checkpoints = True
		return reload_checkpoints


	def save_checkpoints(self):
		'''
			Save checkpoints object.
		'''

		with open(self.checkpoints_path, 'w') as f:
			json.dump(self.checkpoints, f)


	def checkpointing(self, epoch: int, 
		train_loss: float, test_loss: float,
	):
		'''
			Save a model.
		'''

		path = self.create_model_path(epoch)

		torch.save({
			'optimizer': self.optimizer.state_dict(),
			'model': self.model.state_dict(),
			'model_name': config.model_name,
			'test_loss': test_loss,
			'train_loss': train_loss,
			'epoch': epoch,
			'path': path,
			'lang': config.language,
			}, path)
		self.checkpoints['checkpoints'][epoch] = {
			'model_name': config.model_name,
			'test_loss': test_loss,
			'train_loss': train_loss,
			'epoch': epoch,
			'path': path,
		}
		self.save_checkpoints()


	def resume(self, path: str):
		'''
			Load a model.
		'''

		try:
			checkpoint = torch.load(path)
			self.model.load_state_dict(checkpoint['model'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
		except FileNotFoundError:
			print(f"File {path} not found.")
			sys.exit(1)
		except RuntimeError:
			print(f"Error loading {path}.")
			sys.exit(1)


	def before_train(self):
		'''
			To do before training.
		'''

		if config.model_mode in ('train',):
			self.model.train()

		if config.epoch_start > 0:
			if config.model_mode == 'train':
				start = config.epoch_start
				start = start - (start % config.test_freq)
				start = max(start, config.test_freq)

				print(f"Resuming training at epoch {start}...")
				path = self.create_model_path(start)
				self.resume(path)
				config.epoch_start = start + 1

		self.model.to(config.device)


	def before_test(self):
		'''
			To do before test
		'''
		if config.model_mode in ('live', 'test'):
			self.model.eval()


	def after_train(self):
		'''
			To do after training.
		'''
		if config.model_mode == 'train':
			self.save_checkpoints()
			acc, wer, train_loss, test_loss = util.plot_metrics(self.metrics, self.train_id)
			with open('results/results.csv', 'w') as fp:
				fp.write(f"{get_timestamp()},{self.train_id},{acc},{wer},{train_loss},{test_loss},{config.nlayers},{config.nheads},{config.dim},{config.batch_size},{config.causal_mode}\n")


	def select_top_model(self, checkpoints: dict):
		'''
			Select the top model.
		'''

		top_model = None
		top_model_loss = float('inf')
		for _, checkpoint in enumerate(checkpoints['checkpoints']):
			checkpoint = checkpoints['checkpoints'][checkpoint]
			if checkpoint['test_loss'] < top_model_loss:
				top_model_loss = checkpoint['test_loss']
				top_model = checkpoint.copy()
		return top_model


	@torch.no_grad()
	def calculate_loss(self, steps: int) -> dict[str, int]:
		'''
			Returns train/test loss
		'''
		out = {}
		for split in (('train', self.dataset_train), ('test', self.dataset_test)):
			losses = torch.zeros(steps if steps > 0 else len(split[1]))
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
				if k == steps:
					break
			out[split[0]] = losses.mean()

		return out


	@torch.no_grad()
	def test(self, epoch: int, step: int = None, mode: str):
		'''
			Test a model.
		'''
		self.model.eval()
		mode = mode == 'main'
		loss = self.calculate_loss(steps=-1 if mode else 32)

		if mode:
			config.logger(f"[{epoch}] > train: {loss['train']}, test: {loss['test']}")
		else:
			config.logger(f"\t[{epoch}][step] > train: {loss['train']}, test: {loss['test']}")

		config.logger('-' * 30)
		if config.wandb:
			if mode:
				wandb.log({
					'test/loss': loss['test'],
					'test/perplexity': round(torch.exp(loss['test']).item(), 5),
					'train/loss': loss['train'],
					'train/perplexity': round(torch.exp(loss['train']).item(), 5),
				})
			else:
				wandb.log({
					'test_micro/loss': loss['test'],
					'test_micro/perplexity': round(torch.exp(loss['test']).item(), 5),
					'train_micro/loss': loss['train'],
					'train_micro/perplexity': round(torch.exp(loss['train']).item(), 5),
				})
		if config.save_checkpoints:
			self.checkpointing(epoch, loss['train'].item(), loss['test'].item())
		res = self.comprehensive_test(n_samples=64 if mode else 16, mode='main' if mode else 'micro')

		self.capture_metrics(epoch, step, loss['train'], loss['test'], res['wer'], res['accuracy'])

		self.model.train()


	def train_loop(self):
		'''
			Train loop.
		'''
		epoch_loss = 0
		for step, chunk in enumerate(self.dataset_train):
			lr = self.get_lr(self.steps + 1)

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
			if step % 100 == 99:
				self.test(epoch=epoch, step=step, mode='micro')
			self.steps += 1


	def train(self):
		'''
			Training loop.
		'''

		self.before_train()

		for epoch in range(config.epoch_start, config.epoch_stop):
			test_cond = epoch % config.test_freq == 0
			lr = self.get_lr(epoch + 1)

			try:
				self.train_loop()

				if test_cond:
					self.test(epoch, 'main')
			except KeyboardInterrupt:
				print(f"Keyboard interrupt at epoch {epoch}.")
				break

		self.after_train()
		self.comprehensive_test(mode='main')


	@torch.no_grad()
	def get_model_complexity(model, logger):
		parameter_dict = parameter_count(self.model)
		num_parameters = parameter_dict['']

		if logger is not None:
			logger.info(f"Number of parameters: {num_parameters:,}")


		return num_parameters


	@torch.no_grad()
	def comprehensive_test(self, n_samples: int = -1, mode: str = 'micro'):
		'''
			Comprehensive test.
			Calculate accuracy with word error rate.
			Estimate the time it takes to transcribe all the test data.
		'''
		if n_samples == -1:
			self.before_test()

		all_data_size = n_samples if n_samples > 0 else len(self.dataset_ctest)
		results = {}
		all_r = []
		all_h = []
		total_time = 0
		print_steps = 4
		for step, chunk in enumerate(self.dataset_ctest):
			mel, sequence, labels = chunk
			start = time.perf_counter()
			_, seqx = self.model.inference(
				mel,
				config.block_size,
			)
			end = time.perf_counter()
			total_time += end - start
			text = config.text_process.decoder(seqx[0].tolist(), True)
			groundtruth = config.text_process.decoder(labels[0].tolist(), True)
			diff = len(groundtruth) - len(text)
			if diff > 0:
				text += '~' * diff
			elif diff < 0:
				text = text[:len(groundtruth)]
			all_r.append(groundtruth)
			all_h.append(text)
			if step < print_steps:
				config.logger('GT: \t\t', groundtruth)
				config.logger('Predicted: \t\t', text)
			if step == n_samples:
				break

		accuracy, performance, _, _, wer = util.overall_accuracy(all_r, all_h)
		time_per_sample = total_time / all_data_size

		results['total_time'] = total_time
		results['accuracy'] = accuracy
		results['performance'] = performance
		results['wer'] = wer
		results['time_per_sample'] = time_per_sample
		results['all_data_size'] = all_data_size

		config.logger('\t\t', '-' * 10)
		config.logger('\t\t', 'wer:', wer)
		config.logger('\t\t', 'accuracy:', accuracy)
		config.logger('\t\t', 'performance:', performance)
		config.logger('\t\t', 'total_time:', total_time)
		config.logger('\t\t', 'per sample:', time_per_sample)
		config.logger('\t\t', '-' * 10)
		if config.wandb:
			if mode == 'main':
				wandb.log({
					'test/wer': wer,
					'test/accuracy': accuracy,
					'test/performance': performance,
				})
			else:
				wandb.log({
					'test_micro/wer': wer,
					'test_micro/accuracy': accuracy,
					'test_micro/performance': performance,
				})

		return results


	@torch.no_grad()
	def live_demo(self):
			'''
				Live demo.
			'''
			self.model.eval()

			while True:
				voice_path = input('Voice path: ')
				mel = util.prepare_audio(voice_path, 'cuda').unsqueeze(0)
				for method in ('multinomial', 'top-k', 'greedy'):
					print('Using ', method)
					_, seqx = self.model.inference(
						mel,
						config.block_size,
						method
					)
					text = config.text_process.decoder(seqx[0].tolist(), True)
					print('\t', text)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--action', '-a', type=str, help='train, live, and test', required=True)
	parser.add_argument('--epoch_start', type=int, default=config.epoch_start, help='epoch to start training from')
	parser.add_argument('--epoch_stop', type=int, default=config.epoch_stop, help='epoch to end training')
	parser.add_argument('--checkpoint_dir', type=str, default=config.checkpoint_path, help='directory to save checkpoints')
	parser.add_argument('--variation', type=str, default=config.variation, help='variation')
	parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch')
	parser.add_argument('--nlayers', '-nl', type=int, default=config.nlayers, help='num layers')
	parser.add_argument('--nheads', '-nh', type=int, default=config.nheads, help='num heads')
	parser.add_argument('--dim', '-d', type=int, default=config.dim, help='dim')
	parser.add_argument('--causal_mode', type=int, default=config.causal_mode, help="causalty mode ('causal', 'non-causal')")
	parser.add_argument('--log_anything', type='store_true', default=config.log_anything, help='log states (verbosity)')
	parser.add_argument('--wandb', type='store_true', default=config.wandb, help='use wandb')
	parser.add_argument('--save_checkpoints', type='store_true', default=config.save_checkpoints, help='save checkpoints')
	parser.add_argument('--model_path', type=str, default=config.model_path, help='which model do you want to test')
	parser.add_argument('--accumulation_steps', '-as', type=str, default=config.accumulation_steps, help='accumulation steps')

	args = parser.parse_args()

	config.set_args(args)

	whisper = model.Whisper(config)
	manager = Manager(whisper)

	match config.action:
		case 'train':
			config.mode = 'train'
			manager.train()
		case 'live':
			manager.resume(config.model_path)
			manager.live_demo()
		case 'test':
			manager.resume(config.model_path)
			results = manager.comprehensive_test()
			for key in results:
				key_title = key.replace('_', ' ').title()
				print(f"{key_title}:\n\t{results[key]}")
		case _ :
			print('Invalid action.')
