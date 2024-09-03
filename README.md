To train with non-causal(base) configuration: 
```bash

python main.py -a train --dataset_name librispeech10h --no_footprint --nlayers 12 --dim 768 --nheads 12 --batch_size 8 --model_pa
th small.en --freeze_encoder
```
