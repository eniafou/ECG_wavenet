"""
A script for WaveNet training
"""
import os
from tqdm import tqdm
import wavenet.config as config
from wavenet.model import WaveNet
from wavenet.utils.data import Dataset
import torch.utils.data as data


class Trainer:
    def __init__(self, args):
        self.args = args

        self.wavenet = WaveNet(args.layer_size, args.stack_size,
                               args.in_channels, args.res_channels,
                               lr=args.lr)

        self.dataset = Dataset(args.data_dir, self.wavenet.receptive_fields, args.in_channels, args.data_len)
        self.data_loader = data.DataLoader(self.dataset, batch_size=args.batch_size,shuffle=True)



    def run(self):
        
        num_epoch = args.num_epoch
        loss_per_epoch = []
        for epoch in range(num_epoch):
            for i, (inputs, targets) in enumerate(self.data_loader):
                print(i)
                loss = self.wavenet.train(inputs, targets)
                if True :#(i+1)%5 == 0:
                    print('[{0}/{1}] loss: {2}'.format(epoch + 1, num_epoch, loss))
            
            self.wavenet.lr_scheduler.step(loss)
            loss_per_epoch.append(loss)
        
        print(loss_per_epoch)

        self.wavenet.save(args.model_dir)


def prepare_output_dir(args):
    args.log_dir = os.path.join(args.output_dir, 'log')
    args.model_dir = os.path.join(args.output_dir, 'model')
    args.test_output_dir = os.path.join(args.output_dir, 'test')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)


if __name__ == '__main__':
    args = config.parse_args()

    prepare_output_dir(args)

    trainer = Trainer(args)

    trainer.run()
