from turtle import pd
import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

from send_email import send_email
import time
import datetime
import pdb

torch.manual_seed(args.seed)    #
checkpoint = utility.checkpoint(args)


def main():
    time_start = datetime.datetime.now()
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()
    
    time_end = datetime.datetime.now()
    #send_email(str(time_end-time_start))

if __name__ == '__main__':
    main()
