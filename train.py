import logging
import os
import time, datetime, math
import torch
from torch.utils.data import DataLoader

from data import LoadDataset
from config import *
from model import init_model
from optimizer import init_optimizer
from loss import init_loss
from utils import parse_opt, load_cfg


def train(config):
    os.makedirs(config['train']['model_savepath'], exist_ok=True)

    logging.basicConfig(filename = config['log_savepath'],
                        format = '%(asctime)s : %(levelname)s : %(name)s : %(message)s',
                        filemode = config['log_mode'], )

    logger = logging.getLogger() 
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO) 

    config['logger'] = logger

    model_type = config['modelname']

    train_data = LoadDataset(config, phase = 'train')
    valid_data = LoadDataset(config, phase = 'valid')
    train_loader = DataLoader(train_data, batch_size=config['train']['batch_size'], num_workers=config['train']['num_workers'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config['valid']['batch_size'], num_workers=config['valid']['num_workers'])
    
    model = init_model(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = init_optimizer(model, config)
    loss_fn = init_loss(config)
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    highest_acc = 0

    batch_number = math.ceil(len(train_loader.dataset) / config['train']['batch_size'])
    logger.info("Start training loop")
    for epoch in range(config['train']['epoch']):
        # Start Training
        model.train()
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_number // 10 > 0:
                print_fre = batch_number // 10
            else:
                print_fre = 1
            if batch_idx % print_fre == print_fre - 1:
                iter_num = batch_idx * len(data)
                total_data = len(train_loader.dataset)
                iter_num = str(iter_num).zfill(len(str(total_data)))
                total_percent = 100. * batch_idx / len(train_loader)
                logger.info(f'Train Epoch {epoch + 1}: [{iter_num}/{total_data} ({total_percent:2.0f}%)] | Loss: {loss.item():.6f}')
                

        # Start Validating
        logger.info(f"Validating {len(valid_loader.dataset)} images")
        model.eval()
        correct = 0
        for (data, target) in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        logger.info(str(correct) + '/' + str(len(valid_loader.dataset)))
        accuracy = 100. * correct / len(valid_loader.dataset)
        logger.info('\nValid set: Accuracy: {}/{} ({:.2f}%)'.format(
            correct, len(valid_loader.dataset), accuracy))

        
        stop = time.time()
        runtime = stop - start
        eta = int(runtime * (config['train']['epoch'] - epoch - 1))
        eta = str(datetime.timedelta(seconds=eta))
        logger.info(f'Runing time: Epoch {epoch + 1}: {str(datetime.timedelta(int(second=runtime)))} | ETA: {eta}')

        torch.save(model.state_dict(), os.path.join(config['train']['model_savepath'], f'{model_type}_last.pth'))
        logger.info(f"Saving last model to {os.path.join(config['train']['model_savepath'], f'{model_type}_last.pth')}\n")

        if accuracy >= highest_acc:
            highest_acc = accuracy
            torch.save(model.state_dict(), os.path.join(config['train']['model_savepath'], f'{model_type}_best.pth'))
            logger.info(f"Saving best model to {os.path.join(config['train']['model_savepath'], f'{model_type}_best.pth')}\n")

if __name__ == '__main__':
    opt = parse_opt()
    cfg = load_cfg(opt.cfg)
    train(config = cfg)