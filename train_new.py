import torch
import torch.optim as optim
import torch.nn as nn
from model_new import CapGenerator
from dataLoader import ImageCaptionSet, build_dictionary,get_loader, prepare_entry
from torch.autograd import Variable
import sys
import numpy as np
import os
import pickle
import shutil
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms
from utils import coco_eval

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    img_list = prepare_entry(args.train_dir, args.train_cap)
    sentences = [c for img in img_list for c in img['cap']]
    vocab = build_dictionary(sentences,
                             threshold=args.threshold, dict_path=args.dict_path,
                             override=False)
    train_set = ImageCaptionSet(img_list, vocab, transform, shuffle=True)
    train_loader = get_loader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2,drop_last=True)

    num_words = vocab.ukn_id + 1
    print('num_words:', num_words)
    model = CapGenerator(args.emb_dim, num_words, args.hidden_dim)

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
        start_epoch = int( args.pretrained.split('/')[-1].split('_')[1] ) + 1
    else:
        start_epoch = 1

    cnn_blocks = list(model.encoder.resnet_conv.children())[args.fine_tune_start_layer:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_blocks]
    cnn_params = [item for sublist in cnn_params for item in sublist]
    cnn_optimizer = torch.optim.Adam(cnn_params, lr=args.lr_cnn, betas=(args.alpha, args.beta))

    other_params = (list(model.encoder.ai2v.parameters()) + 
                   list(model.encoder.ag2v.parameters()) +
                   list(model.decoder.parameters()))

    lr = args.lr
    criterion = nn.CrossEntropyLoss().cuda()
    model.cuda()
    iter_size = len(train_loader)
    #val_iter = len(val_loader)
    cider_scores = []
    best_cider = 0.0
    best_epoch = 0
    print('ITER size: {}', iter_size)
    for epoch in range(start_epoch, args.num_epochs + 1):
        if train_set.shuffle:
            np.random.shuffle(train_set.entries)
            print('shuffle train dataset')
        if epoch > args.lr_decay_start:
            frac = float(epoch - args.lr_decay_start) / args.lr_decay_ratio
            decay_fac = np.power(0.5, frac)
            lr = lr * decay_fac

        print('learning rate for Epoch {}: {:.3e}'.format(epoch, lr))
        optimizer = torch.optim.Adam(other_params, lr=lr, betas=(args.alpha, args.beta))
        model.train()
        for i, data in enumerate(train_loader):
            inputs, _, caps, last_pos = data
            inputs, caps = Variable(inputs).cuda(), Variable(caps).cuda()
            lstm_steps = max(last_pos)
            #targets = pack_padded_sequence(caps, last_pos, batch_first=True)
            model.zero_grad()
            packed_scores = model(inputs, caps, last_pos)
            targets = pack_padded_sequence(caps[:, 1:], last_pos, batch_first=True)
            #print(caps.shape, caps[:, 1:].shape, last_pos)
            loss = criterion(packed_scores[0], targets[0])
            loss.backward()


            #????
            for p in model.decoder.LSTM.parameters():
                p.data.clamp_(-args.clip, args.clip)

            optimizer.step()

            cnn_lr = args.lr_cnn
            if epoch > args.cnn_epoch:
                #cnn_lr = cnn_lr * decay_fac
                cnn_optimizer = torch.optim.Adam(cnn_params, lr=cnn_lr, betas=(args.alpha, args.beta))
                cnn_optimizer.step()
                
            scores = pad_packed_sequence(packed_scores, batch_first=True)[0]
            last = scores[-1]
            last_ind = list(last.max(1)[1].data)
            last_truth = list(caps[-1, 1:].data)
            print('TRAIN ITER: {} / {}, lstm_steps:{}, loss: {:.4f},Perplexity:{}\r'.format(i, iter_size, lstm_steps, loss.data[0], np.exp(loss.data[0])), end="")
        print("\n",end="")
        if epoch % args.save_freq == args.save_freq - 1:
            name = os.path.join(args.model_dir, 'epoch_{}'.format(epoch))
            torch.save(model.state_dict(), name)

        scores = pad_packed_sequence(packed_scores, batch_first=True)[0]
        last = scores[-1]
        last_ind = list(last.max(1)[1].data)
        last_truth = list(caps[-1, 1:].data)
        print(last_truth, last_pos[-1])

        print('pred: ',end="")
        for ix in last_ind:
            print(vocab.ix2word(ix), end="")
            if ix == 0:
                print("")
                break
            print(' ', end="")
        if ix != 0:
            print("\b.")
        print('truth: ',end="")
        for ix in last_truth:
            print(vocab.ix2word(ix), end="")
            if ix == 0:
                print("")
                break
            print(' ', end="")
        if ix != 0:
            print("\b.")

        #cider scores
        cider = coco_eval(model, args, epoch)
        cider_scores.append(cider)

        if cider > best_cider:
            best_cider = cider
            best_epoch = epoch

        if len(cider_scores) > 5:
            last_6 = np.array(cider_scores[-6:])
            if max(last_6) < best_cider:
                print('No improvement with CIDEr in the last 6 epochs...Early stopping triggered.')
                print('Model of best epoch #: %d with CIDEr score %.2f'%( best_epoch, best_cider ))
                break

    torch.save(model.state_dict(), os.path.join(args.model_dir, 'trained_model'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./models',
            help='dir name to save trained models')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--dict_path', default='./dict')
    parser.add_argument('--threshold', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_dir', default='/home/liang.tan.image_caption/ai_challenger_caption_train_20170902/caption_train_images_20170902')
    parser.add_argument('--train_cap', default='/home/liang.tan/image_caption/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json')
    parser.add_argument('--val_dir', default='/data/public/weixishuo/chinese-caption/ai_challenger_caption_validation_20170910/caption_validation_images_20170910')
    parser.add_argument('--val_cap', default='/data/public/weixishuo/chinese-caption/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--vocab_path', default='./dict')
    parser.add_argument('--batch_size', type=int, default=40)

    parser.add_argument('--fine_tune_start_layer', type=int, default=5)
    parser.add_argument('--cnn_epoch', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--lr_cnn', type=float, default=1e-5)


    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)


    parser.add_argument('--pretrained', default='')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--lr_decay_start', type=int, default=5)
    parser.add_argument('--lr_decay_ratio', type=int, default=50)

    args = parser.parse_args()

    main(args)
