import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import pickle
from model_new import CapGenerator
from dataLoader import prepare_entry
import argparse
import os, shutil
from PIL import Image
import numpy as np


def main(args):
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    with open(args.dict_path, 'rb') as f:
        vocab = pickle.load(f)
    img_list = prepare_entry(args.val_dir, args.val_cap, save_as='val_img_list')
    if args.image_id:
        file_name = os.path.join(args.val_dir, args.image_id)
    else:
        img = np.random.choice(img_list)
        cap = img['cap']
        file_name = img['file_name']
        print(cap)
    num_words = vocab.ukn_id + 1
    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    if not args.img_path:
        img = Image.open(file_name).convert('RGB')
    else:
        file_name = args.img_path
        img = Image.open(args.img_path).convert('RGB')
    print(os.path.split(file_name)[-1])
    shutil.copyfile(file_name, './val_img.jpg')
    img_inputs = transform(img).unsqueeze(0)
    img_inputs = Variable(img_inputs).cuda()
    net = CapGenerator(args.emb_dim, num_words, args.hidden_dim)
    net.load_state_dict(torch.load(args.weight_path))
    net.cuda()
    net.eval()

    #cap, _, _ = net.beam_search(img_inputs, beam=5)
    cap, _, _ = net.sampler(img_inputs)
    cap = cap.squeeze().data
    #print(cap)
    cap_str = []
    for ix in cap:
        word = vocab.ix2word(ix)
        if word == '.':
            break
        else:
            cap_str.append(word)
    sentence = ' '.join(cap_str)
    print(sentence + '.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--weight-path", default="/data/public/weixishuo/chinese-caption/models/epoch_49")
    parser.add_argument('-i', "--image-id",default=None)
    parser.add_argument('--img_path', default=None)
    parser.add_argument('--dict_path', default='/data/public/weixishuo/chinese-caption/dict')
    parser.add_argument('--val_dir', default='/home/liang.tan/image_caption/ai_challenger_caption_validation_20170910/caption_validation_images_20170910')
    parser.add_argument('--val_cap', default='/home/liang.tan/image_caption/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--emb_dim',type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    args = parser.parse_args()
    main(args)
