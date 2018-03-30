import json, os, pickle, sys
import hashlib
import argparse
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from model_new import CapGenerator
from torchvision import transforms, datasets
from dataLoader import ImageCaptionSet, prepare_entry, get_loader,ImageSet
from PIL import Image
from torch.autograd import Variable
from Evaluation.caption_eval.run_evaluations import compute_m1

def dump_eval(args):
    transform = transforms.Compose([
                transforms.Resize((args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    num_words = vocab.ukn_id + 1
    model = CapGenerator(args.emb_dim, num_words, args.hidden_dim)
    model.load_state_dict(torch.load(args.weigth_path))
    model.cuda()
    model.eval()
    img_list = prepare_entry(args.val_dir, args.val_cap, save_as='val_img_list')
    id_to_refs = {}
    for img in img_list:
        id_to_refs[img["img_id"]] = img["cap"]
    val_set = ImageSet(img_list, vocab, transform, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = args.batch_size,shuffle=False, num_workers=2, drop_last=True)
    
    scores = [0] * args.ngram
    results={}
    to_dump = []
    id_set = set()
    val_iter = len(val_loader)
    smoothie = nltk.translate.bleu_score.SmoothingFunction().method4
    for i, data in enumerate(val_loader):
        img_inputs, img_ids = data
        img_inputs = Variable(img_inputs).cuda()
        gen_caps, _, _ = model.sampler(img_inputs)
        gen_caps = gen_caps.cpu().data.numpy()
        for img_ind in range(args.batch_size):
            sentence = []
            for word_ix in gen_caps[img_ind]:
                if word_ix == 0:
                    break
                else:
                    sentence.append(vocab.ix2word(word_ix))
            to_dump.append({"image_id":img_ids[img_ind].split('.')[0], "caption": "".join(sentence)})
            results[img_ids[img_ind]] = sentence
        print('VAL ITER : {} / {}\r'.format(i, val_iter), end="")
    with open('id_to_test_caption.json', 'w') as f:
        json.dump(to_dump, f)


def coco_eval(model, args, epoch, res_file=None):
    ann_file = '/data/public/weixishuo/chinese-caption/id_to_words.json'
    model.eval()

    transform = transforms.Compose([
                transforms.Resize((args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    img_list = prepare_entry(args.val_dir, args.val_cap, save_as='val_img_list')
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    results = []
    id_set = set()
    val_set = ImageSet(img_list, vocab, transform, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = args.batch_size,shuffle=False, num_workers=2, drop_last=True)

    for i, data in enumerate(val_loader):
        if res_file:
           break
        inputs, img_ids = data
        inputs = Variable(inputs).cuda()
        generated, _, _ = model.sampler(inputs)

        gen_caps = generated.cpu().data.numpy()
        batch_size = inputs.shape[0]

        for img_ind in range(batch_size):
            sampled_ids = gen_caps[img_ind]
            sampled_caption = []

            for word_id in sampled_ids:

                word = vocab.ix2word(word_id)

                if word == '.':
                    break
                else:
                    sampled_caption.append(word)

            sentence = ''.join(sampled_caption)
            id = img_ids[img_ind]
            if not id in id_set:
                id_set.add(id)
            else:
                print('duplicated id : {}'.format(id))
                raise Exception
            temp = {'image_id': id.split('.')[0],'caption':sentence}
            results.append(temp)

        if (i + 1) % 10 == 0:
            print('\rEVAL ITER: {}/{}'.format(i+1, len(val_loader)), end="")

    print("")
    resFile = './results/epoch_{}.json'.format(epoch)
    if not res_file:
        with open(resFile, 'w') as f:
            json.dump(results, f)
    else:
        resFile = res_file

    scores = compute_m1(resFile, ann_file)
    cider = scores['CIDEr']

    return cider



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('-w','--weigth_path', default='./models/epoch_2')
    parser.add_argument('--ngram', type=int, default=4)
    parser.add_argument('--vocab_path', default='./dict')
    parser.add_argument('--val_dir', default='./ai_challenger_caption_validation_20170910/caption_validation_images_20170910')
    parser.add_argument('--val_cap', default='./ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json')
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=30)
    args = parser.parse_args()

    model = CapGenerator(args.emb_dim, 7171, args.hidden_dim)
    model.load_state_dict(torch.load(args.weigth_path))
    model.eval()
    model.cuda()
    print(compute_m1('/data/public/weixishuo/chinese-caption/results/epoch_2.json', '/data/public/weixishuo/chinese-caption/id_to_words.json'))
