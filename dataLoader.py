import os, pickle, torch, cv2, collections
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data
from skimage import io,transform
from PIL import Image
import jieba
import json


class ImageSet(data.Dataset):
    def __init__(self, img_list, vocab, transform=None, shuffle=False):
        self.entries = [{'img_id':img['img_id'],
                         'file_name': img['file_name']} for img in img_list]
        if shuffle:
            np.random.shuffle(self.entries)
        self.trans = transform

    def __getitem__(self, index):

        # assuming that each image has five captions
        img = self.entries[index]
        file_name = img['file_name']
        # TODO: resize image
        img_data = Image.open(file_name).convert('RGB')
        img_id = img['img_id']
        if self.trans:
            img_input = self.trans(img_data)
        return img_input, img_id

    def __len__(self):
        return len(self.entries)

    def get_name(self):
        return CLASS_NAME

class Vocab():
    def __init__(self, word2ix, ix2word, max_len):
        self._ix2word = ix2word
        self.max_len = max_len
        self._word2ix = word2ix
        self.ukn_id = len(word2ix)
        self._ix2word[self.ukn_id] = "<UNKNOWN>"

    def ix2word(self, ix):
        return self._ix2word[ix]

    def word2ix(self, word):
        return self._word2ix[word] if word in self._word2ix else self.ukn_id

def prepare_entry(img_dir, capfile, save_as='./img_list'):
    if os.path.exists(save_as):
        print('entry exists, loading...')
        with open(save_as, 'rb') as f:
            return pickle.load(f)
    with open(capfile) as f:
        annotations = json.load(f)

    res = []
    length = len(annotations)
    for idx, ann in enumerate(annotations):
        img_id = ann["image_id"]
        cap = [process_cap(c) for c in ann["caption"]]
        file_name = os.path.join(img_dir, img_id)
        res.append({"img_id":img_id, "file_name": file_name, "cap": cap, "url":ann["url"]})
        if idx % 100 == 0:
            print("{}/{}\r".format(idx,length))

    with open(save_as, 'wb') as f:
        pickle.dump(res, f)
        
    return res


class ImageCaptionSet(data.Dataset):
    def __init__(self, img_list, vocab, transform=None, shuffle=False):
        self.entries = [{'img_id':img['img_id'],
                         'file_name': img['file_name'],
                         'cap': cap} for img in img_list for cap in img['cap']]
        self.trans = transform
        self.vocab = vocab
        self.max_len = vocab.max_len
        self.shuffle = shuffle

    def __getitem__(self, index):
        # TODO: resize image
        img = self.entries[index]
        img_id = img['img_id']
        img_data = Image.open(img['file_name']).convert('RGB')
        cap_ind = cap2ind(img['cap'], self.vocab)
        last_pos = len(cap_ind)
        #TODO: max len in val set is longer than train set
        if self.trans:
            img_data = self.trans(img_data)
        return img_data, img_id, cap_ind
    
    def __len__(self):
        return min(210000, len(self.entries))

def collate_fn(data):
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, img_ids, cap_inds = zip(*data)
    img_ids = list(img_ids)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in cap_inds]
    targets = torch.zeros(len(cap_inds), max(lengths)).long()
    for i, cap in enumerate(cap_inds):
        end = lengths[i]
        targets[i, :end] = cap

    lengths = [l - 1 for l in lengths]
    return images, img_ids, targets, lengths

def get_loader(dataset, batch_size, shuffle=True, num_workers=1, drop_last=False):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers,
                                       drop_last=drop_last, collate_fn=collate_fn)

def process_cap(caption):
    tokenized_caption = ["<START>"]
    tokenized_caption.extend(jieba.cut(caption), cut_all=False)
    tokenized_caption.append(".")
    return tokenized_caption

def build_dictionary(sentences, threshold=30, save=True,
                     dict_path='./dict', override=False):
    print('building dictionary...')
    if os.path.exists(dict_path) and not override:
        print('dictionary already exists, loading')
        with open(dict_path, 'rb') as f:
            dict_obj = pickle.load(f)
            return dict_obj

    print('{} sentences'.format(len(sentences)))
    print('calculating word counts...'.format(threshold))
    word2ix = {}
    ix2word = dict()
    word2ix["."] = 0
    ix2word[0] = "."
    word2ix["<START>"] = 1
    ix2word[1] = "<START>"
    word_counts = collections.defaultdict(int)
    max_len = 0
    for sentence in sentences:
        for word in sentence[1:-1]:
            word_counts[word] += 1
        max_len = max(max_len, len(sentence))

    print('{} different words fond including "<START>" and "."'.format(len(word_counts) + 2))
    print('max length of sentences: {}'.format(max_len))
    print('building dictionary...')
    crt_ix = 2
    
    for w in word_counts:
        if word_counts[w] > threshold:
            if w in word2ix:
                print(w, word2ix[w])
                continue
            word2ix[w] = crt_ix

            ix2word[crt_ix] = w
            crt_ix += 1
    print('dictionary built, {} words including "<START>" and ".", threshold is {}'.format(crt_ix, threshold))
    print(len(word2ix), len(ix2word))
    assert len(word2ix) == len(ix2word)
    vocab = Vocab(word2ix, ix2word, max_len)
    if save:
        with open(dict_path, 'wb') as f:
            pickle.dump(vocab, f)
        print('saved as {}'.format(dict_path))

    return vocab

def cap2ind(cap, vocab):
    """
    convert caption to corresponding indices
    :return: index of words in caption
    """
    ixs = [vocab.word2ix(w) for w in cap]

    return torch.LongTensor(ixs)

