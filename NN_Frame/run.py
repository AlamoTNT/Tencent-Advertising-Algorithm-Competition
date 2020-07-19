import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import numpy as np
import pandas as pd
import argparse
from tqdm import trange,tqdm
import os
from utils import read_corpus,batch_iter
from vocab import Vocab
from model import LSTM, CNN, WordAVGModel
import math
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import AdamW,get_linear_schedule_with_warmup

def set_seed():
    random.seed(3344)
    np.random.seed(3344)
    torch.manual_seed(3344)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3344)

def split_sents(batch_data, vocab_list, device):
    #ad, advertiser, creative, industry, product
    ad_batch, advertiser_batch, creative_batch, industry_batch, product_batch = [], [], [], [], []
    for i in range(len(batch_data)):
        ad_batch.append(batch_data[i][0])
        advertiser_batch.append(batch_data[i][1])
        creative_batch.append(batch_data[i][2])
        industry_batch.append(batch_data[i][3])
        product_batch.append(batch_data[i][4])
    
    ad_batch = vocab_list[0].vocab.to_input_tensor(ad_batch, device)
    advertiser_batch = vocab_list[1].vocab.to_input_tensor(advertiser_batch, device)
    creative_batch = vocab_list[2].vocab.to_input_tensor(creative_batch, device)
    industry_batch = vocab_list[3].vocab.to_input_tensor(industry_batch, device)
    product_batch = vocab_list[4].vocab.to_input_tensor(product_batch, device)
    
    return [ad_batch, advertiser_batch, creative_batch, industry_batch, product_batch]

def train(args,model, train_data,dev_data,vocab,dtype='CNN'):
    LOG_FILE = args.output_file
    with open(LOG_FILE, "a") as fout:
        fout.write('\n')
        fout.write('=========='*6)
        fout.write('start trainning: {}'.format(dtype))
        fout.write('\n')

    time_start = time.time()
    if not os.path.exists(os.path.join('./runs',dtype)):
        os.makedirs(os.path.join('./runs',dtype))
    tb_writer = SummaryWriter(os.path.join('./runs',dtype))

    t_total = args.num_epoch * (math.ceil(len(train_data) / args.train_batch_size))
    optimizer = AdamW(model.parameters(), lr=args.learnning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    total_loss = 0.
    logg_loss = 0.
    val_acces = []
    train_epoch = trange(args.num_epoch, desc='train_epoch')
    for epoch in train_epoch:
        model.train()

        for src_sents,labels in batch_iter(train_data, args.train_batch_size, shuffle=True):
            src_sents = split_sents(src_sents, vocab, args.device)
            global_step += 1
            optimizer.zero_grad()

            logits = model(src_sents)
            #print(len(logits))
            y_labels = torch.tensor(labels, device=args.device)
            
            #print(logits, y_labels)
            example_losses = criterion(logits, y_labels)

            example_losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += example_losses.item()
            if global_step % args.out_step == 0:
                loss_scalar = (total_loss - logg_loss) / args.out_step
                logg_loss = total_loss

                with open(LOG_FILE, "a") as fout:
                    fout.write("epoch: {}, iter: {}, loss: {},learn_rate: {}\n".format(epoch, global_step, loss_scalar,
                                                                                       scheduler.get_lr()[0]))
                print("epoch: {}, iter: {}, loss: {}, learning_rate: {}".format(epoch, global_step, loss_scalar,
                                                                                scheduler.get_lr()[0]))
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", loss_scalar, global_step)
            torch.cuda.empty_cache()

        print("Epoch", epoch, "Training loss", total_loss / global_step)

        eval_loss,eval_result = evaluate(args,criterion, model, dev_data,vocab)  # 评估模型
        with open(LOG_FILE, "a") as fout:
            fout.write("EVALUATE: epoch: {}, loss: {},eval_result: {}\n".format(epoch, eval_loss,eval_result))
            # tb_writer.add_scalars('eval_result',eval_result,epoch)
            # tb_writer.add_scalar('eval loss',eval_loss,epoch)
        eval_acc = eval_result['acc']
        if len(val_acces) == 0 or eval_acc > max(val_acces):
            # 如果比之前的acc要da，就保存模型
            print("best model on epoch: {}, eval_acc: {}".format(epoch, eval_acc))
            torch.save(model.state_dict(), "classifa-best-{}.th".format(dtype))
            val_acces.append(eval_acc)

    time_end = time.time()
    print("run model of {},taking total {} m".format(dtype,(time_end-time_start)/60))
    with open(LOG_FILE, "a") as fout:
        fout.write("run model of {},taking total {} m\n".format(dtype,(time_end-time_start)/60))

def evaluate(args, criterion, model, dev_data,vocab):
    model.eval()
    total_loss = 0.
    total_step = 0.
    preds = None
    out_label_ids = None
    with torch.no_grad():#不需要更新模型，不需要梯度
        for src_sents, labels in batch_iter(dev_data, args.train_batch_size):
            src_sents = split_sents(src_sents, vocab, args.device)
            logits = model(src_sents)
            labels = torch.tensor(labels,device=args.device)
            example_losses = criterion(logits,labels)

            total_loss += example_losses.item()
            total_step += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            torch.cuda.empty_cache()

    preds = np.argmax(preds, axis=1)
    result = acc_and_f1(preds, out_label_ids)
    model.train()
    print("Evaluation loss", total_loss/total_step)
    print('Evaluation result', result)
    return total_loss/total_step,result

def test(args, criterion, model, te_data, vocab):
    model.eval()
    #total_loss = 0.
    #total_step = 0.
    preds = None
    #out_label_ids = None
    #不需要更新模型，不需要梯度
    with torch.no_grad():
        for src_sents in batch_iter(te_data, args.test_batch_size, test_batch=True):
            src_sents = split_sents(src_sents, vocab, args.device)
            logits = model(src_sents)
            #labels = torch.tensor(labels,device=args.device)
            #example_losses = criterion(logits,labels)

            #total_loss += example_losses.item()
            #total_step += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                #out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                #out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            torch.cuda.empty_cache()

    #preds = np.argmax(preds, axis=1)
    #result = acc_and_f1(preds, out_label_ids)
    #model.train()
    #print("Evaluation loss", total_loss/total_step)
    #print('Evaluation result', result)
    return preds
    

def acc_and_f1(preds,labels):
    acc = (preds == labels).mean()
    f1 = f1_score(y_true=labels,y_pred=preds,average='weighted')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def build_vocab(vocab_path):
    
    ad_vocab = Vocab.load(vocab_path+'ad_vocab.json')
    advertiser_vocab = Vocab.load(vocab_path+'advertiser_vocab.json')
    creative_vocab = Vocab.load(vocab_path+'creative_vocab.json')
    industry_vocab = Vocab.load(vocab_path+'industry_vocab.json')
    product_vocab = Vocab.load(vocab_path+'product_vocab.json')
    return [ad_vocab, advertiser_vocab, creative_vocab, industry_vocab, product_vocab]



def main():
    parse = argparse.ArgumentParser()

    parse.add_argument("--train_data_dir", default='./work/cache/cache_train_data', type=str, required=False)
    parse.add_argument("--dev_data_dir", default='./work/cache/cache_dev_data', type=str, required=False)
    parse.add_argument("--test_data_dir", default='./work/cache/cache_test_data', type=str, required=False)
    parse.add_argument("--src_train_data_dir", default='./work/cache/cache_creative_src_train_data', type=str, required=False)
    parse.add_argument("--output_file", default='creative_deep_model.log', type=str, required=False)
    parse.add_argument("--train_batch_size", default=256, type=int)
    parse.add_argument("--test_batch_size", default=8, type=int)
    parse.add_argument("--train_len", default=900000, type=int)
    parse.add_argument("--sample_rate", default=0.3, type=float)
    parse.add_argument("--random_state", default=1017, type=float)
    parse.add_argument("--do_train",default=False, action="store_true", help="Whether to run training.")
    parse.add_argument("--do_test",default=True, action="store_true", help="Whether to run testing.")
    parse.add_argument("--learnning_rate", default=5e-4, type=float)
    parse.add_argument("--num_epoch", default=5, type=int)
    parse.add_argument("--out_step", default=200, type=int)
    parse.add_argument("--max_vocab_size", default=1000000, type=int)
    parse.add_argument("--min_freq", default=2, type=int)
    parse.add_argument("--embed_size", default=300, type=int)
    parse.add_argument("--hidden_size", default=256, type=int)
    parse.add_argument("--dropout_rate", default=0.2, type=float)
    parse.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
    parse.add_argument("--GRAD_CLIP", default=1, type=float)
    parse.add_argument("--vocab_path", default='./work/json/', type=str)
    parse.add_argument("--do_cnn",default=False, action="store_true", help="Whether to run cnn training.")
    parse.add_argument("--do_rnn", default=True, action="store_true", help="Whether to run rnn training.")
    parse.add_argument("--do_avg",default=False, action="store_true", help="Whether to run avg training.")

    parse.add_argument("--num_filter", default=100, type=int,help="CNN模型一个filter的输出channels")

    args = parse.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if torch.cuda.is_available():
        print('=======use gpu=======')
    else:
        print('=======use cpu=======')
        
        
    set_seed()

    if os.path.exists(args.train_data_dir) and os.path.exists(args.dev_data_dir) and os.path.exists(args.test_data_dir):
        print('=======load data========')
        train_data = torch.load(args.train_data_dir)
        dev_data = torch.load(args.dev_data_dir)
        test_data = torch.load(args.test_data_dir)
    else:
        src_data, labels = read_corpus(args.train_data_dir)
        src_tr_data, src_te_data = src_data[:args.train_len], src_data[args.train_len:]
        X_train, X_val, y_train, y_val= train_test_split(src_tr_data, labels, test_size=args.sample_rate, random_state=args.random_state)
        
        train_data = [(text,labs) for text,labs in zip(X_train, y_train)]
        dev_data = [(text,labs) for text,labs in zip(X_val, y_val)]
        torch.save(train_data,'./work/cache/cache_train_data')
        torch.save(train_data,'./work/cache/cache_dev_data')

    vocab_list = build_vocab(args.vocab_path)
    #label_map = vocab.labels
    #print(label_map)
    num_classes = 20

    if args.do_train:
        if args.do_cnn:
            cnn_model = CNN(len(vocab.vocab),args.embed_size,args.num_filter,[2,3,4],len(label_map),dropout=args.dropout_rate)
            cnn_model.to(device)
            train(args,cnn_model,train_data,dev_data,vocab,dtype='CNN')

        if args.do_avg:
            avg_model = WordAVGModel(len(vocab.vocab),args.embed_size,len(label_map),dropout=args.dropout_rate)
            avg_model.to(device)
            train(args, avg_model, train_data, dev_data, vocab, dtype='AVG')

        if args.do_rnn:
            rnn_model = LSTM(vocab_list,args.embed_size,args.hidden_size,
                            num_classes,n_layers=1,bidirectional=True,dropout=args.dropout_rate)
            rnn_model.to(device)
            train(args, rnn_model, train_data, dev_data, vocab_list, dtype='LSTM')
    else:
        print('pass training...')

    if args.do_test:


        cirtion = nn.CrossEntropyLoss()
        
        """
        cnn_model = CNN(len(vocab.vocab), args.embed_size, args.num_filter, [2, 3, 4], len(label_map),
                        dropout=args.dropout_rate)
        print('load model')
        cnn_model.load_state_dict(torch.load('creative_classifa-best-CNN.th'))
        cnn_model.to(device)
        print('start predict')
        cnn_result = test(args, cirtion, cnn_model, test_data, vocab)
        print('end predict')
        cnn_sub = pd.DataFrame(cnn_result)
        cnn_sub.to_csv('./work/cnn_test_props_creative_id.csv')
        print('Submission save sucessfully!')
        """
        """
        avg_model = WordAVGModel(len(vocab.vocab), args.embed_size, len(label_map), dropout=args.dropout_rate)
        avg_model.load_state_dict(torch.load('classifa-best-AVG.th'))
        avg_model.to(device)
        avg_result = test(args, cirtion, avg_model, test_data, vocab)
        avg_sub = pd.DataFrame(avg_result)
        avg_sub.to_csv('./work/test_props_feature/ad_avg_sub.csv')
        print('Submission save sucessfully!')
        """
        rnn_model = LSTM(vocab_list, args.embed_size, args.hidden_size,
                        num_classes, n_layers=1, bidirectional=True, dropout=args.dropout_rate)
        rnn_model.load_state_dict(torch.load('classifa-best-LSTM.th'))
        rnn_model.to(device)
        rnn_result = test(args, cirtion, rnn_model, test_data, vocab_list)
        rnn_sub = pd.DataFrame(rnn_result)
        len(rnn_sub)
        rnn_sub.to_csv('./work/lstm_sub.csv')
        print('Submission save sucessfully!')
        
if __name__ == '__main__':
    main()
