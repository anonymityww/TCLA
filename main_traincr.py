import argparse
import os
import time
import torch
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
from model import Model
from dataloader import TrainDataset, ValidDataset, TestDataset
from script import init_seeds, xavier_init, evaluate_function, get_metrics, stat_operation, seqs_normalization

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-tf', type=str, default='./Sports/train.dat', help='train dataset')
    parser.add_argument('-vf', type=str, default='./Sports/valid.dat', help='valid dataset')
    parser.add_argument('-ef', type=str, default='./Sports/test.dat', help='test dataset')
    parser.add_argument('-vn', type=str, default='./Sports/valid_neg.dat', help='neg for valid')
    parser.add_argument('-en', type=str, default='./Sports/test_neg.dat', help='neg for test')
    parser.add_argument('-b', type=int, default=256, help='batch size')
    parser.add_argument('-ls', type=int, default=50, help='log_step')
    parser.add_argument('-lr', type=float, default=1e-3, help='lr')
    parser.add_argument('-e', type=int, default=150, help='epoch')
    parser.add_argument('-ae', type=int, default=50, help='pre-training epoch')
    parser.add_argument('-dr', type=float, default=0.5, help='reg')
    parser.add_argument('-hd', type=int, default=64, help='hidden size')
    parser.add_argument('-hn', type=int, default=2, help='head num')
    parser.add_argument('-ln', type=int, default=2, help='transformer layers')
    parser.add_argument('-o', type=str, default='./save_model/sports/par_test/1.0-0.005/', help='save path')
    parser.add_argument('-m', type=str, default="test", help='train valid or test')
    parser.add_argument('-r', action='store_true', help='resume')
    parser.add_argument('-n', type=int, default=18360, help='item nums + 3')  #real item nums + padding + EOS + mask
    parser.add_argument('-ml', type=int, default=50, help='max_seqs_len')
    parser.add_argument('-mml', type=int, default=60, help='modified_max_seqs_len')
    parser.add_argument('-mi', type=int, default=5, help='max_insert_num')
    parser.add_argument('-mb', type=float, default=0.5, help='mask_prob')
    parser.add_argument('-p', type=float, nargs='+', default=[0.4, 0.85], help='plist')
    parser.add_argument('-g', type=int, default=4, help='gradient accumulation')
    args = parser.parse_args()
    train_file = args.tf
    valid_file = args.vf
    test_file = args.ef
    valid_neg_file = args.vn
    test_neg_file = args.en
    batch_size = args.b
    log_step = args.ls
    learning_rate = args.lr
    epochs = args.e
    aug_epochs = args.ae
    dropout_rate = args.dr
    hidden_unit = args.hd
    head_num = args.hn
    layer_num = args.ln
    save_path = args.o
    mode = args.m
    resume = args.r
    item_num = args.n
    max_seqs_len = args.ml
    modified_seqs_len = args.mml
    max_insert_num = args.mi
    mask_prob = args.mb
    plist = args.p
    grad_acc = args.g

    init_seeds()
    if mode == 'train':
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + 'model/'):
            os.makedirs(save_path + 'model/')
        if resume:
            fw = open(save_path + 'train_result.txt', 'a')
            fw2 = open(save_path + 'similarity.txt', 'a')
        else:
            fw = open(save_path + 'train_result.txt', 'w')
            fw2 = open(save_path + 'similarity.txt', 'w')
        dataset = TrainDataset(train_file, item_num, max_seqs_len, modified_seqs_len, max_insert_num, mask_prob, plist)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        model = Model(item_num, layer_num, head_num, hidden_unit, dropout_rate, max_insert_num)

        last_epoch = 0
        if resume:
            with open(save_path + 'train_result.txt', 'r') as f:
                content = f.readlines()
            last_epoch = int(len(content) / 2) - 1
            print('load model: epoch %d' % (last_epoch,))
            model.load_state_dict(torch.load(save_path + 'model/TCLA-' + str(last_epoch) + '.pth'))
        else:
            print('initialize model')
            model.apply(xavier_init)
        if torch.cuda.is_available():
            print('cuda is available')
            model.cuda()
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        epoch = last_epoch
        while epoch < epochs:
            epoch += 1
            step = 0
            similarity_list = []
            operation_list = []
            acc_loss_recommender = 0
            acc_loss_augmentor = 0
            acc_loss_cl = 0
            acc_loss_tri_cl = 0
            start_time = time.time()
            if epoch <= aug_epochs:
                for batch in dataloader:
                    step += 1
                    optimizer.zero_grad()
                    input_seqs_ids, targets, negatives, rand_aug_seqs, random_modified_seqs, l1_ground_truth, l2_ground_truth, insert_seqs = batch
                    if torch.cuda.is_available():
                        input_seqs_ids = input_seqs_ids.cuda()
                        targets = targets.cuda()
                        negatives = negatives.cuda()
                        rand_aug_seqs = rand_aug_seqs.cuda()
                        random_modified_seqs = random_modified_seqs.cuda()
                        l1_ground_truth = l1_ground_truth.cuda()
                        l2_ground_truth = l2_ground_truth.cuda()
                        insert_seqs = insert_seqs.cuda()

                    """ 
                    train augmentor and get augmentor loss 
                    """
                    full_layer_output, insert_net_output, padding_mask = model.augmentor_forward(random_modified_seqs,
                                                                                                 insert_seqs)
                    l1_loss, l2_loss = model.augmentor_loss(full_layer_output, insert_net_output, l1_ground_truth,
                                                            l2_ground_truth, padding_mask)
                    loss1 = l1_loss.sum() / (random_modified_seqs != 0).sum()
                    if (l2_ground_truth != 0).sum() == 0:
                        loss2 = l2_loss.sum()
                    else:
                        loss2 = l2_loss.sum() / (l2_ground_truth != 0).sum()
                    total_loss = loss1 + loss2
                    acc_loss_augmentor += total_loss
                    if step % log_step == 0:
                        print('Epoch %d Step %d augmentor loss %0.4f Time %d' % (
                            epoch, step, acc_loss_augmentor / step, time.time() - start_time
                        ))

                    model.train()

                    loss = total_loss
                    loss.backward()
                    optimizer.step()

                torch.save(model.state_dict(), save_path + 'model/TCLA-' + str(epoch) + '.pth')
                print('Epoch %d augmentor loss %0.4f Time %d' % (
                epoch, acc_loss_augmentor / step, time.time() - start_time))
                fw.write('Epoch %d augmentor loss %0.4f' % (epoch, acc_loss_augmentor / step) + '\n')
                fw.flush()

            elif epoch > aug_epochs:
                for batch in dataloader:
                    step += 1
                    optimizer.zero_grad()
                    input_seqs_ids, targets, negatives, rand_aug_seqs, random_modified_seqs, l1_ground_truth, l2_ground_truth, insert_seqs = batch
                    if torch.cuda.is_available():
                        input_seqs_ids = input_seqs_ids.cuda()
                        targets = targets.cuda()
                        negatives = negatives.cuda()
                        rand_aug_seqs = rand_aug_seqs.cuda()
                        random_modified_seqs = random_modified_seqs.cuda()
                        l1_ground_truth = l1_ground_truth.cuda()
                        l2_ground_truth = l2_ground_truth.cuda()
                        insert_seqs = insert_seqs.cuda()

                    model.eval()
                    with torch.no_grad():
                        full_layer_output_ori, insert_net_output_ori = model.augmentor_inference(input_seqs_ids)
                        ssl_aug_seqs = model.seqs_augmentation(full_layer_output_ori, insert_net_output_ori, input_seqs_ids)
                        ssl_padding_seqs = seqs_normalization(ssl_aug_seqs, modified_seqs_len, item_num)

                    model.train()

                    if torch.cuda.is_available():
                        ssl_aug_seqs = ssl_padding_seqs.cuda()

                    operation = stat_operation(full_layer_output_ori, input_seqs_ids)
                    operation_list.extend(operation)

                    output, cl_loss, tri_cl_loss, distance1, distance2 = model.train_forward(input_seqs_ids, rand_aug_seqs, ssl_aug_seqs)

                    loss_rec = model.rec_loss(output, targets, cl_loss, tri_cl_loss)
                    acc_loss_recommender += loss_rec
                    acc_loss_cl += cl_loss
                    acc_loss_tri_cl += tri_cl_loss
                    loss = loss_rec
                    loss.backward()
                    optimizer.step()
                    if step % log_step == 0:
                        print('epoch %d step %d recommender loss %0.4f time %d' %
                              (epoch, step, acc_loss_recommender/step, time.time()-start_time))
                        print('epoch %d step %d cl loss %0.4f time %d' %
                              (epoch, step, acc_loss_cl/step, time.time()-start_time))
                        print('epoch %d step %d tri_cl loss %0.4f time %d' %
                              (epoch, step, acc_loss_tri_cl/step, time.time()-start_time))

                torch.save(model.state_dict(), save_path + 'model/TCLA-' + str(epoch) + '.pth')
                print('epoch %d recommender loss %0.4f time %d' %
                      (epoch, acc_loss_recommender / step, time.time() - start_time))
                print('epoch %d cl loss %0.4f time %d' %
                      (epoch, acc_loss_cl / step, time.time() - start_time))
                print('epoch %d tri_cl loss %0.4f time %d' %
                      (epoch, acc_loss_tri_cl / step, time.time() - start_time))

                fw.write('epoch %d recommender loss %0.4f' % (epoch, acc_loss_recommender / step) + '\n')
                fw.flush()

                operation_list = np.array(operation_list)
                operation_list = operation_list.sum(0)/operation_list.sum()

                similarity_result = {'operation': operation_list.tolist()}

                print(str(similarity_result))
                fw2.write(str(similarity_result) + '\n')
        fw.close()
        fw2.close()


    if mode == "valid":
        if resume:
            fw = open(save_path + 'valid_result.txt', 'a')
        else:
            fw = open(save_path + 'valid_result.txt', 'w')
        dataset = ValidDataset(valid_file, valid_neg_file, item_num, modified_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        model = Model(item_num, layer_num, head_num, hidden_unit, dropout_rate, max_insert_num)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        next_epoch = aug_epochs + 1
        if resume:
            with open(save_path + 'valid_result.txt', 'r') as f:
                content = f.readlines()
            next_epoch = len(content) + 1
            print('valid from epoch %d' % (next_epoch,))
        epoch = next_epoch
        while epoch <= epochs:
            step = 0
            total_result = []
            model.load_state_dict(torch.load(save_path + 'model/TCLA-' + str(epoch) + '.pth'))
            with torch.no_grad():
                for batch in dataloader:
                    step += 1
                    input_seqs_ids, targets, negatives, neg_num = batch
                    if torch.cuda.is_available():
                        input_seqs_ids = input_seqs_ids.cuda()
                        targets = targets.cuda()
                        negatives = negatives.cuda()
                        neg_num = neg_num.cuda()

                    output = model.test_forward(input_seqs_ids)  # [batch_size, hidden_unit]
                    output = torch.matmul(output, model.item_embedding.weight.T)  # [batch_size, item_num]
                    result = evaluate_function(output, targets, negatives, neg_num)
                    total_result.extend(result)

            total_result_dict = {'epoch': epoch}
            total_result_dict['recall@5'] = get_metrics('recall@5', total_result)
            total_result_dict['recall@10'] = get_metrics('recall@10', total_result)
            total_result_dict['recall@20'] = get_metrics('recall@20', total_result)
            total_result_dict['mrr@5'] = get_metrics('mrr@5', total_result)
            total_result_dict['mrr@10'] = get_metrics('mrr@10', total_result)
            total_result_dict['mrr@20'] = get_metrics('mrr@20', total_result)
            total_result_dict['ndcg@5'] = get_metrics('ndcg@5', total_result)
            total_result_dict['ndcg@10'] = get_metrics('ndcg@10', total_result)
            total_result_dict['ndcg@20'] = get_metrics('ndcg@20', total_result)

            total_result_dict['sum'] = total_result_dict['recall@5'] + total_result_dict['recall@10'] + \
                                       total_result_dict['recall@20'] + total_result_dict['mrr@5'] + \
                                       total_result_dict['mrr@10'] + total_result_dict['mrr@20'] + \
                                       total_result_dict['ndcg@5'] + total_result_dict['ndcg@10'] + total_result_dict['ndcg@20']

            print(total_result_dict)
            fw.write(str(total_result_dict) + '\n')

            with open(save_path + 'valid_result_' + str(epoch) + '.txt', 'w') as f:
                for result in total_result:
                    f.write(str(result) + '\n')
            epoch += 1
        fw.close()


    if mode == "test":
        if resume:
            fw = open(save_path + 'test_result.txt', 'a')
        else:
            fw = open(save_path + 'test_result.txt', 'w')
        dataset = TestDataset(test_file, test_neg_file, item_num, modified_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        model = Model(item_num, layer_num, head_num, hidden_unit, dropout_rate, max_insert_num)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        next_epoch = aug_epochs + 1
        if resume:
            with open(save_path + 'test_result.txt', 'r') as f:
                content = f.readlines()
            next_epoch = len(content) + 1
            print('test from epoch %d' % (next_epoch,))
        epoch = next_epoch
        while epoch <= epochs:
            step = 0
            total_result = []
            model.load_state_dict(torch.load(save_path + 'model/TCLA-' + str(epoch) + '.pth'))
            with torch.no_grad():
                for batch in dataloader:
                    step += 1
                    input_seqs_ids, targets, negatives, neg_num = batch
                    if torch.cuda.is_available():
                        input_seqs_ids = input_seqs_ids.cuda()
                        targets = targets.cuda()
                        negatives = negatives.cuda()
                        neg_num = neg_num.cuda()

                    output = model.test_forward(input_seqs_ids)  # [batch_size, hidden_unit]
                    output = torch.matmul(output, model.item_embedding.weight.T)  # [batch_size, item_num]
                    result = evaluate_function(output, targets, negatives, neg_num)
                    total_result.extend(result)

            total_result_dict = {'epoch': epoch}
            total_result_dict['recall@5'] = get_metrics('recall@5', total_result)
            total_result_dict['recall@10'] = get_metrics('recall@10', total_result)
            total_result_dict['recall@20'] = get_metrics('recall@20', total_result)
            total_result_dict['mrr@5'] = get_metrics('mrr@5', total_result)
            total_result_dict['mrr@10'] = get_metrics('mrr@10', total_result)
            total_result_dict['mrr@20'] = get_metrics('mrr@20', total_result)
            total_result_dict['ndcg@5'] = get_metrics('ndcg@5', total_result)
            total_result_dict['ndcg@10'] = get_metrics('ndcg@10', total_result)
            total_result_dict['ndcg@20'] = get_metrics('ndcg@20', total_result)
            total_result_dict['sum'] = total_result_dict['recall@5'] + total_result_dict['recall@10'] + \
                                       total_result_dict['recall@20'] + total_result_dict['mrr@5'] + \
                                       total_result_dict['mrr@10'] + total_result_dict['mrr@20'] + \
                                       total_result_dict['ndcg@5'] + total_result_dict['ndcg@10'] + total_result_dict['ndcg@20']

            print(total_result_dict)
            fw.write(str(total_result_dict) + '\n')

            with open(save_path + 'test_result_' + str(epoch) + '.txt', 'w') as f:
                for result in total_result:
                    f.write(str(result) + '\n')
            epoch += 1
        fw.close()


