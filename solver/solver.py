import torch
import torch.nn as nn
import numpy as np
import editdistance as ed

def LetterErrorRate(pred_y,true_y):
    ed_accumalate = []
    for p, t in zip(pred_y, true_y):
        compressed_t = [w for w in t if (w!=1 and w !=0)]
        compressed_p = []
        for p_w in p:
            if p_w == 0:
                continue
            if p_w == 1:
                break
            compressed_p.append(p_w)
        ed_accumalate.append(ed.eval(compressed_p, compressed_t) / len(compressed_t))
    return ed_accumalate

def label_smoothing_loss(pred_y, true_y, label_soomthing = 0.1):
    assert pred_y.size() == true_y.size()
    seq_len = torch.sum(torch.sum(true_y,dim=-1),dim=-1, keepdim=True)
    class_dim = true_y.size()[-1]
    smooth_y = ((1.0 - label_soomthing) * true_y +(label_soomthing / class_dim)) * torch.sum(true_y, dim=-1, keepdim=True)
    loss = -torch.mean(torch.sum((torch.sum(smooth_y * pred_y, dim=-1) / seq_len), dim=-1))
    return loss

def tensor2text(y,vocab):
    rounded_y = np.around(y).astype(int)
    rounded_y = np.array([vocab[str(n)] for n in rounded_y])
    return rounded_y

def batch_iterator(
        batch_data,
        batch_label,
        las_model,
        optimizer,
        tf_rate,
        is_training,
        max_label_len,
        label_smoothing,
        use_gpu = True,
        vocab_dict = None,
):
    label_smoothing = label_smoothing
    max_label_len = min([batch_label.size()[1], max_label_len])
    criterion = nn.NLLLoss(ignore_index=0).cuda()
    optimizer.zero_grad()
    raw_pred_seq, _ = las_model(
        batch_data = batch_data, batch_label = batch_label, teacher_force_rate = tf_rate, is_training = is_training,
    )
    pred_y = (torch.cat([torch.unsqueeze(each_y,1) for each_y in raw_pred_seq],1) [:,:max_label_len,:]).contiguous()

    if label_smoothing == 0.0 or not (is_training):
        pred_y = pred_y.permute(0,2,1)
        true_y = torch.max(batch_label, dim=2)[1][:,:max_label_len].contiguous()
        loss = criterion(pred_y, true_y)
        batch_ler = LetterErrorRate(
            torch.max(pred_y.permute(0,2,1),dim=2)[1].cpu().numpy(),
            true_y.cpu().data.numpy(),
        )
    else:
        true_y = batch_label[:, :max_label_len,:].contiguous()
        true_y = true_y.type(torch.cuda.FloatTensor) if use_gpu else true_y.type(torch.FloatTensor)
        loss = label_smoothing_loss(pred_y, true_y, label_soomthing=label_smoothing)
        batch_ler = LetterErrorRate(
            torch.max(pred_y,dim=2)[1].cpu().numpy(),
            torch.max(true_y, dim=2)[1].cpu().numpy(),
        )
    if is_training:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(las_model.parameters(),1)
        optimizer.step()

    batch_loss = loss.cpu().data.numpy()
    return batch_loss, batch_ler, true_y, pred_y, raw_pred_seq

def mmWavoice_batch_iterator(
        voice_batch_data,
        mmwave_batch_data,
        batch_label,
        mmWavoice_model,
        optimizer,
        tf_rate,
        is_training,
        max_label_len,
        label_smoothing,
        use_gpu = True,
        vocab_dict = None,
):
    label_smoothing = label_smoothing
    max_label_len = min([batch_label.size()[1], max_label_len])
    criterion = nn.NLLLoss(ignore_index=0).cuda()
    optimizer.zero_grad()
    raw_pred_seq, _ = mmWavoice_model(
        voice_batch_data = voice_batch_data, mmwave_batch_data = mmwave_batch_data, batch_label = batch_label, teacher_force_rate = tf_rate, is_training = is_training,
    )
    pred_y = (torch.cat([torch.unsqueeze(each_y, 1) for each_y in raw_pred_seq],1)[:, :max_label_len,:]).contiguous()
    if label_smoothing == 0.0 or not (is_training):
        pred_y = pred_y.permute(0,2,1)
        true_y = torch.max(batch_label, dim=2)[1][:,:max_label_len].contiguous()
        loss = criterion(pred_y, true_y)
        batch_ler  = LetterErrorRate(
            torch.max(pred_y.permute(0,2,1), dim=2)[1].cpu().numpy(), 
            true_y.cpu().data.numpy(),
        )
    else:
        true_y = batch_label[:,:max_label_len,:].contiguous()
        true_y = true_y.type(torch.cuda.FloatTensor) if use_gpu else true_y.type(torch.FloatTensor)
        loss = label_smoothing_loss(pred_y, true_y, label_soomthing=label_smoothing)
        batch_ler = LetterErrorRate(
            torch.max(pred_y, dim=2)[1].cpu().numpy(),
            torch.max(true_y, dim=2)[1].cpu().data.numpy(),
        )
    if is_training:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mmWavoice_model.parameters(),1)
        optimizer.step()

    batch_loss = loss.cpu().data.numpy()
    return batch_loss, batch_ler, true_y, pred_y, raw_pred_seq



