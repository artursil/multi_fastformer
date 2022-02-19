
from transformers import BertConfig
config = BertConfig.from_json_file("config.json")


def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


model = Model(config)
import torch.optim as optim

optimizer = optim.Adam([{"params": model.parameters(), "lr": 0.0001}])
model.cuda()


for epoch in range(4):
    loss = 0.0
    accuary = 0.0
    for cnt in range(len(train_index) // 64):

        log_ids = news_words[train_index][cnt * 64 : cnt * 64 + 64, :256]
        targets = label[train_index][cnt * 64 : cnt * 64 + 64]

        log_ids = torch.LongTensor(log_ids).cuda(non_blocking=True)
        targets = torch.LongTensor(targets).cuda(non_blocking=True)
        bz_loss, y_hat = model(log_ids, targets)
        loss += bz_loss.data.float()
        accuary += acc(targets, y_hat)
        unified_loss = bz_loss
        optimizer.zero_grad()
        unified_loss.backward()
        optimizer.step()

        if cnt % 100 == 0:
            print(
                " Ed: {}, train_loss: {:.5f}, acc: {:.5f}".format(
                    cnt * 64, loss.data / (cnt + 1), accuary / (cnt + 1)
                )
            )
    model.eval()
    allpred = []
    for cnt in range(len(test_index) // 64 + 1):

        log_ids = news_words[test_index][cnt * 64 : cnt * 64 + 64, :256]
        targets = label[test_index][cnt * 64 : cnt * 64 + 64]
        log_ids = torch.LongTensor(log_ids).cuda(non_blocking=True)
        targets = torch.LongTensor(targets).cuda(non_blocking=True)

        bz_loss2, y_hat2 = model(log_ids, targets)
        allpred += y_hat2.to("cpu").detach().numpy().tolist()

    y_pred = np.argmax(allpred, axis=-1)
    y_true = label[test_index]
    from sklearn.metrics import *

    print(accuracy_score(y_true, y_pred))
    model.train()
