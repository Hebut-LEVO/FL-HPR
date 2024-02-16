# coding=utf-8
from alg.fedavg import fedavg
from util.traineval import train, train_prox


class fedprox(fedavg):
    def __init__(self, args):
        super(fedprox, self).__init__(args)

    def client_train(self, args, c_idx, test_dataloaders, train_dataloader, round, io):
        if round > 0:
            train_loss, train_iou, train_acc = train_prox(
                self.args, self.client_model[c_idx], self.server_model, test_dataloaders, train_dataloader,
                self.optimizers[c_idx], self.loss_fun, self.args.device, io)
        else:
            train_loss, train_iou, train_acc = train(
                args, self.client_model[c_idx], test_dataloaders, train_dataloader, self.optimizers[c_idx],
                self.loss_fun, self.args.device, io)
        return train_loss, train_acc, train_iou
