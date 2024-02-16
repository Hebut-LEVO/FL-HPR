import enum
import numpy as np
import torch


def  evalandprint(args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter, best_changed):
    # evaluation on training data
    for client_idx in range(args.n_clients):
        train_loss, train_acc, train_iou = algclass.client_eval(
            client_idx, train_loaders[client_idx], args)
        print(
            f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train acc: {train_acc:.4f}| Train iou: {train_iou:.4f}')

    # evaluation on valid data
    val_acc_list = [None] * args.n_clients
    for client_idx in range(args.n_clients):
        val_loss, val_acc, val_iou = algclass.client_eval(
            client_idx, val_loaders[client_idx], args)
        val_acc_list[client_idx] = val_iou
        print(
            f' Site-{client_idx:02d} | Val Loss: {val_loss:.4f} | Val acc: {val_acc:.4f}| Val iou: {val_iou:.4f}')

    if np.mean(val_acc_list) > np.mean(best_acc):
        for client_idx in range(args.n_clients):
            best_acc[client_idx] = val_acc_list[client_idx]
            best_epoch = a_iter
        best_changed = True

    if best_changed:
        best_changed = False
        # test
        for client_idx in range(args.n_clients):
            test_loss, test_acc, test_iou = algclass.client_eval(
                client_idx, test_loaders[client_idx], args)
            print(
                f' Test site-{client_idx:02d} | Epoch:{best_epoch} | Test acc: {test_acc:.4f}| Test iou: {test_iou:.4f}')
            best_tacc[client_idx] = test_iou
        print(f' Saving the local and server checkpoint to {SAVE_PATH}')
        tosave = {'best_epoch': best_epoch, 'best_Iou': best_acc, 'best_tacc': np.mean(np.array(best_tacc))}
        for i,tmodel in enumerate(algclass.client_model):
            tosave['client_model_'+str(i)]=tmodel.state_dict()
        tosave['server_model']=algclass.server_model.state_dict()
        torch.save(tosave, SAVE_PATH)

    return best_acc, best_tacc, best_changed
