from network.models import lenet5v, DGCNN_partseg
import copy


def modelsel(args, device):
    if args.dataset in ['vlcs', 'pacs', 'off_home', 'off-cal', 'covid']:
        #train_dataset = DGCNN_partseg(partition='trainval', num_points=args.num_points, class_choice=args.class_choice)
        server_model = lenet5v().to(device)
    elif 'medmnist' in args.dataset:
        server_model = lenet5v().to(device)
    elif 'ShapeNetPart' in args.dataset:
        server_model = DGCNN_partseg(args, 10).to(device)

    client_weights = [1/args.n_clients for _ in range(args.n_clients)]
    models = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_clients)]
    return server_model, models, client_weights
