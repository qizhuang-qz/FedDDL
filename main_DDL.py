import json
import torch.optim as optim
import argparse
import logging
import copy
import datetime
# from utils import *
from nico_tied import *
from resnet import *
from util import *
from loss import PrototypeContrastiveLoss
import ipdb
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='NICO_Animal', help='dataset used for training')
    parser.add_argument('--mode', type=str, default='F7', help='F7: first 7 training; L7: last 7 training')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=7, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=1000, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/DDL/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--mu_x', type=float, default=0.6, help='the weight of original image')
    # parser.add_argument('--mu_back', type=float, default=0.2, help='the weight of background')
    parser.add_argument('--re_bs', type=float, default=64, help='the mu parameter for fedprox or moon')
    # parser.add_argument('--muavg', type=float, default=0.1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--gama', type=float, default=0.1, help='the gama parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100,
                        help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--experiment', type=str, default='both_update',
                        help='both_update/single_update/shared/fpl/dafkd/fediir')
    args = parser.parse_args()
    return args


def init_nets(n_parties: object, args: object, device: object = 'cuda') -> object:
    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        net = resnet18(args.dataset, kernel_size=7)
        nets[net_i] = net.cuda()
    
    return nets


def train_net(net_id, net, global_net, train_dataloader, glo_protos, epochs, lr, args_optimizer, args, round_i, device="cpu"):
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    PCLoss = PrototypeContrastiveLoss(temperature=args.temperature)
    cnt = 0
    
    for epoch in range(epochs):
        epoch_loss1_collector  = []
        epoch_loss2_collector  = []
        for batch_idx, (x, obj, fusion_img, target) in enumerate(train_dataloader):  # train_img,train_back,train_label
            x, target = x.cuda(), target.cuda()  
            fusion_img = fusion_img.cuda()
            # ipdb.set_trace()
            optimizer.zero_grad()
            target = target.long()

            feats, out = net(x)

            loss1 = criterion(out, target)
            
            if round_i >= 2:
                glo_protos = glo_protos.cuda()
                feats_mix, _ = net(fusion_img)
                feats = torch.cat([feats, feats_mix], dim=0)
                target = torch.cat([target, target], dim=0)
                loss2 = args.mu * PCLoss(feats, target, glo_protos)
            else:
                loss2 = torch.tensor(0)
            
            loss = loss1 + loss2
            
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)

        logger.info('Epoch: %d  Loss1: %f  Loss2: %f ' % (epoch, epoch_loss1, epoch_loss2))


    # global_model.load_state_dict(global_model_state_dict)
    net.to('cpu')
    # net_frame.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0


def local_train_net(nets, global_model, args, train_images, train_objectimages, confounders, train_labels, round_i, glo_proto, device="cpu"):  # confounds,
    avg_acc = 0.0
    acc_list = []

    n_epoch = args.epochs
    
    ANCHORS = []
    ANCHORS_labels = []
    
    for net_id, net in nets.items():
        train_img = train_images[net_id]  #
        confounder = confounders[net_id]
        train_object = train_objectimages[net_id]
        train_label = train_labels[net_id]  #
        dataset_train = NICO_dataset_F(train_img, train_object, confounder, train_label)  # 对图像处理
        train_dl = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4)


        trainacc, testacc = train_net(net_id, net, global_model, train_dl, glo_proto, n_epoch, 
                                      args.lr, args.optimizer, args, round_i,
                                      device=device)
        
        anchors, anchors_labels = gen_proto_local(global_model, train_dl, n_class=n_classes)
        
        ANCHORS.append(anchors)
        ANCHORS_labels.append(anchors_labels)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))  
        
    ANCHORS = torch.cat(ANCHORS, dim=0)
    ANCHORS_labels = torch.cat(ANCHORS_labels, dim=0)
    
    return nets, ANCHORS, ANCHORS_labels


if __name__ == '__main__':
    args = get_args()
    args.logdir = args.logdir + args.dataset + '/' + args.mode + '/'
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
      
    # ready for the models
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]

    # party_list_rounds中每个元素内包含的是每一轮需要进行参与的clients的索引
    party_list_rounds = []
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

    # confounds = build_confounds()
    
    train_dl = None
    if args.dataset == 'color_mnist': #######################################################################3
        train_images, train_backimages, train_labels = get_color_mnist_dataloader(10)
        test_images,test_labels = get_color_mnist_test_dataloader(10)

        
    elif args.dataset == 'NICO_Animal':

        train_images, train_objimages, train_backimages, train_labels = build_client(7, "../dino/NICO_Contextual", 10, mode=args.mode)
        test_images, test_labels = make_test("../Datasets/NICO", 10, 10, 7, transform=transforms.Compose([
             transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                             std=[0.21851876, 0.2175944, 0.22552039])
        ]), mode=args.mode)
        test_images = torch.stack(test_images, dim=0)
        test_labels = torch.tensor(test_labels)
        dataset_test = NICO_dataset(test_images, test_labels)
        test_dl = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=4) 
        
        confounders = build_confounds(args.dataset, args.mode)
    elif args.dataset == 'NICO_Vehicle':
#         # NICO_Vehicle
        train_images, train_objimages, train_backimages, train_labels = build_client(7, "../dino/Vehicle_Contextual", 9, mode=args.mode)
        test_images, test_labels = make_test("../Datasets/Vehicle", 9, 10, 7, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                             std=[0.21851876, 0.2175944, 0.22552039])
        ]), mode=args.mode)
        test_images = torch.stack(test_images, dim=0)
        test_labels = torch.tensor(test_labels)
        dataset_test = NICO_dataset(test_images, test_labels)
        test_dl = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=4)

        confounders = build_confounds(args.dataset, args.mode)
        
    n_classes = 10
    logger.info("Initializing nets")
    # 初始化模型
    nets = init_nets(args.n_parties, args, device='cpu')

    global_models = init_nets(1, args, device='cpu')

    global_model = global_models[0]

    n_comm_rounds = args.comm_round

    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    glo_proto = None
    for round_i in range(n_comm_rounds):
        logger.info("in comm round:" + str(round_i))
        party_list_this_round = party_list_rounds[round_i]

        global_model.eval()
        global_w = global_model.state_dict()
        for param in global_model.parameters():
            param.requires_grad = False
        # global_w_frame=global_frame.state_dict()

        nets_this_round = {k: nets[k] for k in party_list_this_round}
        # nets_frame_round = {k: nets_frame[k] for k in party_list_this_round}

        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        nets_this_round, ANCHORS, ANCHORS_labels = local_train_net(nets_this_round, global_model, args, train_images, train_objimages, confounders, train_labels, round_i, glo_proto)  
#         ipdb.set_trace()
        if args.dataset == 'color_mnist': #######################################################################3
            if args.beta == 0.1:
                fed_avg_freqs = [7471 / 60000, 6885 / 60000, 6423 / 60000, 516 / 60000, 3078 / 60000, 6507 / 60000, 11146 / 60000, 6153/60000, 3964/60000, 7857/60000]
            elif args.beta == 0.5:
                fed_avg_freqs = [5473 / 60000, 6055 / 60000, 6477 / 60000, 6271 / 60000, 3285 / 60000, 7469 / 60000, 8499 / 60000, 6337/60000, 6722/60000, 3412/60000]
                
        elif args.dataset == 'NICO_Animal':
            # NICO_Animal
            if args.mode == 'F7':
                fed_avg_freqs = [1714 / 10633, 1448 / 10633, 1463 / 10633, 1503 / 10633, 1603 / 10633, 1526 / 10633, 1376 / 10633]
            elif args.mode == 'L7':
                fed_avg_freqs = [1472 / 8311, 1571 / 8311, 1511 / 8311, 1352 / 8311, 743 / 8311, 757 / 8311, 905 / 8311]
                
        elif args.dataset == 'NICO_Vehicle':
            # NICO_Vehicle
            if args.mode == 'F7':
                fed_avg_freqs = [1060 / 8027, 1277 / 8027, 1060 / 8027, 1180 / 8027, 1351 / 8027, 954 / 8027, 1145 / 8027]
            elif args.mode == 'L7':
                fed_avg_freqs = [1136 / 8352, 1207 / 8352, 1028 / 8352, 1291 / 8352, 1110 / 8352, 1473 / 8352, 1107 / 8352]
        
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id]
        
        global_model.load_state_dict(global_w)
        
        glo_proto = gen_proto_global(ANCHORS, ANCHORS_labels, n_classes)
        
        
        
        logger.info('global n_test: %d' % len(test_dl))
        global_model.cuda()

        test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl,args, get_confusion_matrix=True, device=device)

        logger.info('>> Global Model Test accuracy: %f' % test_acc)



#         mkdirs(args.modeldir + 'mi_plan4/' + args.dataset + '/' + argument_path + '/' + str(round_i))
#         global_model.to('cpu')
#         torch.save(global_model.state_dict(),
#                    args.modeldir + 'mi_plan4/' + args.dataset + '/' + argument_path + '/' + str(round_i) + '/global_model.pth')
#         for i in range(7):
#             torch.save(nets[i].state_dict(), args.modeldir + 'mi_plan4/' + args.dataset + '/' + argument_path + '/' + str(round_i) + '/local_' + str(i) + '.pth')