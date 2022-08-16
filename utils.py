import torch
import torchvision.models as models

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, max_count=100):
        self.reset(max_count)

    def reset(self, max_count):
        self.val = 0
        self.avg = 0
        self.data_container = []
        self.max_count = max_count
    
    def update(self, val):
        self.val = val
        if (len(self.data_container) < self.max_count):
            self.data_container.append(val)
            self.avg = sum(self.data_container) * 1.0 / len(self.data_container)
        else:
            self.data_container.pop(0)
            self.data_container.append(val)
            self.avg = sum(self.data_container) * 1.0 / self.max_count


def save_model_optimizer_history(model, optimizer, filepath, device):
    # print("saving model and optimizer")
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, filepath)


def load_model(model, filepath, device):
    print("loading model")
    load_state = torch.load(filepath, map_location=device)

    model.load_state_dict(load_state['state_dict'])
    return model

def load_optimizer(optimizer, filepath, device):
    print("loading optimizer")
    state = torch.load(filepath)#, map_location=device)
    optimizer.load_state_dict(state['optimizer'])
    return optimizer

def getVgg_frame(pretrained=True):
    if pretrained == False:
        model = models.vgg16(pretrained=False, num_classes=101)
    else:
        model_np = models.vgg16(pretrained=False, num_classes=101)
        model_p = models.vgg16(pretrained=True)

        pretrained_dict = model_p.state_dict()
        unload_model_dict = model_np.state_dict()
        load_model = {k: v for k, v in pretrained_dict.items() if
                      (k in unload_model_dict and pretrained_dict[k].shape == unload_model_dict[k].shape)}

        print('load_model:')
        for dict_inx, (k, v) in enumerate(load_model.items()):
            print(dict_inx, k, v.shape)
        unload_model_dict.update(load_model)
        model_np.load_state_dict(unload_model_dict)
        model = model_np

    return model


def load_pretrained_model(model_path):
    model = getVgg_frame(pretrained=False)
    pretrained_dict = torch.load(model_path)
    unload_model_dict = model.state_dict()

    print('unload_model_dict:')
    for dict_inx, (k, v) in enumerate(unload_model_dict.items()):
        print(dict_inx, k, v.shape)

    load_model = {}
    for k, v in pretrained_dict.items():
        load_k = k.split('.')[1:]
        load_k = '.'.join(load_k)
        # load_k = k
        if pretrained_dict[k].shape == unload_model_dict[load_k].shape:
            load_model[load_k] = pretrained_dict[k]
    print('len of load_model:', len(load_model))
    print('pretrained_dict:')
    for dict_inx, (k, v) in enumerate(pretrained_dict.items()):
        print(dict_inx, k, v.shape)
    unload_model_dict.update(load_model)
    model.load_state_dict(unload_model_dict)

    return model

###################################### complementary functions, not used while training ################################
# load model and calculate CS score
def load_calculate_CS(model_path):
    global loader_train, loader_val
    loader_train, loader_val = load_data()

    device = set_device()
    setup_seed(RANDOM_SEED)

    vgg_model = getVgg_frame(pretrained=False)
    face_feature_model = resnet50()
    model = Vgg_net(vgg_model=vgg_model, face_feature_model=face_feature_model, device=device)
    model = nn.DataParallel(model)

    model = load_model(model, model_path, device=device)
    model = model.to(device=device)
    
    mae = AverageMeter(1500)

    model.eval()
    total = 0
    correct = 0
    all_mae = 0
    end_time = time.time()
    result_nums = [0 for i in range(1,11)]
    for batch_idx, (x224, targets) in enumerate(loader_val):
        x224 = x224.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        targets = targets.to(device=device, dtype=torch.long)
        x224, targets = Variable(x224), Variable(targets)

        outputs = outputs.view(bs, ncrops, -1).mean(1)

        output = model(x224)
        output = output.to(device=device)

        total += targets.size(0)

        softmax_layer = nn.Softmax(dim=1)
        preb = softmax_layer(output)
        preb_data = preb.cpu().data.numpy()
        target_data = targets.cpu().data.numpy()
        label_arr = np.array(range(101))
        estimate_ages = np.sum(preb_data * label_arr, axis=1)
        age_dis = abs(estimate_ages - target_data)
        for age_max in range(1,11):
          smaller_age = torch.zeros(age_dis.shape)
          smaller_age[age_dis<=age_max] = 1
          num = sum(smaller_age)
          result_nums[age_max-1] += int(num)

        batch_mae = sum(abs(estimate_ages - target_data)) * 1.0 / len(target_data)        

        mae.update(batch_mae)
        all_mae = all_mae + batch_mae * targets.size(0)

        if batch_idx % 20 == 0:
            print('Test: [%d/%d]\t'
                  'MAE %.3f (%.3f)' % (batch_idx, len(loader_val),
                                       mae.val, mae.avg))
            print('CS num:', result_nums)

    print('End MAE: %.3f' % (all_mae * 1.0 / total))
    print('CS num:', result_nums)
    result_num_radio = []
    for i in range(1,11):
      result_num_radio.append(float(result_nums[i-1])/float(total))
    print('CS:', result_num_radio)
    print('total:', total)