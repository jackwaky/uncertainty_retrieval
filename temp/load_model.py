import torch

def load_model(file_path):
    dict_of_models = torch.load(file_path)['model_state_dict']
    # print(dict_of_models['model_state_dict'].keys())

    # print(dict_of_models['layer4'])
    # print(dict_of_models['text_fc'])

    keys = dict_of_models.keys()
    for key in keys:
        self.models[key].load_state_dict(dict_of_models[key])


if __name__ == '__main__':
    file_path = './experiments/test_fashionIQ_2024-03-20_1/best.pth'
    load_model(file_path)
    exit(0)