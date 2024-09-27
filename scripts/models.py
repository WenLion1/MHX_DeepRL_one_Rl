import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def model_type(model_name: str, input_size: int, hidden_size: int, num_layers: int, batch_first: bool):
    """
    选择模型类型，rnn、lstm还是gru

    :param batch_first: 批次数是否在第一个位置
    :param num_layers: 一次输出中间的层数
    :param hidden_size: 隐藏层维度大小
    :param input_size: 输入维度大小
    :param model_name: rnn、lstm、gru
    :return: 具体的模型
    """

    model = None
    if model_name == "rnn":
        model = nn.RNN(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=batch_first, )
    elif model_name == "lstm":
        model = nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=batch_first, )
    elif model_name == "gru":
        model = nn.GRU(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=batch_first, )
    elif model_name == "resnet50":
        # 加载预训练的ResNet50模型
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # 替换最后的全连接层以适应新的输出类数
        model.fc = nn.Linear(model.fc.in_features, hidden_size)
        # 冻结前面的层（如果需要）
        for param in model.parameters():
            param.requires_grad = False
        # 只训练最后一层（可选）
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        print("模型加载出现错误!")

    return model


def hidden_activation_function(activation_func_name: str):
    """
    选择激活函数类型

    :param activation_func_name: 激活函数名称
    :return: 具体的激活函数
    """

    funcs = dict(relu=nn.ReLU(),
                 selu=nn.SELU(),
                 elu=nn.ELU(),
                 celu=nn.CELU(),
                 gelu=nn.GELU(),
                 silu=nn.SiLU(),
                 sigmoid=nn.Sigmoid(),
                 tanh=nn.Tanh(),
                 linear=None,
                 leaky_relu=nn.LeakyReLU(),
                 hardshrink=nn.Hardshrink(lambd=.1),
                 softshrink=nn.Softshrink(lambd=.1),
                 tanhshrink=nn.Tanhshrink(),
                 )
    return funcs[activation_func_name]


class perceptual_net(nn.Module):
    """
    主网络模型
    """

    def __init__(self,
                 device="cpu",
                 input_size=2 * 128 * 128,
                 hidden_size=128,
                 num_layers=1,
                 model_name="rnn",
                 activation_func_name="relu",
                 batch_first=True,
                 image_channel=1, ):
        super(perceptual_net, self).__init__()

        torch.manual_seed(20010509)
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = model_name
        self.activation_func_name = activation_func_name
        self.batch_first = batch_first
        self.image_channel = image_channel

        # 定义模型
        self.model = model_type(model_name=self.model_name,
                                input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=self.batch_first, ).to(self.device)

        self.fc_rule = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                               out_features=2, ),
                                     nn.Softmax(dim=-1), ).to(self.device)
        self.fc_action = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                                 out_features=2, ),
                                       nn.Softmax(dim=-1), ).to(self.device)
        self.fc_angle = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                                out_features=2, ),
                                      nn.Softmax(dim=-1), ).to(self.device)
        self.fc_zoomin_rule = nn.Sequential(nn.Linear(in_features=2,
                                                      out_features=1024, )).to(self.device)

    def forward(self, image_input):

        out, _ = self.model(image_input)  # out: (batch_size, time_length, hidden_size)
        # out = out[:, -1, :]  # 获取最后一个时间步的数据即可

        # 输出
        rule_prediction = self.fc_rule(out)
        action_prediction = self.fc_action(out)
        angle_prediction = self.fc_angle(out)

        return rule_prediction, action_prediction, angle_prediction, out


if __name__ == "__main__":
    rnn = perceptual_net()
    x = torch.rand(8, 1, 128, 128)
    y = torch.rand(8, 1, 128, 128)

    combined = torch.cat((x, y), dim=1)  # combined: (8, 2, 224, 224)
    rnn_input = combined.view(combined.size(0), 1, -1)  # rnn_input: (8, 1, 2*224*224)

    rule_pre, angle_pre, out = rnn(rnn_input)
    print("rule_pre: ", rule_pre)
    print("angle_pre: ", angle_pre)
