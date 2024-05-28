import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pytorchlab.models.yoloVID.darknet import CSPDarknet
from src.pytorchlab.models.yoloVID.network_blocks import BaseConv, CSPLayer, DWConv
from src.pytorchlab.utils.MemMatYOLOX.utils import conv1x1, conv3x3


def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x


class CSPBackbone(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark1", "dark2", "dark3", "dark4", "dark5"),
            in_channels=[64, 128, 256, 512, 1024],
            depthwise=False,
            act="silu",

    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.incidence = {33: [34, 43, 44, 22], 34: [35], 43: [53], 44: [45, 55, 54], 22: [11, 32, 23], 11: [12, 21],
                          32: [42], 23: [24], 12: [13], 21: [31], 42: [52], 24: [25], 13: [14], 31: [41], 14: [15],
                          41: [51]}
        # Visited tracks the set of nodes that need to visited
        self.visited = set()

        self.layers = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        # keeps tracks the nodes that need to be returned (-1 in the layer_ranges implies that node is not visited)
        self.keeps = set()

        for i, l in enumerate(self.layers):
            for j, e in enumerate(l):
                if e != -1:
                    self.keeps.add((j + 1) * 10 + (i + 1))

        def _bfs(graph, start, end):
            queue = []
            queue.append([start])
            while queue:
                path = queue.pop(0)
                node = path[-1]
                if node == end:
                    return path
                for n in graph.get(node, []):
                    new_path = list(path)
                    new_path.append(n)
                    queue.append(new_path)

        _keeps = self.keeps.copy()

        while _keeps:
            node = _keeps.pop()
            vs = set(_bfs(self.incidence, 33, node))  # for us start is at 33  as that's the first node
            self.visited = vs | self.visited
            _keeps = _keeps - self.visited

        # applied in a pyramid
        self.pyramid_transformation_1 = conv1x1(int(in_channels[0] * width), 256)
        self.pyramid_transformation_2 = conv1x1(int(in_channels[1] * width), 256)
        self.pyramid_transformation_3 = conv1x1(int(in_channels[2] * width), 256)
        self.pyramid_transformation_4 = conv1x1(int(in_channels[3] * width), 256)
        self.pyramid_transformation_5 = conv1x1(int(in_channels[4] * width), 256)


        # applied after upsampling
        self.upsample_transform_1 = conv3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3(256, 256, padding=1)
        self.downsample_transform_1=conv3x3(256, 256, padding=1)
        self.downsample_transform_2=conv3x3(256, 256, padding=1)

        self.downsample_transformation_12 = conv3x3(256, 256, padding=1, stride=(1, 2))
        self.downsample_transformation_21 = conv3x3(256, 256, padding=1, stride=(2, 1))

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        height, width = scaled_feature.size()[2:]
        return F.interpolate(original_feature, scale_factor=scale_factor)[:, :, :height, :width]

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [d1, d2, d3, d4, d5] = features

        _dict = {}

        if 33 in self.visited:
            _dict[33] = self.pyramid_transformation_5(d3)
        if 33 in self.visited and 22 in self.visited:
            upsampled_feature_d2 = self._upsample(_dict[33], d2)
        if 22 in self.visited:
            _dict[22] = self.upsample_transform_1(torch.add(upsampled_feature_d2, self.pyramid_transformation_2(d2)))
        if 22 in self.visited and 11 in self.visited:
            upsampled_feature_d1 = self._upsample(_dict[22], d1)
        if 11 in self.visited:
            _dict[11] = self.upsample_transform_2(torch.add(upsampled_feature_d1, self.pyramid_transformation_1(d1)))
        if 33 in self.visited and 44 in self.visited:
            downsampled_feature_d4 = self._upsample(_dict[33], d4, scale_factor=0.5)
        if 44 in self.visited:
            _dict[44] = self.downsample_transform_1(torch.add(downsampled_feature_d4, self.pyramid_transformation_4(d4)))
        if 44 in self.visited and 55 in self.visited:
            downsampled_feature_d5 = self._upsample(_dict[44], d5, scale_factor=0.5)
        if 55 in self.visited:
            _dict[55] = self.downsample_transform_2(torch.add(downsampled_feature_d5, self.pyramid_transformation_5(d5)))


        if 12 in self.visited:
            _dict[12] = self.downsample_transformation_12(_dict[11])
        if 13 in self.visited:
            _dict[13] = self.downsample_transformation_12(_dict[12])
        if 14 in self.visited:
            _dict[14] = self.downsample_transformation_12(_dict[13])
        if 15 in self.visited:
            _dict[15] = self.downsamole_transformation_12(_dict[14])

        if 21 in self.visited:
            _dict[21] = self.downsample_transformation_21(_dict[11])
        if 31 in self.visited:
            _dict[31] = self.downsample_transformation_21(_dict[21])
        if 41 in self.visited:
            _dict[41] = self.downsample_transformation_21(_dict[31])
        if 51 in self.visited:
            _dict[51] = self.downsample_transformation_21(_dict[41])

        if 23 in self.visited:
            _dict[23] = self.downsample_transformation_12(_dict[22])
        if 24 in self.visited:
            _dict[24] = self.downsample_transformation_12(_dict[23])
        if 25 in self.visited:
            _dict[25] = self.downsample_transformation_12(_dict[24])

        if 32 in self.visited:
            _dict[32] = self.downsample_transformation_21(_dict[22])
        if 42 in self.visited:
            _dict[42] = self.downsample_transformation_21(_dict[32])
        if 52 in self.visited:
            _dict[52] = self.downsample_transformation_21(_dict[42])

        if 34 in self.visited:
            _dict[34] = self.downsample_transformation_12(_dict[33])
        if 35 in self.visited:
            _dict[35] = self.downsample_transformation_12(_dict[34])

        if 43 in self.visited:
            _dict[43] = self.downsample_transformation_21(_dict[33])
        if 53 in self.visited:
            _dict[53] = self.downsample_transformation_21(_dict[43])

        if 45 in self.visited:
            _dict[45] = self.downsample_transformation_12(_dict[44])
        if 54 in self.visited:
            _dict[54] = self.downsample_transformation_21(_dict[44])


        return _dict
