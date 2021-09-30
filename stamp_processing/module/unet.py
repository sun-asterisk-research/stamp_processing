from typing import List

import numpy as np
import torch
from fastai.vision.all import *


# from backend.StampRemoval.util import *
__all__ = ["CustomUnetBlock", "CustomDynamicUnet", "UnetInference"]


class CustomUnetBlock(Module):
    """A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."""

    @delegates(ConvLayer.__init__)
    def __init__(
        self,
        up_in_c,
        x_in_c,
        hook,
        final_div=True,
        blur=False,
        act_cls=defaults.activation,
        self_attention=False,
        init=nn.init.kaiming_normal_,
        norm_type=None,
        **kwargs,
    ):
        self.hook = hook
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, act_cls=act_cls, norm_type=norm_type)
        self.bn = BatchNorm(x_in_c)
        ni = up_in_c // 2 + x_in_c
        #         nf = ni if final_div else ni//2
        nf = ni // 2 if final_div else ni // 4
        self.conv1 = ConvLayer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.conv2 = ConvLayer(
            nf,
            nf,
            act_cls=act_cls,
            norm_type=norm_type,
            xtra=SelfAttention(nf) if self_attention else None,
            **kwargs,
        )
        self.relu = act_cls()
        apply_init(nn.Sequential(self.conv1, self.conv2), init)

    def forward(self, up_in):
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode="nearest")
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class CustomDynamicUnet(SequentialEx):
    """Create a U-Net from a given architecture."""

    def __init__(
        self,
        encoder,
        n_out,
        img_size,
        blur=False,
        blur_final=True,
        self_attention=False,
        y_range=None,
        last_cross=True,
        bottle=False,
        act_cls=defaults.activation,
        init=nn.init.kaiming_normal_,
        norm_type=None,
        **kwargs,
    ):
        imsize = img_size
        sizes = model_sizes(encoder, size=imsize)

        sz_chg_idxs = list(reversed(self._get_sz_change_idxs(sizes)))
        self.sfs = hook_outputs([encoder[i] for i in sz_chg_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sizes[-1][1]

        middle_conv = nn.Sequential(
            ConvLayer(ni, ni, act_cls=act_cls, norm_type=norm_type, **kwargs),
            ConvLayer(ni, ni, act_cls=act_cls, norm_type=norm_type, **kwargs),
        ).eval()
        x = middle_conv(x)
        layers = [encoder, BatchNorm(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sz_chg_idxs):
            not_final = i != len(sz_chg_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sizes[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sz_chg_idxs) - 3)
            unet_block = CustomUnetBlock(
                up_in_c,
                x_in_c,
                self.sfs[i],
                final_div=not_final,
                blur=do_blur,
                self_attention=sa,
                act_cls=act_cls,
                init=init,
                norm_type=norm_type,
                **kwargs,
            ).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sizes[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, act_cls=act_cls, norm_type=norm_type))
        layers.append(ResizeToOrig())
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(
                ResBlock(
                    1,
                    ni,
                    ni // 2 if bottle else ni,
                    act_cls=act_cls,
                    norm_type=norm_type,
                    **kwargs,
                )
            )
        layers += [ConvLayer(ni, n_out, ks=1, act_cls=None, norm_type=norm_type, **kwargs)]
        apply_init(nn.Sequential(layers[3], layers[-2]), init)
        # apply_init(nn.Sequential(layers[2]), init)
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def _get_sz_change_idxs(self, sizes):
        "Get the indexes of the layers where the size of the activation changes."
        feature_szs = [size[-1] for size in sizes]
        sz_chg_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
        return sz_chg_idxs

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()


class PerceptualLoss:
    pass


class UnetInference:
    def __init__(self, model_path):
        """Inference interface for unet model"""
        self.learn = load_learner(model_path)
        self.learn.model.eval()

    def __call__(self, image_array: str, bs: int = 1) -> List[np.ndarray]:
        """Perform forward pass and decode the prediction of Unet model

        Args:
            image_array (list): list of numpy array
            bs (int, optional): [batch size]. Defaults to 1.

        Returns:
            [list]: list of numpy array
        """
        if len(image_array) < 1:
            return []

        batches = self.__build_batches(image_array, bs=bs)
        outs = []
        with torch.no_grad():
            for b in batches:
                outs.append(self.learn.model(b))
                del b
        pil_images = self.__decode_prediction(outs)
        return pil_images

    def __decode_prediction(self, preds):
        out = []
        i2f = IntToFloatTensor()
        for pred in preds:
            img_np = i2f.decodes(pred.squeeze()).numpy()
            img_np = img_np.transpose(1, 2, 0)
            img_np = img_np.astype(np.uint8)
            out.append(img_np)
            # out.append(Image.fromarray(img_np))
            del img_np
        return out

    def __build_batches(self, image_array: list, bs=1):
        "Builds batches to skip `DataLoader` overhead"
        type_tfms = [PILImage.create]
        item_tfms = [ToTensor()]
        type_pipe = Pipeline(type_tfms)
        item_pipe = Pipeline(item_tfms)
        i2f = IntToFloatTensor()
        batches = []
        batch = []
        k = 0
        for i, im in enumerate(image_array):
            batch.append(item_pipe(type_pipe(im)))
            k += 1
            if i == len(image_array) - 1 or k == bs:
                # batches.append(torch.cat([norm(i2f(b.cuda())) for b in batch]))
                batches.append(torch.stack([i2f(b.cpu()) for b in batch], axis=0))
                batch = []
                k = 0
        return batches
