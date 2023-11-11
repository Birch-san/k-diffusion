from k_diffusion.augmentation import translate2d, scale2d

class LatentAugmentationPipeline:
    def __init__(self, a_prob=0.12, a_scale=2**0.2, a_aniso=2**0.2, a_trans=1/8, disable_all=False):
        self.a_prob = a_prob
        self.a_scale = a_scale
        self.a_aniso = a_aniso
        self.a_trans = a_trans
        self.disable_all = disable_all

    def __call__(self, image):
        h, w = image.size
        mats = [translate2d(h / 2 - 0.5, w / 2 - 0.5)]

        # x-flip
        a0 = torch.randint(2, []).float()
        mats.append(scale2d(1 - 2 * a0, 1))
        # y-flip
        do = (torch.rand([]) < self.a_prob).float()
        a1 = torch.randint(2, []).float() * do
        mats.append(scale2d(1, 1 - 2 * a1))
