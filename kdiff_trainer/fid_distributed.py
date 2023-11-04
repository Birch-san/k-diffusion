from torchmetrics.image.fid import FrechetInceptionDistance
from torch import Tensor, FloatTensor
from accelerate import Accelerator

class FIDDistributed(FrechetInceptionDistance):
    accelerator: Accelerator
    def __init__(
        self,
        accelerator: Accelerator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.accelerator = accelerator

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features: FloatTensor = self.inception(imgs)
        features = self.accelerator.gather(features)
        if self.accelerator.is_main_process:
            self.orig_dtype = features.dtype
            features = features.double()

            if features.dim() == 1:
                features = features.unsqueeze(0)
            if real:
                self.real_features_sum += features.sum(dim=0)
                self.real_features_cov_sum += features.t().mm(features)
                self.real_features_num_samples += imgs.shape[0]
            else:
                self.fake_features_sum += features.sum(dim=0)
                self.fake_features_cov_sum += features.t().mm(features)
                self.fake_features_num_samples += imgs.shape[0]
    
    def compute(self) -> Tensor:
        if not self.accelerator.is_main_process:
            raise RuntimeError('Features are only accumulated on main process, so we allow compute() only on the main process')
        return super(FrechetInceptionDistance, self).compute()