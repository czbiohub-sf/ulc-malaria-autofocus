import math
import torch


class AdaptiveLRClipping:
    """Adaptive Learning Rate Clipping

    Clip loss spikes, from small batches, high order loss functions, and poor
    data.
    """

    def __init__(self, n=3, mu1=25, mu2=30**2, beta1=0.999, beta2=0.999):
        assert mu1**2 < mu2, "mu1**2 must be less than mu2"
        assert 0 < beta1 < 1, "beta1 must be between 0 and 1"
        assert 0 < beta2 < 1, "beta2 must be between 0 and 1"

        self._n = n
        self._mu1 = mu1
        self._mu2 = mu2
        self._beta1 = beta1
        self._beta2 = beta2
        self._prev_loss: Optional[float] = None

    def clip(self, loss: torch.Tensor) -> torch.Tensor:
        # TODO: add support for neg losses - reqs careful mgmt of div by 0
        assert loss > 0, "ALRC currently only supports positive losses"

        if self._prev_loss is not None:
            self._mu1, self._mu2 = (
                self._beta1 * self._mu1 + (1 - self._beta1) * self._prev_loss,
                self._beta2 * self._mu2 + (1 - self._beta2) * self._prev_loss**2,
            )

        sigma = math.sqrt(self._mu2 - self._mu1**2)
        max_loss = self._mu1 + self._n * sigma
        if loss > max_loss:
            ldiv = max_loss / loss.item()
            dyn_loss = ldiv * loss
        else:
            dyn_loss = loss

        self._prev_loss = dyn_loss.item()
        return dyn_loss

    def state_dict(self):
        return {"n": self._n,
                "mu1": self._mu1,
                "mu2": self._mu2,
                "beta1": self._beta1,
                "beta2": self._beta2,
                "prev_loss": self._prev_loss}

    def load_state_dict(self, state_dict):
        self._n = state_dict["n"]
        self._mu1 = state_dict["mu1"]
        self._mu2 = state_dict["mu2"]
        self._beta1 = state_dict["beta1"]
        self._beta2 = state_dict["beta2"]
        self._prev_loss = state_dict["prev_loss"]
