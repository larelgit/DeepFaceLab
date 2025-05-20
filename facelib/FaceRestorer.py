from pathlib import Path
import numpy as np

class FaceRestorer:
    """Optional face restoration using GFPGAN.

    This wrapper integrates the `gfpgan` package if it is available. If the
    package or model weights are missing, an informative error is raised.
    """
    def __init__(self, model_path=None, upscale=1):
        try:
            from gfpgan import GFPGANer
        except ImportError as e:
            raise ImportError(
                "FaceRestorer requires the gfpgan package. Install it to use this feature."
            ) from e

        if model_path is None:
            model_path = Path(__file__).parent / "GFPGANv1.3.pth"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Unable to load GFPGAN model weights at {model_path}."
            )

        self.restorer = GFPGANer(model_path=str(model_path), upscale=upscale)

    def restore(self, img_bgr):
        """Restore an image using GFPGAN.

        Parameters
        ----------
        img_bgr : np.ndarray
            Image in BGR format with values in [0, 255].
        Returns
        -------
        np.ndarray
            Restored image in BGR format with values in [0, 255].
        """
        _, _, restored_img = self.restorer.enhance(
            img_bgr, has_aligned=False, only_center_face=False, paste_back=True
        )
        return restored_img
