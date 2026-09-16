# Space-Tracker MOT release, non-car split, in the DanceTrack on-disk layout.
#
# Upstream: MOTIP -- https://github.com/MCG-NJU/MOTIP, commit ``ffc0e90``. This
# tree is that clone; its upstream history is parked at ``MOTIP/.git-upstream``,
# so ``git --git-dir=.git-upstream diff`` lists every change we made:
#
#   * data/joint_dataset.py -- register this dataset.
#   * data/transforms.py    -- call torchvision v2's ``make_params`` /
#                              ``transform`` when they exist and the old
#                              ``_get_params`` / ``_transform`` otherwise.
#                              MOTIP targets torchvision 0.19, which renamed
#                              both in 0.20; the shim runs on either rather than
#                              pinning the whole environment for one method.
#   * submit_and_evaluate.py -- take the MOTChallenge branch for SpaceTracker,
#                              whose output format is identical to DanceTrack's.
#   * data/spacetracker.py, configs/motip_spacetracker_nocar.yaml -- ours.
#
# The paper declares one further departure for MOTIP: a shortened training
# schedule, which is set in configs/motip_spacetracker_nocar.yaml rather than in
# code. Weights (``pretrains/``) and the symlinked frame tree (``datasets/``)
# are not committed -- see the repository .gitignore.

import os

from .dancetrack import DanceTrack


class SpaceTracker(DanceTrack):
    """
    Same tree as DanceTrack (`<split>/<seq>/{img1,gt}`), two differences:

    * frames are 8-digit `%08d` and may be .jpg or .png, because the release
      mixes both and the tree is symlinked straight from the MOTRv2 export
      rather than re-encoded;
    * a single foreground class -- `train` has no val/test ground truth in the
      release, so the detector was built 2-class and MOTIP, like MOTRv2, is run
      class-agnostically.
    """

    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "SpaceTracker",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super(SpaceTracker, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )
        return

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        base = os.path.join(sequence_dir, "img1", f"{frame_idx + 1:08d}")
        for ext in (".jpg", ".png"):
            if os.path.exists(base + ext):
                return base + ext
        return base + ".jpg"
