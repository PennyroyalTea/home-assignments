#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def reformat(image):
    return np.around(image * 255).astype(dtype=np.uint8)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    corners_sz = 14

    frame_sequence = map(reformat, frame_sequence)

    # get corners for first frame
    img = next(frame_sequence)
    points = cv2.goodFeaturesToTrack(img, 5000, 0.004, corners_sz // 2).reshape((-1, 2))
    mx_id = len(points)
    ids = np.arange(mx_id)

    corners = FrameCorners(
        ids,
        points,
        np.full(len(points), corners_sz)
    )

    builder.set_corners_at_frame(0, corners)

    # get corners for other frames
    for frame_id, img_cur in enumerate(frame_sequence, 1):
        # calculate motion for existing points
        points_cur, st, errs = cv2.calcOpticalFlowPyrLK(
            prevImg=img,
            nextImg=img_cur,
            prevPts=points,
            nextPts=None,
            winSize=(50, 50),
            maxLevel=3,
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            criteria=(
                cv2.TermCriteria_COUNT | cv2.TermCriteria_EPS,
                20,
                0.03
            )
        )

        # remove dead points
        alive = st.flatten() == 1
        points_cur = points_cur[alive]
        errs = errs.flatten()[alive]
        ids = ids[alive]

        # remove low quality points
        perm = np.argsort(errs.flatten())

        points_in_order = points_cur[perm]
        errs_in_order = errs[perm]
        ids_in_order = ids[perm]

        best_quality = errs_in_order[-1]

        alive_min_distance = np.array(
            [np.count_nonzero(np.linalg.norm(points_cur - point, axis=1) <= corners_sz // 2) == 1
             for point, err in zip(points_in_order, errs_in_order)
             ]
        )
        alive_big_error = (errs_in_order > best_quality * 0.01)

        points_cur = points_in_order[alive_min_distance & alive_big_error]
        ids = ids_in_order[alive_min_distance & alive_big_error]
        errs = errs_in_order[alive_min_distance & alive_big_error]

        # add new corners (with better quality)

        new_points_candidates = cv2.goodFeaturesToTrack(img, 10000, 0.01, corners_sz // 2).reshape((-1, 2))
        new_points = [point for point in new_points_candidates if np.all(np.linalg.norm(points_cur - point, axis=1) > corners_sz // 2)]
        new_points = np.array(new_points).reshape((-1, 2))

        if new_points.shape[0] > 0:
            points_cur = np.concatenate((points_cur, new_points))
            ids = np.concatenate((ids, np.arange(mx_id, mx_id + new_points.shape[0])))
            mx_id += new_points.shape[0]

        corners = FrameCorners(
            ids,
            points_cur,
            np.full(len(points_cur), corners_sz)
        )
        builder.set_corners_at_frame(frame_id, corners)

        img = img_cur
        points = points_cur


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
