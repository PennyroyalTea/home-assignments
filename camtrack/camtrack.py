#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp

from corners import CornerStorage
from _camtrack import *
from _corners import filter_frame_corners

from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


def initialize_first_two_frames(corner_storage, intrinsic_mat, triangulation_parameters):
    view_mat1 = eye3x4()
    fc = len(corner_storage)  # frame count
    frame_pairs = np.argsort(np.random.rand(fc, 2 * fc), axis=0)[:2].T

    best_count = -1
    best_hmat_emat = 1
    for (frame1, frame2) in frame_pairs:
        corrs = build_correspondences(corner_storage[frame1], corner_storage[frame2])  # correspondences
        count = -1
        if len(corrs.ids) >= 5:
            emat, emat_mask = cv2.findEssentialMat(
                corrs.points_1,
                corrs.points_2,
                intrinsic_mat,
                threshold=1,
                prob=0.999,
                method=cv2.RANSAC
            )
            emat_mask = emat_mask.astype(np.bool).flatten()
            correspondences_filtered = Correspondences(corrs.ids[emat_mask],
                                                       corrs.points_1[emat_mask],
                                                       corrs.points_2[emat_mask])
            hmat, hmat_mask = cv2.findHomography(correspondences_filtered.points_1,
                                                 correspondences_filtered.points_2,
                                                 method=cv2.RANSAC,
                                                 ransacReprojThreshold=1.0,
                                                 confidence=0.999,
                                                 maxIters=10 ** 4)
            hmat_emat = np.count_nonzero(emat_mask) / np.count_nonzero(hmat_mask)

            rot_1, rot_2, translation = cv2.decomposeEssentialMat(emat)
            for rot in (rot_1.T, rot_2.T):
                for tran in (translation, -translation):
                    view = pose_to_view_mat3x4(Pose(rot, rot @ tran))
                    pt_cnt = len(triangulate_correspondences(
                        correspondences_filtered,
                        view_mat1,
                        view,
                        intrinsic_mat,
                        triangulation_parameters
                    )[1])
                    if count < pt_cnt:
                        view_2 = view
                        count = pt_cnt

        if best_count < count or (best_count < 2 * count and best_hmat_emat > hmat_emat):
            best_frame1 = frame1
            best_frame2 = frame2
            view_mat2 = view_2
            best_count = count
            best_hmat_emat = hmat_emat
    return (best_frame1, view_mat3x4_to_pose(view_mat1)), (best_frame2, view_mat3x4_to_pose(view_mat2))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) -> Tuple[List[Pose], PointCloud]:
    MX_REPROJECTION_ERROR = 4.0
    MN_TRIANG_ANGLE_DEG = 1.0
    MN_DEPTH = 0.01
    triang_params = TriangulationParameters(max_reprojection_error=MX_REPROJECTION_ERROR,
                                            min_triangulation_angle_deg=MN_TRIANG_ANGLE_DEG,
                                            min_depth=MN_DEPTH)
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    fc = len(corner_storage)

    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = initialize_first_two_frames(corner_storage, intrinsic_mat,
                                                                 triang_params)
    view_mats = [None] * fc

    frame_1, camera_pose_1 = known_view_1
    frame_2, camera_pose_2 = known_view_2
    view_mat_1 = pose_to_view_mat3x4(camera_pose_1)
    view_mat_2 = pose_to_view_mat3x4(camera_pose_2)

    view_mats[frame_1] = view_mat_1
    view_mats[frame_2] = view_mat_2

    initial_correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
    points3d, ids, _ = triangulate_correspondences(initial_correspondences,
                                                   view_mat_1, view_mat_2,
                                                   intrinsic_mat,
                                                   triang_params)

    point_cloud_builder = PointCloudBuilder(ids, points3d)

    changed = True
    processed_frames = 0
    while changed:
        changed = False
        for frame_1 in range(fc):
            if view_mats[frame_1] is not None:
                continue
            corners = corner_storage[frame_1]
            intersection, (ids_3d, ids_2d) = snp.intersect(point_cloud_builder.ids.flatten(),
                                                           corners.ids.flatten(),
                                                           indices=True)

            if len(intersection) <= 3:
                continue
            points3d = point_cloud_builder.points[ids_3d]
            points2d = corners.points[ids_2d]

            IT_CNT = 111
            CONF = 0.999
            success, r_vec, t_vec, inliers = cv2.solvePnPRansac(
                objectPoints=points3d,
                imagePoints=points2d,
                cameraMatrix=intrinsic_mat,
                distCoeffs=np.array([]),
                iterationsCount=IT_CNT,
                reprojectionError=MX_REPROJECTION_ERROR,
                confidence=CONF,
                flags=cv2.SOLVEPNP_EPNP
            )

            if not success:
                continue
            success, r_vec, t_vec = cv2.solvePnP(
                objectPoints=points3d[inliers.flatten()],
                imagePoints=points2d[inliers.flatten()],
                cameraMatrix=intrinsic_mat,
                distCoeffs=np.array([]),
                rvec=r_vec,
                tvec=t_vec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
            view_mats[frame_1] = view_mat
            processed_frames += 1
            print(f'\rProcessing frame {frame_1}, inliers: {len(inliers)}, processed {processed_frames} out'
                  f' of {fc} frames, {len(point_cloud_builder.ids)} points in cloud', end='')

            corners_1 = filter_frame_corners(corner_storage[frame_1], inliers)
            for frame_2 in range(fc):
                if view_mats[frame_2] is None:
                    continue
                corners_2 = corner_storage[frame_2]
                correspondences = build_correspondences(corners_1, corners_2, point_cloud_builder.ids)
                if len(correspondences.ids) == 0:
                    continue
                points3d, ids, _ = triangulate_correspondences(correspondences,
                                                               view_mat, view_mats[frame_2],
                                                               intrinsic_mat,
                                                               triang_params)
                changed = True
                point_cloud_builder.add_points(ids, points3d)

    print(f'\rProcessed {processed_frames + 2} out of {fc} frames, {len(point_cloud_builder.ids)} points in cloud')

    first_processed_view_mat = next((view_mat for view_mat in view_mats if view_mat is not None), None)
    if first_processed_view_mat is None:
        print('\rFailed to solve scene')
        exit(0)

    view_mats[0] = first_processed_view_mat
    for i in range(1, len(view_mats)):
        if view_mats[i] is None:
            view_mats[i] = view_mats[i - 1]

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )

    poses = list(map(view_mat3x4_to_pose, view_mats))
    point_cloud = point_cloud_builder.build_point_cloud()
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
