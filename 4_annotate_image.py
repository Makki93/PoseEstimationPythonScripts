def draw_skel_and_kp(img, instance_scores, keypoint_scores, keypoint_coords, min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    out_img = cv2.drawKeypoints(
        out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints,
                            isClosed=False, color=(255, 255, 0))
    return out_img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter COCO JSON: "
                                                 "Filters a COCO Instances JSON file to only include specified categories. "
                                                 "This includes images, and annotations. Does not modify 'info' or 'licenses'.")

    parser.add_argument("-img", "--image_path", dest="image_path",
                        help="path to the images")

    parser.add_argument("-i", "--input_json", dest="input_json",
                        help="path to a json file in coco format")
    parser.add_argument("-o", "--output_json", dest="output_json",
                        help="path to save the output json")

    args = parser.parse_args()

    self.image_path = Path(args.image_path)
    self.input_json_path = Path(args.input_json)
    self.output_json_path = Path(args.output_json)
    self.filter_categories = ['person']  # only filters persons
