dataset_info = dict(
    dataset_name="coco_wholebody",
    paper_info=dict(
        author="Jin, Sheng and Xu, Lumin and Xu, Jin and "
        "Wang, Can and Liu, Wentao and "
        "Qian, Chen and Ouyang, Wanli and Luo, Ping",
        title="Whole-Body Human Pose Estimation in the Wild",
        container="Proceedings of the European " "Conference on Computer Vision (ECCV)",
        year="2020",
        homepage="https://github.com/jin-s13/COCO-WholeBody/",
    ),
    keypoint_info={
        0: dict(name="nose", id=0, color=[51, 153, 255], type="upper", swap=""),
        1: dict(
            name="left_eye", id=1, color=[51, 153, 255], type="upper", swap="right_eye"
        ),
        2: dict(
            name="right_eye", id=2, color=[51, 153, 255], type="upper", swap="left_eye"
        ),
        3: dict(
            name="left_ear", id=3, color=[51, 153, 255], type="upper", swap="right_ear"
        ),
        4: dict(
            name="right_ear", id=4, color=[51, 153, 255], type="upper", swap="left_ear"
        ),
        5: dict(
            name="left_shoulder",
            id=5,
            color=[0, 255, 0],
            type="upper",
            swap="right_shoulder",
        ),
        6: dict(
            name="right_shoulder",
            id=6,
            color=[255, 128, 0],
            type="upper",
            swap="left_shoulder",
        ),
        7: dict(
            name="left_elbow", id=7, color=[0, 255, 0], type="upper", swap="right_elbow"
        ),
        8: dict(
            name="right_elbow",
            id=8,
            color=[255, 128, 0],
            type="upper",
            swap="left_elbow",
        ),
        9: dict(
            name="left_wrist", id=9, color=[0, 255, 0], type="upper", swap="right_wrist"
        ),
        10: dict(
            name="right_wrist",
            id=10,
            color=[255, 128, 0],
            type="upper",
            swap="left_wrist",
        ),
        11: dict(
            name="left_hip", id=11, color=[0, 255, 0], type="lower", swap="right_hip"
        ),
        12: dict(
            name="right_hip", id=12, color=[255, 128, 0], type="lower", swap="left_hip"
        ),
        13: dict(
            name="left_knee", id=13, color=[0, 255, 0], type="lower", swap="right_knee"
        ),
        14: dict(
            name="right_knee",
            id=14,
            color=[255, 128, 0],
            type="lower",
            swap="left_knee",
        ),
        15: dict(
            name="left_ankle",
            id=15,
            color=[0, 255, 0],
            type="lower",
            swap="right_ankle",
        ),
        16: dict(
            name="right_ankle",
            id=16,
            color=[255, 128, 0],
            type="lower",
            swap="left_ankle",
        ),
        17: dict(
            name="left_thumb4", id=17, color=[255, 128, 0], type="", swap="right_thumb4"
        ),
        18: dict(
            name="left_forefinger4",
            id=18,
            color=[255, 153, 255],
            type="",
            swap="right_forefinger4",
        ),
        19: dict(
            name="left_middle_finger4",
            id=19,
            color=[102, 178, 255],
            type="",
            swap="right_middle_finger4",
        ),
        20: dict(
            name="left_ring_finger4",
            id=20,
            color=[255, 51, 51],
            type="",
            swap="right_ring_finger4",
        ),
        21: dict(
            name="left_pinky_finger4",
            id=21,
            color=[0, 255, 0],
            type="",
            swap="right_pinky_finger4",
        ),
        22: dict(
            name="right_thumb4", id=22, color=[255, 128, 0], type="", swap="left_thumb4"
        ),
        23: dict(
            name="right_forefinger4",
            id=23,
            color=[255, 153, 255],
            type="",
            swap="left_forefinger4",
        ),
        24: dict(
            name="right_middle_finger4",
            id=24,
            color=[102, 178, 255],
            type="",
            swap="left_middle_finger4",
        ),
        25: dict(
            name="right_ring_finger4",
            id=25,
            color=[255, 51, 51],
            type="",
            swap="left_ring_finger4",
        ),
        26: dict(
            name="right_pinky_finger4",
            id=26,
            color=[0, 255, 0],
            type="",
            swap="left_pinky_finger4",
        ),
    },
    skeleton_info={
        0: dict(link=("left_ankle", "left_knee"), id=0, color=[0, 255, 0]),
        1: dict(link=("left_knee", "left_hip"), id=1, color=[0, 255, 0]),
        2: dict(link=("right_ankle", "right_knee"), id=2, color=[255, 128, 0]),
        3: dict(link=("right_knee", "right_hip"), id=3, color=[255, 128, 0]),
        4: dict(link=("left_hip", "right_hip"), id=4, color=[51, 153, 255]),
        5: dict(link=("left_shoulder", "left_hip"), id=5, color=[51, 153, 255]),
        6: dict(link=("right_shoulder", "right_hip"), id=6, color=[51, 153, 255]),
        7: dict(link=("left_shoulder", "right_shoulder"), id=7, color=[51, 153, 255]),
        8: dict(link=("left_shoulder", "left_elbow"), id=8, color=[0, 255, 0]),
        9: dict(link=("right_shoulder", "right_elbow"), id=9, color=[255, 128, 0]),
        10: dict(link=("left_elbow", "left_wrist"), id=10, color=[0, 255, 0]),
        11: dict(link=("right_elbow", "right_wrist"), id=11, color=[255, 128, 0]),
        12: dict(link=("left_eye", "right_eye"), id=12, color=[51, 153, 255]),
        13: dict(link=("nose", "left_eye"), id=13, color=[51, 153, 255]),
        14: dict(link=("nose", "right_eye"), id=14, color=[51, 153, 255]),
        15: dict(link=("left_eye", "left_ear"), id=15, color=[51, 153, 255]),
        16: dict(link=("right_eye", "right_ear"), id=16, color=[51, 153, 255]),
        17: dict(link=("left_ear", "left_shoulder"), id=17, color=[51, 153, 255]),
        18: dict(link=("right_ear", "right_shoulder"), id=18, color=[51, 153, 255]),
        19: dict(link=("left_wrist", "left_thumb4"), id=19, color=[255, 128, 0]),
        20: dict(link=("left_wrist", "left_forefinger4"), id=20, color=[255, 153, 255]),
        21: dict(
            link=("left_wrist", "left_middle_finger4"), id=21, color=[102, 178, 255]
        ),
        22: dict(link=("left_wrist", "left_ring_finger4"), id=22, color=[255, 51, 51]),
        23: dict(link=("left_wrist", "left_pinky_finger4"), id=23, color=[0, 255, 0]),
        24: dict(link=("right_wrist", "right_thumb4"), id=24, color=[255, 128, 0]),
        25: dict(
            link=("right_wrist", "right_forefinger4"), id=25, color=[255, 153, 255]
        ),
        26: dict(
            link=("right_wrist", "right_middle_finger4"), id=26, color=[102, 178, 255]
        ),
        27: dict(
            link=("right_wrist", "right_ring_finger4"), id=27, color=[255, 51, 51]
        ),
        28: dict(link=("right_wrist", "right_pinky_finger4"), id=28, color=[0, 255, 0]),
    },
    joint_weights=[1.0] * 27,
    # 'https://github.com/jin-s13/COCO-WholeBody/blob/master/'
    # 'evaluation/myeval_wholebody.py#L175'
    sigmas=[
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
        0.047,
        0.035,
        0.026,
        0.032,
        0.031,
        0.047,
        0.035,
        0.026,
        0.032,
        0.031,
    ],
)
