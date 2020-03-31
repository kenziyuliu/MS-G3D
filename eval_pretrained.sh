## Generate test scores

# NTU 60 XSub
python3 main.py --config ./config/nturgbd-cross-subject/test_joint.yaml --work-dir pretrain_eval/ntu60/xsub/joint-fusion --weights pretrained-models/ntu60-xsub-joint-fusion.pt

python3 main.py --config ./config/nturgbd-cross-subject/test_bone.yaml --work-dir pretrain_eval/ntu60/xsub/bone --weights pretrained-models/ntu60-xsub-bone.pt


# NTU 60 XView
python3 main.py --config ./config/nturgbd-cross-view/test_joint.yaml --work-dir pretrain_eval/ntu60/xview/joint --weights pretrained-models/ntu60-xview-joint.pt

python3 main.py --config ./config/nturgbd-cross-view/test_bone.yaml --work-dir pretrain_eval/ntu60/xview/bone --weights pretrained-models/ntu60-xview-bone.pt


# NTU 120 XSub
python3 main.py --config ./config/nturgbd120-cross-subject/test_joint.yaml --work-dir pretrain_eval/ntu120/xsub/joint --weights pretrained-models/ntu120-xsub-joint.pt

python3 main.py --config ./config/nturgbd120-cross-subject/test_bone.yaml --work-dir pretrain_eval/ntu120/xsub/bone --weights pretrained-models/ntu120-xsub-bone.pt


# NTU 120 XSet
python3 main.py --config ./config/nturgbd120-cross-setup/test_joint.yaml --work-dir pretrain_eval/ntu120/xset/joint --weights pretrained-models/ntu120-xset-joint.pt

python3 main.py --config ./config/nturgbd120-cross-setup/test_bone.yaml --work-dir pretrain_eval/ntu120/xset/bone --weights pretrained-models/ntu120-xset-bone.pt


# Kinetics Skeleton 400
python3 main.py --config ./config/kinetics-skeleton/test_joint.yaml --work-dir pretrain_eval/kinetics/joint --weights pretrained-models/kinetics-joint.pt

python3 main.py --config ./config/kinetics-skeleton/test_bone.yaml --work-dir pretrain_eval/kinetics/bone --weights pretrained-models/kinetics-bone.pt



## Perform all ensembles at once

# NTU 60 XSub
printf "\nNTU RGB+D 60 XSub\n"
python3 ensemble.py --dataset ntu/xsub --joint-dir pretrain_eval/ntu60/xsub/joint-fusion --bone-dir pretrain_eval/ntu60/xsub/bone

# NTU 60 XView
printf "\nNTU RGB+D 60 XView\n"
python3 ensemble.py --dataset ntu/xview --joint-dir pretrain_eval/ntu60/xview/joint --bone-dir pretrain_eval/ntu60/xview/bone

# NTU 120 XSub
printf "\nNTU RGB+D 120 XSub\n"
python3 ensemble.py --dataset ntu120/xsub --joint-dir pretrain_eval/ntu120/xsub/joint --bone-dir pretrain_eval/ntu120/xsub/bone

# NTU 120 XSet
printf "\nNTU RGB+D 120 XSet\n"
python3 ensemble.py --dataset ntu120/xset --joint-dir pretrain_eval/ntu120/xset/joint --bone-dir pretrain_eval/ntu120/xset/bone

# Kinetics Skeleton 400
printf "\nKinetics Skeleton 400\n"
python3 ensemble.py --dataset kinetics --joint-dir pretrain_eval/kinetics/joint --bone-dir pretrain_eval/kinetics/bone