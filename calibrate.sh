# ----- EfficientNet Model -----
# ISIC Model
python3 main.py --experiment isic --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019;

# SD-260 Model
python3 main.py --experiment sd260 --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019;

# Tune ISIC Using SD260
python3 main.py --experiment isic_sd260 --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019;

# Tune SD260 Using ISIC
python3 main.py --experiment sd260_isic --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019;

# ----- SWIN MODEL -----
# ISIC Model
python3 main.py --experiment isic_swin --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_swin --swin_model True;

# SD260 Model
python3 main.py --experiment sd260_swin --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_swin --swin_model True;

# Tune ISIC Using SD260
python3 main.py --experiment isic_sd260_swin --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_sd260_swin --swin_model True;

# Tune SD260 Using ISIC
python3 main.py --experiment sd260_isic_swin --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_isic_swin --swin_model True;