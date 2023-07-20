# ----- EfficientNet Model -----
# ISIC Model
#python3 main.py --swin_model True --experiment isic_swin --task train --dataset isic --dataset_dir ../Datasets/ISIC_2019;
python3 main.py --swin_model True --experiment isic_swin --task temperature --dataset isic --dataset_dir ../Datasets/ISIC_2019;
python3 main.py --swin_model True --experiment isic_swin --task test --dataset isic --dataset_dir ../Datasets/ISIC_2019 --load_model isic_swin;
python3 main.py --swin_model True --experiment isic_swin --task test --dataset sd260 --dataset_dir ../Datasets/SD260 --load_model isic_swin;
python3 main.py --swin_model True --experiment isic_swin --task test --dataset tayside --dataset_dir ../Datasets/Dermatology/Tayside --load_model isic_swin;
python3 main.py --swin_model True --experiment isic_swin --task test --dataset forth-valley --dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_swin;

# SD260 Model
python3 main.py --swin_model True --experiment sd260_swin --task train --dataset sd260 --dataset_dir ../Datasets/SD260;
python3 main.py --swin_model True --experiment sd260_swin --task temperature --dataset isic --dataset_dir ../Datasets/ISIC_2019;
python3 main.py --swin_model True --experiment sd260_swin --task test --dataset isic --dataset_dir ../Datasets/ISIC_2019 --load_model sd260_swin;
python3 main.py --swin_model True --experiment sd260_swin --task test --dataset sd260 --dataset_dir ../Datasets/SD260 --load_model sd260_swin;
python3 main.py --swin_model True --experiment sd260_swin --task test --dataset tayside --dataset_dir ../Datasets/Dermatology/Tayside --load_model sd260_swin;
python3 main.py --swin_model True --experiment sd260_swin --task test --dataset forth-valley --dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_swin;

# Tayside Cross Validation
python3 main.py --swin_model True --experiment tayside_swin --task train_cv --dataset tayside --dataset_dir ../Datasets/Dermatology/Tayside --load_model tayside_swin --additional_dataset forth-valley --additional_dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro;

# Forth Valley Cross Validation
python3 main.py --swin_model True --experiment forth_valley_swin --task train_cv --dataset forth-valley --dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro --load_model forth_valley_swin --additional_dataset tayside --additional_dataset_dir ../Datasets/Dermatology/Tayside;

# ISIC -> SD260 Model
python3 main.py --swin_model True --experiment isic_sd260_swin --task finetune --dataset sd260 --dataset_dir ../Datasets/SD260 --load_model isic_swin;
python3 main.py --swin_model True --experiment isic_sd260_swin --task temperature --dataset sd260 --dataset_dir ../Datasets/SD260;
python3 main.py --swin_model True --experiment isic_sd260_swin --task test --dataset isic --dataset_dir ../Datasets/ISIC_2019 --load_model isic_sd260_swin --temperature;
python3 main.py --swin_model True --experiment isic_sd260_swin --task test --dataset sd260 --dataset_dir ../Datasets/SD260 --load_model isic_sd260_swin --temperature;
python3 main.py --swin_model True --experiment isic_sd260_swin --task test --dataset tayside --dataset_dir ../Datasets/Dermatology/Tayside --load_model isic_sd260_swin;
python3 main.py --swin_model True --experiment isic_sd260_swin --task test --dataset forth-valley --dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_sd260_swin;

# SD-260 -> ISIC
python3 main.py --swin_model True --experiment sd260_isic_swin --task finetune --dataset isic --dataset_dir ../Datasets/ISIC_2019 --load_model sd260_swin;
python3 main.py --swin_model True --experiment sd260_isic_swin --task temperature --dataset isic --dataset_dir ../Datasets/ISIC_2019;
python3 main.py --swin_model True --experiment sd260_isic_swin --task test --dataset isic --dataset_dir ../Datasets/ISIC_2019 --load_model sd260_isic_swin;
python3 main.py --swin_model True --experiment sd260_isic_swin --task test --dataset sd260 --dataset_dir ../Datasets/SD260 --load_model sd260_isic_swin;
python3 main.py --swin_model True --experiment sd260_isic_swin --task test --dataset tayside --dataset_dir ../Datasets/Dermatology/Tayside --load_model sd260_isic_swin;
python3 main.py --swin_model True --experiment sd260_isic_swin --task test --dataset forth-valley --dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_isic_swin;

# ISIC -> Tayside Cross Validation
python3 main.py --swin_model True --experiment isic_tayside_swin --task tune_cv --dataset tayside --dataset_dir ../Datasets/Dermatology/Tayside --load_model isic_swin --additional_dataset forth-valley --additional_dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro;

# ISIC -> Forth Valley Cross Validation
python3 main.py --swin_model True --experiment isic_forth_valley_swin --task tune_cv --dataset forth-valley --dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_swin --additional_dataset tayside --additional_dataset_dir ../Datasets/Dermatology/Tayside;

# SD-260 -> Tayside Cross Validation
python3 main.py --swin_model True --experiment sd260_tayside_swin --task tune_cv --dataset tayside --dataset_dir ../Datasets/Dermatology/Tayside --load_model sd260_swin --additional_dataset forth-valley --additional_dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro;

# SD-260 -> Forth Valley Cross Validation
python3 main.py --swin_model True --experiment sd260_forth_valley_swin --task tune_cv --dataset forth-valley --dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_swin --additional_dataset tayside --additional_dataset_dir ../Datasets/Dermatology/Tayside;

# ISIC -> SD-260 -> Tayside Cross Validation
python3 main.py --swin_model True --experiment isic_sd260_tayside_swin --task tune_cv --dataset tayside --dataset_dir ../Datasets/Dermatology/Tayside --load_model sd260_isic_swin --additional_dataset forth-valley --additional_dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro;

# ISIC -> SD-260 -> Forth Valley Cross Validation
python3 main.py --swin_model True --experiment isic_sd260_forth_valley_swin --task tune_cv --dataset forth-valley --dataset_dir ../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_isic_swin --additional_dataset tayside --additional_dataset_dir ../Datasets/Dermatology/Tayside;
