# ----- EfficientNet Model -----
# ISIC Model
python3 main.py --experiment isic --task train --dataset isic --dataset_dir ../../Datasets/ISIC_2019;
python3 main.py --experiment isic --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019;
python3 main.py --experiment isic --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic;
python3 main.py --experiment isic --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic;
python3 main.py --experiment isic --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic;
python3 main.py --experiment isic --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic;

# SD260 Model
python3 main.py --experiment sd260 --task train --dataset sd260 --dataset_dir ../../Datasets/SD260;
python3 main.py --experiment sd260 --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019;
python3 main.py --experiment sd260 --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260;
python3 main.py --experiment sd260 --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260;
python3 main.py --experiment sd260 --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260;
python3 main.py --experiment sd260 --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260;

# Tayside Cross Validation
python3 main.py --experiment tayside --task train_cv --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model tayside --additional_dataset forth-valley --additional_dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro;

# Forth Valley Cross Validation
python3 main.py --experiment forth_valley --task train_cv --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model forth_valley --additional_dataset tayside --additional_dataset_dir ../../Datasets/Dermatology/Tayside;

# ISIC -> SD260 Model
python3 main.py --experiment isic_sd260 --task finetune --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic;
python3 main.py --experiment isic_sd260 --task temperature --dataset sd260 --dataset_dir ../../Datasets/SD260;
python3 main.py --experiment isic_sd260 --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_sd260;
python3 main.py --experiment isic_sd260 --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_sd260;
python3 main.py --experiment isic_sd260 --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_sd260;
python3 main.py --experiment isic_sd260 --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_sd260;

# SD-260 -> ISIC
python3 main.py --experiment sd260_isic --task finetune --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260;
python3 main.py --experiment sd260_isic --task temperature --dataset isic --dataset_dir ../../Datasets/ISIC_2019;
python3 main.py --experiment sd260_isic --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_isic;
python3 main.py --experiment sd260_isic --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260_isic;
python3 main.py --experiment sd260_isic --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_isic;
python3 main.py --experiment sd260_isic --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_isic;

# ISIC -> Tayside Cross Validation
python3 main.py --experiment isic_tayside --task tune_cv --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic --additional_dataset forth-valley --additional_dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro;

# ISIC -> Forth Valley Cross Validation
python3 main.py --experiment isic_forth_valley --task tune_cv --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic --additional_dataset tayside --additional_dataset_dir ../../Datasets/Dermatology/Tayside;

# SD-260 -> Tayside Cross Validation
python3 main.py --experiment sd260_tayside --task tune_cv --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260 --additional_dataset forth-valley --additional_dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro;

# SD-260 -> Forth Valley Cross Validation
python3 main.py --experiment sd260_forth_valley --task tune_cv --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260 --additional_dataset tayside --additional_dataset_dir ../../Datasets/Dermatology/Tayside;

# ISIC -> SD-260 -> Tayside Cross Validation
python3 main.py --experiment isic_sd260_tayside --task tune_cv --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_isic --additional_dataset forth-valley --additional_dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro;

# ISIC -> SD-260 -> Forth Valley Cross Validation
python3 main.py --experiment isic_sd260_forth_valley --task tune_cv --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_isic --additional_dataset tayside --additional_dataset_dir ../../Datasets/Dermatology/Tayside;
