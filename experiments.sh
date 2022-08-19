# ----- EfficientNet Model -----
# ISIC Model
python3 main.py --experiment isic --task train --dataset isic --dataset_dir ../../Datasets/ISIC_2019;
python3 main.py --experiment isic --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic;
python3 main.py --experiment isic --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic;
python3 main.py --experiment isic --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic;
python3 main.py --experiment isic --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic;

# SD260 Model
python3 main.py --experiment sd260 --task train --dataset sd260 --dataset_dir ../../Datasets/SD260;
python3 main.py --experiment sd260 --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260;
python3 main.py --experiment sd260 --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260;
python3 main.py --experiment sd260 --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260;
python3 main.py --experiment sd260 --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260;

# Tayside Cross Validation
python3 main.py --experiment tayside_cv --task train_cv --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model tayside_cv;

# Forth Valley Cross Validation
python3 main.py --experiment forth_valley_cv --task train_cv --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model forth_valley_cv;

# Tune ISIC Using SD260
python3 main.py --experiment isic_sd260 --task finetune --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic;
python3 main.py --experiment isic_sd260 --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_sd260;
python3 main.py --experiment isic_sd260 --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_sd260;
python3 main.py --experiment isic_sd260 --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_sd260;
python3 main.py --experiment isic_sd260 --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_sd260;

# Tune ISIC Using Tayside Cross Validation
python3 main.py --experiment isic_tayside --task tune_cv --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic;

# Tune ISIC Using Forth Valley Cross Validation
python3 main.py --experiment isic_forth_valley --task tune_cv --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic;

# Tune SD260 Using ISIC
python3 main.py --experiment sd260_isic --task finetune --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260;
python3 main.py --experiment sd260_isic --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_isic;
python3 main.py --experiment sd260_isic --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260_isic;
python3 main.py --experiment sd260_isic --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_isic;
python3 main.py --experiment sd260_isic --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_isic;

# Tune SD260 Using Tayside Cross Validation
python3 main.py --experiment sd260_tayside --task tune_cv --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260;

# Tune SD260 Using Forth Valley Cross Validation
python3 main.py --experiment sd260_forth_valley --task tune_cv --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260;





# ----- SWIN MODEL -----
# ISIC Model
python3 main.py --experiment isic_swin --task train --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --swin_model True;
python3 main.py --experiment isic_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_swin --swin_model True;
python3 main.py --experiment isic_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_swin --swin_model True;
python3 main.py --experiment isic_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_swin --swin_model True;
python3 main.py --experiment isic_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_swin --swin_model True;

# SD260 Model
python3 main.py --experiment sd260_swin --task train --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True;
python3 main.py --experiment sd260_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_swin --swin_model True;
python3 main.py --experiment sd260_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260_swin --swin_model True;
python3 main.py --experiment sd260_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_swin --swin_model True;
python3 main.py --experiment sd260_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_swin --swin_model True;

# Tayside Cross Validation
python3 main.py --experiment tayside_cv_swin --task train_cv --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model tayside_cv_swin --swin_model True;

# Forth Valley Cross Validation
python3 main.py --experiment forth_valley_cv_swin --task train_cv --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model forth_valley_cv_swin --swin_model True;

# Tune ISIC Using SD260
python3 main.py --experiment isic_sd260_swin --task finetune --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_swin --swin_model True;
python3 main.py --experiment isic_sd260_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_sd260_swin --swin_model True;
python3 main.py --experiment isic_sd260_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_sd260_swin --swin_model True;
python3 main.py --experiment isic_sd260_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_sd260_swin --swin_model True;
python3 main.py --experiment isic_sd260_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_sd260_swin --swin_model True;

# Tune ISIC Using Tayside Cross Validation
python3 main.py --experiment isic_tayside_swin --task tune_cv --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_swin --swin_model True;

# Tune ISIC Using Forth Valley Cross Validation
python3 main.py --experiment isic_forth_valley_swin --task tune_cv --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_swin --swin_model True;

# Tune SD260 Using ISIC
python3 main.py --experiment sd260_isic_swin --task finetune --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_swin --swin_model True;
python3 main.py --experiment sd260_isic_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_isic_swin --swin_model True;
python3 main.py --experiment sd260_isic_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260_isic_swin --swin_model True;
python3 main.py --experiment sd260_isic_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_isic_swin --swin_model True;
python3 main.py --experiment sd260_isic_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_isic_swin --swin_model True;

# Tune SD260 Using Tayside Cross Validation
python3 main.py --experiment sd260_tayside_swin --task tune_cv --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_swin --swin_model True;

# Tune SD260 Using Forth Valley Cross Validation
python3 main.py --experiment sd260_forth_valley_swin --task tune_cv --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_swin --swin_model True;