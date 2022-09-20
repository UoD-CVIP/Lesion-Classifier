# ----- EfficientNet Model -----
# ISIC Model
python3 main.py --experiment isic --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic;
python3 main.py --experiment isic --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic;
python3 main.py --experiment isic --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic;
python3 main.py --experiment isic --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic;

# SD260 Model
python3 main.py --experiment sd260 --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260;
python3 main.py --experiment sd260 --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260;
python3 main.py --experiment sd260 --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260;
python3 main.py --experiment sd260 --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260;

# Tune ISIC Using SD260
python3 main.py --experiment isic_sd260 --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_sd260;
python3 main.py --experiment isic_sd260 --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_sd260;
python3 main.py --experiment isic_sd260 --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_sd260;
python3 main.py --experiment isic_sd260 --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_sd260;

# Tune SD260 Using ISIC
python3 main.py --experiment sd260_isic --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_isic;
python3 main.py --experiment sd260_isic --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260_isic;
python3 main.py --experiment sd260_isic --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_isic;
python3 main.py --experiment sd260_isic --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_isic;





# ----- SWIN MODEL -----
# ISIC Model
python3 main.py --experiment isic_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_swin --swin_model True;
python3 main.py --experiment isic_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_swin --swin_model True;
python3 main.py --experiment isic_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_swin --swin_model True;
python3 main.py --experiment isic_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_swin --swin_model True;

# SD260 Model
python3 main.py --experiment sd260_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_swin --swin_model True;
python3 main.py --experiment sd260_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260_swin --swin_model True;
python3 main.py --experiment sd260_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_swin --swin_model True;
python3 main.py --experiment sd260_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_swin --swin_model True;

# Tune ISIC Using SD260
python3 main.py --experiment isic_sd260_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_sd260_swin --swin_model True;
python3 main.py --experiment isic_sd260_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_sd260_swin --swin_model True;
python3 main.py --experiment isic_sd260_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_sd260_swin --swin_model True;
python3 main.py --experiment isic_sd260_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_sd260_swin --swin_model True;

# Tune SD260 Using ISIC
python3 main.py --experiment sd260_isic_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_isic_swin --swin_model True;
python3 main.py --experiment sd260_isic_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260_isic_swin --swin_model True;
python3 main.py --experiment sd260_isic_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_isic_swin --swin_model True;
python3 main.py --experiment sd260_isic_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_isic_swin --swin_model True;
