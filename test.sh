# ----- EfficientNet Model -----
# ISIC Model
python3 main.py --experiment isic --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic --temperature  1.0909483432769775;
python3 main.py --experiment isic --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic --temperature  1.0909483432769775;
python3 main.py --experiment isic --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic --temperature  1.0909483432769775;
python3 main.py --experiment isic --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic --temperature  1.0909483432769775;

# SD260 Model
python3 main.py --experiment sd260 --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260 --temperature 2.086536407470703;
python3 main.py --experiment sd260 --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260 --temperature 2.086536407470703;
python3 main.py --experiment sd260 --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260 --temperature 2.086536407470703;
python3 main.py --experiment sd260 --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260 --temperature 2.086536407470703;

# Tune ISIC Using SD260
python3 main.py --experiment isic_sd260 --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_sd260 --temperature 1.842851996421814;
python3 main.py --experiment isic_sd260 --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_sd260 --temperature 1.842851996421814;
python3 main.py --experiment isic_sd260 --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_sd260 --temperature 1.842851996421814;
python3 main.py --experiment isic_sd260 --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_sd260 --temperature 1.842851996421814;

# Tune SD260 Using ISIC
python3 main.py --experiment sd260_isic --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_isic --temperature 1.0351802110671997;
python3 main.py --experiment sd260_isic --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260_isic --temperature 1.0351802110671997;
python3 main.py --experiment sd260_isic --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_isic --temperature 1.0351802110671997;
python3 main.py --experiment sd260_isic --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_isic --temperature 1.0351802110671997;





# ----- SWIN MODEL -----
# ISIC Model
python3 main.py --experiment isic_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_swin --swin_model True --temperature 1.16023588180542;
python3 main.py --experiment isic_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_swin --swin_model True --temperature 1.16023588180542;
python3 main.py --experiment isic_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_swin --swin_model True --temperature 1.16023588180542;
python3 main.py --experiment isic_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_swin --swin_model True --temperature 1.16023588180542;

# SD260 Model
python3 main.py --experiment sd260_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_swin --swin_model True --temperature 3.108156681060791;
python3 main.py --experiment sd260_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260_swin --swin_model True --temperature 3.108156681060791;
python3 main.py --experiment sd260_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_swin --swin_model True --temperature 3.108156681060791;
python3 main.py --experiment sd260_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_swin --swin_model True --temperature 3.108156681060791;

# Tune ISIC Using SD260
python3 main.py --experiment isic_sd260_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model isic_sd260_swin --swin_model True --temperature 2.709517240524292;
python3 main.py --experiment isic_sd260_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model isic_sd260_swin --swin_model True --temperature 2.709517240524292;
python3 main.py --experiment isic_sd260_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model isic_sd260_swin --swin_model True --temperature 2.709517240524292;
python3 main.py --experiment isic_sd260_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model isic_sd260_swin --swin_model True --temperature 2.709517240524292;

# Tune SD260 Using ISIC
python3 main.py --experiment sd260_isic_swin --task test --dataset isic --dataset_dir ../../Datasets/ISIC_2019 --load_model sd260_isic_swin --swin_model True --temperature 1.0720878839492798;
python3 main.py --experiment sd260_isic_swin --task test --dataset sd260 --dataset_dir ../../Datasets/SD260 --load_model sd260_isic_swin --swin_model True --temperature 1.0720878839492798;
python3 main.py --experiment sd260_isic_swin --task test --dataset tayside --dataset_dir ../../Datasets/Dermatology/Tayside --load_model sd260_isic_swin --swin_model True --temperature 1.0720878839492798;
python3 main.py --experiment sd260_isic_swin --task test --dataset forth-valley --dataset_dir ../../Datasets/Dermatology/Forth_Valley_Macro --load_model sd260_isic_swin --swin_model True --temperature 1.0720878839492798;
