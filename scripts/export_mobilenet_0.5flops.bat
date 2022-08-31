python amc_search.py ^
    --job=export ^
    --model=mobilenet ^
    --dataset=imagenet ^
    --data_root=<Enter path to the dataset here> ^
    --ckpt_path=./checkpoints/mobilenet_imagenet.pth.tar ^
    --seed=2018 ^
    --n_calibration_batches=100 ^
    --n_worker=16 ^
    --channels=3,24,48,96,80,192,200,328,352,368,360,328,400,736,752 ^
    --export_path=./checkpoints/mobilenet_0.5flops_export.pth.tar ^
    --n_gpu=0

