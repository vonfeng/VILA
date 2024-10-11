source /usr/local/anaconda3/bin/activate vila
export CUDA_VISIBLE_DEVICES=0

python -W ignore /data1/fengjie/CityGPTV/train/VILA/llava/eval/run_vila.py \
    --model-path /data3/wangshengyuan/models/Beijing-Old \
    --conv-mode llama_3 \
    --query "<image>\n Please describe the traffic condition." \
    --image-file "/data3/fengjie/streetview_images/paris_StreetView_Images/KMtGEWWuEl48suiYrPg3MQ&48.85851465585122&2.288922373991138&15&4.jpg"