python video_super_res.py --decode

timeout 5

python inference_final.py --train-eval --model swinir

timeout 5

python video_super_res.py --encode

pause