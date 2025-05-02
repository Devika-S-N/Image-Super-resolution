python video_super_res.py --decode

timeout 5

python inference_final.py --video-mode --model swinir

timeout 5

python video_super_res.py --encode

pause