# GEOMETRY VALIDATION REPORT (real data)

Reference dir: `C:\Users\Student\Desktop\particleTracking\Data/extracted/reference`  
Ball dir: `C:\Users\Student\Desktop\particleTracking\Data/extracted/ball_static`  
Box dir: `C:\Users\Student\Desktop\particleTracking\Data/extracted/box_static`  
Every Nth frame processed: 1


## PLANE

- Frames: attempted=825, succeeded=825, failed=0 (0.0% failure)
- Residual (mm): mean=1.0837, median=0.9319, rmse=1.3454, p95=2.5971, max=9.7551


## BALL

- Frames: attempted=1117, succeeded=1117, failed=0 (0.0% failure)
- Surface residual (mm): mean=0.6924, median=0.5584, rmse=0.8928, p95=1.7988, max=4.0354
- Estimated diameter: mean=60.664mm, std=0.6401mm

- Physical ground-truth accuracy not calculated because physical dimensions were not supplied.


## BOX

- Frames: attempted=1027, succeeded=1027, failed=0 (0.0% failure)
- Plane-set fitting RMSE (mm): mean=0.9554, median=0.9101, p95=1.1551, max=1.2211
- NOTE: a prior synthetic diagnostic found ~1.32mm RMSE at ZERO synthetic noise for this method. Compare against the real median above yourself.

- Physical ground-truth accuracy not calculated because physical dimensions were not supplied.
